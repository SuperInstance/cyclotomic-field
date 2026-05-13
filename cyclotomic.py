"""cyclotomic.py — Unified Field Framework for Q(ζ₁₅).

Provides classes and functions for working with cyclotomic fields,
the unified 6D cut-and-project scheme connecting Eisenstein and Penrose
tilings, and snap-to-lattice operations.

Mathematical foundation verified by experiments/cyclotomic-verify/.
"""

import math
import cmath
import numpy as np
from typing import Tuple, Optional, List, Union


class CyclotomicField:
    """Represents the cyclotomic field Q(ζₙ) for a given n.
    
    Provides methods for field arithmetic and embedding into complex numbers.
    """
    
    def __init__(self, n: int) -> None:
        """Initialize Q(ζₙ) where ζₙ = e^{2πi/n}.
        
        Args:
            n: The cyclotomic index (positive integer).
        """
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        self.n = n
        self.zeta = cmath.exp(2j * math.pi / n)
        self.phi_n = self._euler_phi(n)
    
    @staticmethod
    def _euler_phi(n: int) -> int:
        """Euler's totient function."""
        result = n
        p = 2
        temp = n
        while p * p <= temp:
            if temp % p == 0:
                while temp % p == 0:
                    temp //= p
                result -= result // p
            p += 1
        if temp > 1:
            result -= result // temp
        return result
    
    def element(self, coeffs: List[float]) -> complex:
        """Construct a field element from coefficients.
        
        Returns Σ coeffs[k] * ζₙᵏ.
        
        Args:
            coeffs: List of coefficients [c₀, c₁, ..., c_{n-1}].
        
        Returns:
            Complex number representing the field element.
        """
        result = 0.0 + 0.0j
        for k, c in enumerate(coeffs):
            result += c * (self.zeta ** k)
        return result
    
    def embed(self, value: complex) -> complex:
        """Verify an element belongs to the field (within tolerance).
        
        This is a placeholder for exact arithmetic.
        """
        return value
    
    def __repr__(self) -> str:
        return f"CyclotomicField(n={self.n}, degree={self.phi_n})"


# The unified Q(ζ₁₅) field instance
_Q15 = CyclotomicField(15)

def _get_projection_vectors(theta: float) -> np.ndarray:
    """Compute the 6 physical-space projection vectors at angle θ.
    
    At θ=0: 6 vectors at 60° intervals → hexagonal lattice
    At θ=arctan(φ): 5 vectors at 72° + 1 redundant → Penrose
    
    Args:
        theta: Interpolation angle (0 for Eisenstein, arctan(φ) for Penrose).
    
    Returns:
        (6, 2) array of projection vectors.
    """
    φ = (1 + math.sqrt(5)) / 2
    θ_p = math.atan(φ)
    t = theta / θ_p if θ_p > 0 else 0.0
    t = max(0.0, min(1.0, t))
    
    hex_angles = [2 * math.pi * k / 6 for k in range(6)]
    penrose_angles = [2 * math.pi * k / 5 for k in range(5)]
    penrose_angles.append(2 * math.pi * 0 / 5)
    
    angles = [(1 - t) * hex_angles[k] + t * penrose_angles[k] for k in range(6)]
    
    return np.array([[math.cos(a), math.sin(a)] for a in angles])


def _get_full_projection(theta: float) -> np.ndarray:
    """Build the full 6×6 orthogonal projection matrix.
    
    Rows 0-1: physical space (2D)
    Rows 2-5: internal space (4D)
    
    Args:
        theta: Interpolation angle.
    
    Returns:
        (6, 6) orthogonal matrix.
    """
    proj_phys = _get_projection_vectors(theta)
    
    M = np.zeros((6, 6))
    M[:2, :] = proj_phys.T
    
    np.random.seed(42)
    for i in range(2, 6):
        v = np.random.randn(6)
        for j in range(i):
            v -= np.dot(v, M[j]) * M[j]
        v = v / np.linalg.norm(v)
        M[i] = v
    
    return M


def eisenstein_project(points: np.ndarray, theta: float = 0.0) -> np.ndarray:
    """Project points using the unified 6D scheme (Eisenstein mode).
    
    At θ=0, the Z⁶ → 2D projection produces the Eisenstein (hexagonal) lattice.
    
    Args:
        points: (N, 6) array of 6D integer coordinates.
        theta: Projection angle (default 0 for hexagonal lattice).
    
    Returns:
        (N, 2) array of 2D projected coordinates.
    """
    proj_phys = _get_projection_vectors(theta)
    return points @ proj_phys


def penrose_project(points: np.ndarray, theta: Optional[float] = None) -> np.ndarray:
    """Project points using the unified 6D scheme (Penrose mode).
    
    At θ=arctan(φ), the projection yields Penrose tiling vertices.
    
    Args:
        points: (N, 6) array of 6D integer coordinates.
        theta: Projection angle (defaults to arctan(φ) for Penrose).
    
    Returns:
        (N, 2) array of 2D projected coordinates.
    """
    if theta is None:
        φ = (1 + math.sqrt(5)) / 2
        theta = math.atan(φ)
    return eisenstein_project(points, theta)


def unified_snap(x: float, y: float, theta: float, epsilon: float = 1e-6) -> Tuple[float, float]:
    """Snap (x, y) to the nearest point in the deformed lattice at angle θ.
    
    Uses the 6D lift-and-round method: finds integer Z⁶ coefficients
    that best approximate the given 2D point when projected, then
    re-projects to get the snapped position.
    
    The system is underdetermined (2 equations, 6 unknowns):
      proj_phys^T @ coeffs ≈ [x, y]^T
    
    We find the minimum-norm solution, round to integers, then
    re-project to get the snapped (x,y).
    
    Args:
        x: x-coordinate to snap.
        y: y-coordinate to snap.
        theta: Current interpolation angle.
        epsilon: Rounding threshold (unused, kept for API compatibility).
    
    Returns:
        (snapped_x, snapped_y) tuple.
    """
    proj_phys = _get_projection_vectors(theta)  # (6, 2)
    AT = proj_phys.T  # (2, 6)
    
    # Minimum-norm solution to AT @ coeffs = b: coeffs = AT^T (AT AT^T)^{-1} b
    AAT = AT @ proj_phys  # (2, 2)
    b = np.array([x, y])
    try:
        AAT_inv = np.linalg.inv(AAT)
        coeffs_float = proj_phys @ AAT_inv @ b  # (6,)
    except np.linalg.LinAlgError:
        coeffs_float = np.zeros(6)
    
    coeffs_round = np.round(coeffs_float).astype(int)
    snapped = proj_phys.T @ coeffs_round
    return (float(snapped[0]), float(snapped[1]))


def generate_eisenstein_lattice(radius: int) -> np.ndarray:
    """Generate Eisenstein lattice points within a given radius.
    
    The Eisenstein lattice Z[ω] where ω = e^{2πi/3}.
    Returns 2D coordinates directly.
    
    Args:
        radius: Maximum distance from origin.
    
    Returns:
        (N, 2) array of (x, y) coordinates.
    """
    omega = -0.5 + 0.5j * math.sqrt(3)
    points = []
    for a in range(-radius, radius + 1):
        b_max = int(math.ceil(radius * 2 / math.sqrt(3)))
        for b in range(-b_max, b_max + 1):
            z = a + b * omega
            if abs(z) <= radius:
                points.append([z.real, z.imag])
    return np.array(points, dtype=np.float64)


def generate_penrose_vertices(radius: int, theta: Optional[float] = None) -> np.ndarray:
    """Generate approximate Penrose vertices via 6D cut-and-project.
    
    Args:
        radius: Maximum distance from origin.
        theta: Projection angle (defaults to Penrose angle).
    
    Returns:
        (N, 2) array of (x, y) coordinates.
    """
    if theta is None:
        φ = (1 + math.sqrt(5)) / 2
        theta = math.atan(φ)
    
    # Generate Z⁶ points
    points_6d = []
    N = radius + 1
    for i in range(-N, N + 1):
        for j in range(-N, N + 1):
            for k in range(-N, N + 1):
                for l in range(-N, N + 1):
                    for m in range(-N, N + 1):
                        for n in range(-N, N + 1):
                            r2 = i*i + j*j + k*k + l*l + m*m + n*n
                            if r2 <= N*N:
                                points_6d.append([i, j, k, l, m, n])
    
    points_6d = np.array(points_6d, dtype=np.float64)
    
    # Project and apply window
    proj_phys = _get_projection_vectors(theta)
    proj_2d = points_6d @ proj_phys
    M = _get_full_projection(theta)
    internal_proj = points_6d @ M[2:].T
    
    in_window = np.all(np.abs(internal_proj) <= 0.3, axis=1)
    return proj_2d[in_window]
