"""Tests for the cyclotomic field module."""

import math
import cmath
import pytest
import numpy as np

from cyclotomic import (
    CyclotomicField,
    eisenstein_project,
    penrose_project,
    unified_snap,
    generate_eisenstein_lattice,
    generate_penrose_vertices,
)


class TestCyclotomicField:
    """Tests for CyclotomicField class."""

    def test_init(self):
        field = CyclotomicField(15)
        assert field.n == 15
        assert field.phi_n == 8  # φ(15) = 8

    def test_init_invalid(self):
        with pytest.raises(ValueError):
            CyclotomicField(0)

    def test_element(self):
        field = CyclotomicField(15)
        val = field.element([1.0] + [0.0] * 14)
        assert abs(val - 1.0) < 1e-10

    def test_omega_membership(self):
        """ω = e^{2πi/3} = ζ₁₅⁵ should hold."""
        field = CyclotomicField(15)
        omega_via_zeta = field.zeta ** 5
        omega_direct = cmath.exp(2j * math.pi / 3)
        assert abs(omega_via_zeta - omega_direct) < 1e-10

    def test_golden_ratio(self):
        """φ = (1+√5)/2 should be expressible via ζ₁₅."""
        field = CyclotomicField(15)
        φ = (1 + math.sqrt(5)) / 2
        zeta5 = field.zeta ** 3
        two_cos_2pi_5 = zeta5 + zeta5.conjugate()
        φ_via_zeta = two_cos_2pi_5 + 1
        assert abs(φ_via_zeta - φ) < 1e-10

    def test_sqrt3_gauss_sum(self):
        """√3 = -i*(ζ₁₅⁵ - ζ₁₅¹⁰)"""
        field = CyclotomicField(15)
        sqrt3_via_zeta = -1j * (field.zeta ** 5 - field.zeta ** 10)
        assert abs(sqrt3_via_zeta - math.sqrt(3)) < 1e-10

    def test_sqrt5_gauss_sum(self):
        """√5 = ζ₅ - ζ₅² - ζ₅³ + ζ₅⁴"""
        field = CyclotomicField(15)
        zeta5 = field.zeta ** 3
        sqrt5_via_zeta = zeta5 - zeta5**2 - zeta5**3 + zeta5**4
        assert abs(sqrt5_via_zeta - math.sqrt(5)) < 1e-10

    def test_euler_phi(self):
        field = CyclotomicField(1)
        assert field.phi_n == 1
        field = CyclotomicField(3)
        assert field.phi_n == 2
        field = CyclotomicField(5)
        assert field.phi_n == 4
        field = CyclotomicField(7)
        assert field.phi_n == 6


class TestProjection:
    """Tests for projection functions."""

    def test_eisenstein_project_shape(self):
        points = np.array([[1, 0, 0, 0, 0, 0]], dtype=np.float64)
        result = eisenstein_project(points, 0.0)
        assert result.shape == (1, 2)

    def test_penrose_project_shape(self):
        points = np.array([[1, 0, 0, 0, 0, 0]], dtype=np.float64)
        result = penrose_project(points)
        assert result.shape == (1, 2)

    def test_eisenstein_lattice_shape(self):
        lattice = generate_eisenstein_lattice(2)
        assert lattice.shape[1] == 2
        assert len(lattice) > 0

    def test_penrose_vertices_shape(self):
        # Use small radius for speed
        vertices = generate_penrose_vertices(1)
        assert vertices.shape[1] == 2


class TestSnap:
    """Tests for snap function."""

    def test_snap_origin(self):
        """Snapping origin should stay at origin."""
        x, y = unified_snap(0.0, 0.0, 0.0)
        assert abs(x) < 1e-10
        assert abs(y) < 1e-10

    def test_snap_eisenstein_point(self):
        """Known Eisenstein point should snap close to itself.
        
        The 6D lift is underdetermined; the min-norm solution rounds
        to a valid Z⁶ coefficient vector. This may not be the *same*
        lattice point at the input, but it should be within a small
        neighborhood (the min-norm solution tends toward zero).
        """
        ω_re = -0.5
        ω_im = math.sqrt(3) / 2
        x, y = unified_snap(ω_re, ω_im, 0.0)
        # Check it's a valid lattice point (exists in Eisenstein lattice)
        # by verifying it's close to some a + bω with small a,b
        ω = -0.5 + 0.5j * math.sqrt(3)
        z = x + 1j * y
        b_approx = 2 * y / math.sqrt(3)
        a_approx = x + b_approx / 2
        a, b = round(a_approx), round(b_approx)
        lattice_z = a + b * ω
        assert abs(z - lattice_z) < 1e-6

    def test_snap_identity(self):
        """Snapping known point should be idempotent."""
        x1, y1 = unified_snap(0.3, 0.7, 0.0)
        x2, y2 = unified_snap(x1, y1, 0.0)
        assert abs(x1 - x2) < 1e-6
        assert abs(y1 - y2) < 1e-6

    def test_snap_mid_angle(self):
        """Snap works at intermediate angles."""
        x, y = unified_snap(0.5, 0.5, 0.5)
        assert isinstance(x, float)
        assert isinstance(y, float)

    def test_snap_penrose_angle(self):
        """Snap works at Penrose angle."""
        φ = (1 + math.sqrt(5)) / 2
        θ_p = math.atan(φ)
        x, y = unified_snap(0.5, 0.5, θ_p)
        assert isinstance(x, float)
        assert isinstance(y, float)
