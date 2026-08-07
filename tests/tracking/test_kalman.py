"""Tests for tandemmp.tracking.kalman — Kalman filter."""

from __future__ import annotations

import numpy as np
import pytest

from tandemmp.tracking.kalman import KalmanFilter

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestKalmanFilterInit:
    def test_basic_construction(self):
        x = np.array([1.0, 2.0, 3.0])
        p = np.eye(3) * 0.1
        kf = KalmanFilter(X=x, P=p)
        np.testing.assert_array_equal(kf.X, x)
        np.testing.assert_array_equal(kf.P, p)

    def test_defaults(self):
        x = np.array([1.0, 2.0])
        p = np.eye(2)
        kf = KalmanFilter(X=x, P=p)
        np.testing.assert_array_equal(kf.F, np.eye(2))
        np.testing.assert_array_equal(kf.H, np.eye(2))
        np.testing.assert_array_equal(kf.B, np.eye(2))
        np.testing.assert_array_equal(kf.Q, np.eye(2))
        np.testing.assert_array_equal(kf.u, np.zeros(2))

    def test_custom_matrices(self):
        x = np.array([0.0, 0.0])
        p = np.eye(2)
        f = np.array([[1, 1], [0, 1]], dtype=float)
        h = np.array([[1, 0], [0, 1]], dtype=float)
        b = np.zeros((2, 2))
        q = np.eye(2) * 0.01
        kf = KalmanFilter(X=x, P=p, F=f, H=h, B=b, Q=q)
        np.testing.assert_array_equal(kf.F, f)
        np.testing.assert_array_equal(kf.H, h)
        np.testing.assert_array_equal(kf.B, b)
        np.testing.assert_array_equal(kf.Q, q)

    def test_dimension_mismatch_raises(self):
        x = np.array([1.0, 2.0, 3.0])
        p = np.eye(2)  # wrong size
        with pytest.raises(ValueError, match="dimension"):
            KalmanFilter(X=x, P=p)

    def test_stores_initial_state(self):
        x = np.array([5.0, 6.0])
        p = np.eye(2) * 2.0
        kf = KalmanFilter(X=x, P=p)
        np.testing.assert_array_equal(kf.X, x)
        np.testing.assert_array_equal(kf.P, p)


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------


class TestKalmanFilterPredict:
    def test_identity_transition(self):
        """With F=I and Q=0, predict should not change X."""
        x = np.array([1.0, 2.0])
        p = np.eye(2) * 0.1
        kf = KalmanFilter(X=x, P=p, Q=np.zeros((2, 2)))
        kf.predict()
        np.testing.assert_array_equal(kf.X, x)

    def test_covariance_grows(self):
        """After predict, covariance should grow by Q."""
        x = np.array([1.0, 2.0])
        p = np.eye(2) * 0.1
        q = np.eye(2) * 0.5
        kf = KalmanFilter(X=x, P=p, Q=q)
        kf.predict()
        expected_p = p + q  # F=I, so F@P@F.T + Q = P + Q
        np.testing.assert_allclose(kf.P, expected_p)

    def test_state_transition(self):
        """F != I should transform the state."""
        x = np.array([1.0, 0.5])
        p = np.eye(2) * 0.1
        f = np.array([[1.0, 1.0], [0.0, 1.0]])  # position + velocity model
        kf = KalmanFilter(X=x, P=p, F=f, Q=np.zeros((2, 2)))
        kf.predict()
        # x_new = F @ x = [1 + 0.5, 0.5] = [1.5, 0.5]
        np.testing.assert_allclose(kf.X, [1.5, 0.5])

    def test_control_input(self):
        """Predict with a control input u."""
        x = np.array([0.0, 0.0])
        p = np.eye(2) * 0.1
        b = np.eye(2)
        kf = KalmanFilter(X=x, P=p, B=b, Q=np.zeros((2, 2)))
        kf.predict(new_u=np.array([1.0, 2.0]))
        np.testing.assert_allclose(kf.X, [1.0, 2.0])


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------


class TestKalmanFilterUpdate:
    def test_update_moves_toward_measurement(self):
        """After update, X should move toward the measurement."""
        x = np.array([0.0, 0.0])
        p = np.eye(2) * 10.0  # high uncertainty
        kf = KalmanFilter(X=x, P=p)
        z = np.array([5.0, 5.0])
        r = np.eye(2) * 0.01  # very certain measurement
        kf.update(z, r)
        # With high prior uncertainty and low measurement noise,
        # the filter should snap close to the measurement
        np.testing.assert_allclose(kf.X, z, atol=0.5)

    def test_update_shrinks_covariance(self):
        """After update, the covariance should shrink."""
        x = np.array([0.0, 0.0])
        p = np.eye(2) * 10.0
        kf = KalmanFilter(X=x, P=p)
        z = np.array([1.0, 1.0])
        r = np.eye(2) * 1.0
        kf.update(z, r)
        # Diagonal elements should be less than original
        assert np.all(np.diag(kf.P) < np.diag(p))

    def test_certain_measurement(self):
        """Measurement with near-zero noise should override prior."""
        x = np.array([10.0, 10.0])
        p = np.eye(2) * 100.0
        kf = KalmanFilter(X=x, P=p)
        z = np.array([0.0, 0.0])
        r = np.eye(2) * 1e-6
        kf.update(z, r)
        np.testing.assert_allclose(kf.X, z, atol=0.01)

    def test_uncertain_measurement(self):
        """Measurement with high noise should barely change the prior."""
        x = np.array([5.0, 5.0])
        p = np.eye(2) * 0.01  # very certain prior
        kf = KalmanFilter(X=x, P=p)
        z = np.array([100.0, 100.0])
        r = np.eye(2) * 1e6  # very uncertain measurement
        kf.update(z, r)
        np.testing.assert_allclose(kf.X, x, atol=0.5)


# ---------------------------------------------------------------------------
# Predict-update cycle
# ---------------------------------------------------------------------------


class TestKalmanFilterCycle:
    def test_converges_on_constant_signal(self):
        """Repeated measurements of a constant value should converge."""
        true_val = np.array([3.0, -2.0, 1.0])
        kf = KalmanFilter(
            X=np.zeros(3),
            P=np.eye(3) * 100.0,
            Q=np.eye(3) * 0.01,
        )
        rng = np.random.default_rng(42)
        for _ in range(50):
            kf.predict()
            noise = rng.normal(0, 0.1, size=3)
            kf.update(true_val + noise, np.eye(3) * 0.01)
        np.testing.assert_allclose(kf.X, true_val, atol=0.2)

    def test_tracks_linear_motion(self):
        """Position+velocity model should track a constant-velocity target."""
        f = np.array([[1.0, 1.0], [0.0, 1.0]])  # position, velocity
        q = np.eye(2) * 0.001
        kf = KalmanFilter(
            X=np.array([0.0, 1.0]),  # start at 0, velocity 1
            P=np.eye(2) * 1.0,
            F=f,
            Q=q,
        )
        rng = np.random.default_rng(7)
        for t in range(1, 30):
            kf.predict()
            true_pos = float(t)
            z = np.array([true_pos + rng.normal(0, 0.5), 1.0 + rng.normal(0, 0.1)])
            kf.update(z, np.eye(2) * 0.25)

        # After 30 steps, position should be close to 30
        assert kf.X[0] == pytest.approx(30.0, abs=2.0)
        # Velocity should be close to 1
        assert kf.X[1] == pytest.approx(1.0, abs=0.3)

    def test_3d_psf_parameters(self):
        """Simulate filtering contrast + position (c, dx, dy) like PSF fits."""
        true_c, true_dx, true_dy = -0.05, 0.3, -0.2
        true_state = np.array([true_c, true_dx, true_dy])

        kf = KalmanFilter(
            X=np.array([-0.04, 0.0, 0.0]),
            P=np.eye(3) * 1.0,
            Q=np.eye(3) * 0.001,
        )
        rng = np.random.default_rng(99)
        for _ in range(20):
            kf.predict()
            noise = rng.normal(0, [0.005, 0.3, 0.3])
            measurement_cov = np.diag([0.005**2, 0.3**2, 0.3**2])
            kf.update(true_state + noise, measurement_cov)

        np.testing.assert_allclose(kf.X, true_state, atol=0.15)

    def test_single_dimension(self):
        """KalmanFilter should work with 1-D state."""
        kf = KalmanFilter(
            X=np.array([0.0]),
            P=np.array([[10.0]]),
            Q=np.array([[0.01]]),
        )
        for _ in range(10):
            kf.predict()
            kf.update(np.array([5.0]), np.array([[0.1]]))
        assert kf.X[0] == pytest.approx(5.0, abs=0.2)
