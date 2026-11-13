"""Kalman filter for iSCAT particle tracking.

:class:`KalmanFilter` is a general linear Kalman filter that smooths per-frame PSF
fit parameters (contrast, dx, dy) over time.

Rust acceleration candidate (batch updates across many particles).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


class KalmanFilter:
    """General linear Kalman filter.

    Args:
        X: Initial state vector (n,).
        P: Initial covariance matrix (n, n).
        F: State-transition model (n, n). Defaults to identity.
        H: Observation model (n, n). Defaults to identity.
        B: Control-input model (n, n). Defaults to identity.
        Q: Process noise covariance (n, n). Defaults to identity.
    """

    def __init__(
        self,
        X: npt.NDArray[np.floating],  # noqa: N803
        P: npt.NDArray[np.floating],  # noqa: N803
        F: npt.NDArray[np.floating] | None = None,  # noqa: N803
        H: npt.NDArray[np.floating] | None = None,  # noqa: N803
        B: npt.NDArray[np.floating] | None = None,  # noqa: N803
        Q: npt.NDArray[np.floating] | None = None,  # noqa: N803
    ) -> None:
        dims = len(X)
        expected = (dims, dims)

        def _check(matrix: npt.NDArray[np.floating], name: str) -> None:
            if matrix.shape != expected:
                msg = (
                    f"{name} dimensions {matrix.shape} do not match expected {expected}"
                )
                raise ValueError(msg)

        P_arr = np.asarray(P, dtype=float)  # noqa: N806
        _check(P_arr, "P")

        self.X: npt.NDArray[np.floating] = np.asarray(X, dtype=float)
        self.P: npt.NDArray[np.floating] = P_arr
        self.F: npt.NDArray[np.floating] = np.eye(dims)
        self.H: npt.NDArray[np.floating] = np.eye(dims)
        self.B: npt.NDArray[np.floating] = np.eye(dims)
        self.Q: npt.NDArray[np.floating] = np.eye(dims)

        if F is not None:
            self.F = np.asarray(F, dtype=float)
            _check(self.F, "F")
        if H is not None:
            self.H = np.asarray(H, dtype=float)
            _check(self.H, "H")
        if B is not None:
            self.B = np.asarray(B, dtype=float)
            _check(self.B, "B")
        if Q is not None:
            self.Q = np.asarray(Q, dtype=float)
            _check(self.Q, "Q")

        self.u: npt.NDArray[np.floating] = np.zeros(dims)

    def predict(
        self,
        new_u: npt.NDArray[np.floating] | None = None,
    ) -> None:
        """Predict the next state.

        Args:
            new_u: Control input vector; the stored ``self.u`` if *None*.
        """
        u = self.u if new_u is None else np.asarray(new_u, dtype=float)
        self.X = self.F @ self.X + self.B @ u
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(
        self,
        Z: npt.NDArray[np.floating],  # noqa: N803
        R: npt.NDArray[np.floating],  # noqa: N803
    ) -> None:
        """Incorporate a new measurement.

        Args:
            Z: Measurement vector (n,).
            R: Measurement noise covariance (n, n).
        """
        z = np.asarray(Z, dtype=float)
        r = np.asarray(R, dtype=float)

        innovation = z - self.H @ self.X
        innovation_cov = self.H @ self.P @ self.H.T + r
        # Solve S Kᵀ = (P Hᵀ)ᵀ for Kᵀ instead of inverting S directly;
        # more numerically stable when S is poorly conditioned.
        kalman_gain = np.linalg.solve(innovation_cov.T, (self.P @ self.H.T).T).T
        self.X = self.X + kalman_gain @ innovation
        # Joseph form: (I-KH) P (I-KH)ᵀ + K R Kᵀ — guarantees PSD under
        # floating-point round-off (important for the planned f32 Rust path).
        identity = np.eye(len(self.X))
        factor = identity - kalman_gain @ self.H
        self.P = factor @ self.P @ factor.T + kalman_gain @ r @ kalman_gain.T
