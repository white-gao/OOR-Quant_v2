"""Learnable and frozen matrix transforms for the isolated FlatQuant port.

The trainable path follows the official FlatQuant parameterization:

* every square factor is U @ diag(s) @ V.T;
* U and V use PyTorch's Cayley orthogonal parametrization;
* s is an unconstrained trainable vector initialized to one;
* large feature transforms are represented by two Kronecker factors.

The frozen classes hold materialized forward and inverse-transpose matrices.
They are used by the repository's deployment-matched finalization path so an
inference forward never rebuilds matrices from the Cayley parameters.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn


FlatTransformInit = Literal["identity", "random_orthogonal"]
FlatTransformKind = Literal["kronecker", "smoothquant"]


def closest_factor_pair(size: int) -> tuple[int, int]:
    """Return FlatQuant's difference-of-squares factorization of ``size``."""
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError(f"Transform size must be a positive integer, got {size!r}.")
    # The official get_decompose_dim searches n = a^2 - b^2 and returns
    # (a - b, a + b). Integers congruent to 2 mod 4 have no such
    # factorization; fail explicitly instead of inheriting its infinite loop.
    if size % 4 == 2:
        raise ValueError(
            "FlatQuant Kronecker decomposition requires an odd size or a "
            f"multiple of four, got {size}."
        )
    a = math.isqrt(size)
    if a * a < size:
        a += 1
    while True:
        difference = a * a - size
        b = math.isqrt(difference)
        if b * b == difference:
            return a - b, a + b
        a += 1


def _random_orthogonal(
    size: int,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    matrix = torch.randn(size, size, dtype=torch.float32, device=device)
    q, r = torch.linalg.qr(matrix)
    signs = torch.sign(torch.diagonal(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return q * signs.unsqueeze(0)


def _orthogonal_linear(size: int, *, init: FlatTransformInit) -> nn.Module:
    linear = nn.Linear(size, size, bias=False, dtype=torch.float32)
    with torch.no_grad():
        if init == "identity":
            linear.weight.copy_(torch.eye(size, dtype=torch.float32))
        elif init == "random_orthogonal":
            linear.weight.copy_(_random_orthogonal(size))
        else:
            raise ValueError(
                f"Unsupported FlatQuant transform initialization: {init!r}"
            )
    return nn.utils.parametrizations.orthogonal(
        linear,
        orthogonal_map="cayley",
        use_trivialization=False,
    )


class SVDCayleyFactor(nn.Module):
    """Official square factor U diag(s) V^T with a cheap inverse pair."""

    def __init__(
        self,
        size: int,
        *,
        init: FlatTransformInit = "random_orthogonal",
        singularity_epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        if size <= 0:
            raise ValueError("SVD factor size must be positive.")
        if not math.isfinite(singularity_epsilon) or singularity_epsilon <= 0:
            raise ValueError("singularity_epsilon must be finite and positive.")
        self.size = int(size)
        self.singularity_epsilon = float(singularity_epsilon)
        self.u = _orthogonal_linear(self.size, init=init)
        self.v = _orthogonal_linear(self.size, init=init)
        # Match the official implementation: this is deliberately not exp(s)
        # or softplus(s). A near-zero value is treated as a failed calibration
        # rather than silently changing the baseline parameterization.
        self.singular_values = nn.Parameter(
            torch.ones(self.size, dtype=torch.float32)
        )

    def _resolved_singular_values(
        self,
        *,
        inverse_transpose: bool,
    ) -> torch.Tensor:
        singular_values = self.singular_values
        if inverse_transpose:
            minimum = singular_values.detach().abs().amin()
            if float(minimum) <= self.singularity_epsilon:
                raise FloatingPointError(
                    "FlatQuant transform became near-singular: "
                    f"min_abs_s={float(minimum):.6e}, "
                    f"threshold={self.singularity_epsilon:.6e}."
                )
            singular_values = singular_values.reciprocal()
        return singular_values

    def matrix(self, *, inverse_transpose: bool = False) -> torch.Tensor:
        singular_values = self._resolved_singular_values(
            inverse_transpose=inverse_transpose
        )
        # P = U S V^T and P^{-T} = U S^{-1} V^T.
        return (self.u.weight * singular_values.unsqueeze(0)) @ self.v.weight.T

    @property
    def minimum_abs_singular_value(self) -> float:
        return float(self.singular_values.detach().abs().amin())


def kronecker_matmul(
    x: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    """Compute x @ kron(left, right) without building the full matrix."""
    if x.shape[-1] != left.shape[0] * right.shape[0]:
        raise ValueError(
            "Kronecker transform dimension mismatch: "
            f"input={x.shape[-1]} left={tuple(left.shape)} "
            f"right={tuple(right.shape)}"
        )
    original_shape = x.shape
    matrix_view = x.reshape(-1, left.shape[0], right.shape[0])
    matrix_view = torch.matmul(matrix_view, right)
    matrix_view = torch.matmul(left.T, matrix_view)
    return matrix_view.reshape(original_shape)


class FrozenSingleTransform(nn.Module):
    """Materialized square transform used after finalization."""

    def __init__(
        self,
        matrix: torch.Tensor,
        matrix_inverse_transpose: torch.Tensor,
    ) -> None:
        super().__init__()
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Frozen FlatQuant matrix must be square.")
        if matrix_inverse_transpose.shape != matrix.shape:
            raise ValueError("Forward and inverse-transpose matrix shapes differ.")
        self.size = int(matrix.shape[0])
        self.register_buffer("matrix_forward", matrix.detach().float().clone())
        self.register_buffer(
            "matrix_inverse_transpose",
            matrix_inverse_transpose.detach().float().clone(),
        )

    def matrix(self, *, inverse_transpose: bool = False) -> torch.Tensor:
        return (
            self.matrix_inverse_transpose
            if inverse_transpose
            else self.matrix_forward
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        inverse_transpose: bool = False,
    ) -> torch.Tensor:
        matrix = self.matrix(inverse_transpose=inverse_transpose).to(
            device=x.device,
            dtype=x.dtype,
        )
        if x.shape[-1] % self.size != 0:
            raise ValueError(
                f"Input last dimension {x.shape[-1]} is not divisible by "
                f"the FlatQuant factor size {self.size}."
            )
        original_shape = x.shape
        return x.reshape(-1, self.size).matmul(matrix).reshape(original_shape)


class SingleSVDTransform(nn.Module):
    """One official SVD/Cayley matrix, applied blockwise on the last axis."""

    def __init__(
        self,
        size: int,
        *,
        init: FlatTransformInit = "random_orthogonal",
    ) -> None:
        super().__init__()
        self.size = int(size)
        self.factor = SVDCayleyFactor(self.size, init=init)

    def matrix(self, *, inverse_transpose: bool = False) -> torch.Tensor:
        return self.factor.matrix(inverse_transpose=inverse_transpose)

    def forward(
        self,
        x: torch.Tensor,
        *,
        inverse_transpose: bool = False,
    ) -> torch.Tensor:
        if x.shape[-1] % self.size != 0:
            raise ValueError(
                f"Input last dimension {x.shape[-1]} is not divisible by "
                f"the FlatQuant factor size {self.size}."
            )
        original_shape = x.shape
        matrix = self.matrix(inverse_transpose=inverse_transpose).to(
            device=x.device,
            dtype=x.dtype,
        )
        return x.reshape(-1, self.size).matmul(matrix).reshape(original_shape)

    def materialize(self) -> FrozenSingleTransform:
        with torch.no_grad():
            return FrozenSingleTransform(
                self.matrix(),
                self.matrix(inverse_transpose=True),
            )

    @property
    def effective_matrix_parameters(self) -> int:
        return self.size**2


class FrozenKroneckerTransform(nn.Module):
    """Materialized Kronecker transform with an optional saved diagonal."""

    def __init__(
        self,
        *,
        left: torch.Tensor,
        right: torch.Tensor,
        left_inverse_transpose: torch.Tensor,
        right_inverse_transpose: torch.Tensor,
        diag_scale: torch.Tensor | None = None,
        use_diag: bool = False,
    ) -> None:
        super().__init__()
        if left.ndim != 2 or left.shape[0] != left.shape[1]:
            raise ValueError("Frozen left Kronecker factor must be square.")
        if right.ndim != 2 or right.shape[0] != right.shape[1]:
            raise ValueError("Frozen right Kronecker factor must be square.")
        if left_inverse_transpose.shape != left.shape:
            raise ValueError("Left inverse-transpose factor shape mismatch.")
        if right_inverse_transpose.shape != right.shape:
            raise ValueError("Right inverse-transpose factor shape mismatch.")
        self.left_size = int(left.shape[0])
        self.right_size = int(right.shape[0])
        self.size = self.left_size * self.right_size
        self.use_diag = bool(use_diag)
        self.register_buffer("left_matrix", left.detach().float().clone())
        self.register_buffer("right_matrix", right.detach().float().clone())
        self.register_buffer(
            "left_matrix_inverse_transpose",
            left_inverse_transpose.detach().float().clone(),
        )
        self.register_buffer(
            "right_matrix_inverse_transpose",
            right_inverse_transpose.detach().float().clone(),
        )
        if diag_scale is not None:
            resolved = diag_scale.detach().float().reshape(-1)
            if resolved.numel() != self.size:
                raise ValueError("Frozen diagonal scale size mismatch.")
            self.register_buffer("diag_scale", resolved.clone())
        else:
            self.register_buffer("diag_scale", None)
        if self.use_diag and self.diag_scale is None:
            raise ValueError("use_diag=True requires a diagonal scale.")

    def matrices(
        self,
        *,
        inverse_transpose: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if inverse_transpose:
            return (
                self.left_matrix_inverse_transpose,
                self.right_matrix_inverse_transpose,
            )
        return self.left_matrix, self.right_matrix

    def forward(
        self,
        x: torch.Tensor,
        *,
        inverse_transpose: bool = False,
        include_diag: bool | None = None,
    ) -> torch.Tensor:
        resolved_include_diag = self.use_diag if include_diag is None else include_diag
        if resolved_include_diag:
            if self.diag_scale is None:
                raise ValueError("Requested a missing FlatQuant diagonal scale.")
            scale = self.diag_scale.to(device=x.device, dtype=x.dtype)
            x = x / scale if inverse_transpose else x * scale
        left, right = self.matrices(inverse_transpose=inverse_transpose)
        return kronecker_matmul(
            x,
            left.to(device=x.device, dtype=x.dtype),
            right.to(device=x.device, dtype=x.dtype),
        )

    def transform_weight(self, weight: torch.Tensor) -> torch.Tensor:
        return self.forward(weight, inverse_transpose=True, include_diag=True)

    @property
    def effective_matrix_parameters(self) -> int:
        return self.left_size**2 + self.right_size**2


class KroneckerSVDTransform(nn.Module):
    """Official decomposed transform with an optional full diagonal scale."""

    def __init__(
        self,
        size: int,
        *,
        init: FlatTransformInit = "random_orthogonal",
        add_diag: bool = False,
        diag_init: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.size = int(size)
        self.left_size, self.right_size = closest_factor_pair(self.size)
        self.left = SVDCayleyFactor(self.left_size, init=init)
        self.right = SVDCayleyFactor(self.right_size, init=init)
        self.add_diag = bool(add_diag)
        if self.add_diag:
            initial = (
                torch.ones(self.size, dtype=torch.float32)
                if diag_init is None
                else diag_init.detach().float().reshape(-1)
            )
            if initial.numel() != self.size:
                raise ValueError(
                    f"Expected diagonal scale size {self.size}, got {initial.numel()}."
                )
            if not torch.isfinite(initial).all() or torch.any(initial == 0):
                raise ValueError(
                    "FlatQuant diagonal initialization must be finite and non-zero."
                )
            self.diag_scale = nn.Parameter(initial.clone())
        else:
            self.register_parameter("diag_scale", None)

    def matrices(
        self,
        *,
        inverse_transpose: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.left.matrix(inverse_transpose=inverse_transpose),
            self.right.matrix(inverse_transpose=inverse_transpose),
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        inverse_transpose: bool = False,
        include_diag: bool = True,
    ) -> torch.Tensor:
        if include_diag and self.diag_scale is not None:
            scale = self.diag_scale.to(device=x.device, dtype=x.dtype)
            x = x / scale if inverse_transpose else x * scale
        left, right = self.matrices(inverse_transpose=inverse_transpose)
        return kronecker_matmul(
            x,
            left.to(device=x.device, dtype=x.dtype),
            right.to(device=x.device, dtype=x.dtype),
        )

    def transform_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """Return W D^-1 P^-T paired with (X D) P."""
        if weight.ndim != 2 or weight.shape[1] != self.size:
            raise ValueError(
                f"Expected a 2D weight with in_features={self.size}, "
                f"got {tuple(weight.shape)}"
            )
        return self.forward(weight, inverse_transpose=True, include_diag=True)

    def materialize(self, *, use_diag: bool = False) -> FrozenKroneckerTransform:
        with torch.no_grad():
            left, right = self.matrices()
            left_inv_t, right_inv_t = self.matrices(inverse_transpose=True)
            return FrozenKroneckerTransform(
                left=left,
                right=right,
                left_inverse_transpose=left_inv_t,
                right_inverse_transpose=right_inv_t,
                diag_scale=self.diag_scale,
                use_diag=use_diag,
            )

    @property
    def effective_matrix_parameters(self) -> int:
        return self.left_size**2 + self.right_size**2


class FixedSmoothQuantTransform(nn.Module):
    """Legacy fixed SmoothQuant control retained for prior core experiments."""

    def __init__(self, scale: torch.Tensor) -> None:
        super().__init__()
        resolved = scale.detach().float().reshape(-1)
        if resolved.numel() == 0:
            raise ValueError("SmoothQuant transform scale must be non-empty.")
        if not torch.isfinite(resolved).all() or torch.any(resolved <= 0):
            raise ValueError(
                "SmoothQuant transform scale must be finite and strictly positive."
            )
        self.size = int(resolved.numel())
        self.left_size = self.size
        self.right_size = 1
        self.register_buffer("scale", resolved.clone(), persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.size:
            raise ValueError(
                f"Expected input last dim {self.size}, got {tuple(x.shape)}"
            )
        return x / self.scale.to(device=x.device, dtype=x.dtype)

    def transform_weight(self, weight: torch.Tensor) -> torch.Tensor:
        if weight.ndim != 2 or weight.shape[1] != self.size:
            raise ValueError(
                f"Expected a 2D weight with in_features={self.size}, "
                f"got {tuple(weight.shape)}"
            )
        return weight * self.scale.to(
            device=weight.device,
            dtype=weight.dtype,
        ).view(1, -1)

    @property
    def effective_matrix_parameters(self) -> int:
        return self.size
