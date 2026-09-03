"""Compact equivariant rigid-body features for C2-symmetric molecules.

For a body-to-space rotation matrix

    R = [a b c]

choose one body axis ``a`` as the physical C2 axis.  The molecular
orientation modulo the C2 operation

    (a, b, c) ~ (a, -b, -c)

can be represented equivariantly by

    a                         : l=1, odd parity
    Y_2(b) - Y_2(c)           : l=2, even parity

giving the compact irrep content

    1o + 2e

with total dimension 3 + 5 = 8.

Unlike a generic moment-of-inertia tensor, this representation does
not quotient by the two extra, unphysical D2 pi rotations about the
transverse principal axes.
"""

import math

import torch
from e3nn import o3


C2_BODY_IRREPS = o3.Irreps("1o + 2e")


def _validate_c2_axis(c2_axis: int) -> None:
    if c2_axis not in (0, 1, 2):
        raise ValueError(
            "c2_axis must be one of 0, 1, or 2; "
            f"got {c2_axis}"
        )


def c2_body_irreducible_features(
    rotation_matrices: torch.Tensor,
    c2_axis: int,
) -> torch.Tensor:
    """Return compact C2-invariant, globally equivariant body features.

    Parameters
    ----------
    rotation_matrices
        Body-to-space rotation matrices with shape ``(..., 3, 3)``.
        Columns are the body-frame axes expressed in lab coordinates.

    c2_axis
        Column index of the molecular C2 axis.

    Returns
    -------
    torch.Tensor
        Features with shape ``(..., 8)`` transforming as ``1o + 2e``.
    """
    _validate_c2_axis(c2_axis)

    if rotation_matrices.shape[-2:] != (3, 3):
        raise ValueError(
            "rotation_matrices must have shape (..., 3, 3); "
            f"got {tuple(rotation_matrices.shape)}"
        )

    transverse = [
        i
        for i in range(3)
        if i != c2_axis
    ]

    a = rotation_matrices[..., :, c2_axis]
    b = rotation_matrices[..., :, transverse[0]]
    c = rotation_matrices[..., :, transverse[1]]

    y2_b = o3.spherical_harmonics(
        2,
        b,
        normalize=True,
        normalization="component",
    )

    y2_c = o3.spherical_harmonics(
        2,
        c,
        normalize=True,
        normalization="component",
    )

    transverse_quadrupole = (
        y2_b - y2_c
    ) / math.sqrt(2.0)

    return torch.cat(
        (
            a,
            transverse_quadrupole,
        ),
        dim=-1,
    )
