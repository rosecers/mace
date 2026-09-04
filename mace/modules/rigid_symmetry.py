"""Generic symmetry-adapted equivariant features for rigid bodies.

A rigid body's physical orientation is an element of ``SO(3) / G``, where
``G`` is its proper body-frame rotational symmetry group.  This module builds
body features that are invariant to the right action ``R -> R g`` for every
``g in G`` while remaining equivariant to global left rotations.

The implementation is group-agnostic: downstream code receives an explicit
set of proper rotation matrices.  Convenience constructors for the finite
cyclic and dihedral groups are provided, but the feature machinery itself
does not branch on group names.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from e3nn import o3


def _axis_vector(
    axis: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be one of 0, 1, or 2; got {axis}")
    vector = torch.zeros(3, dtype=dtype, device=device)
    vector[axis] = 1.0
    return vector


def _axis_angle_rotation(
    axis: torch.Tensor,
    angle: torch.Tensor,
) -> torch.Tensor:
    axis = axis / torch.linalg.vector_norm(axis)
    x, y, z = axis.unbind()
    zero = torch.zeros((), dtype=axis.dtype, device=axis.device)
    skew = torch.stack(
        (
            torch.stack((zero, -z, y)),
            torch.stack((z, zero, -x)),
            torch.stack((-y, x, zero)),
        )
    )
    identity = torch.eye(3, dtype=axis.dtype, device=axis.device)
    outer = torch.outer(axis, axis)
    cosine = torch.cos(angle)
    sine = torch.sin(angle)
    return cosine * identity + sine * skew + (1.0 - cosine) * outer


def cyclic_group_rotations(
    order: int,
    *,
    axis: int = 2,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return the proper rotation matrices of the cyclic group C_n."""
    if order < 1:
        raise ValueError(f"order must be positive; got {order}")
    device = torch.device(device)
    body_axis = _axis_vector(axis, dtype=dtype, device=device)
    angles = torch.arange(order, dtype=dtype, device=device) * (
        2.0 * math.pi / float(order)
    )
    return torch.stack(
        tuple(_axis_angle_rotation(body_axis, angle) for angle in angles)
    )


def dihedral_group_rotations(
    order: int,
    *,
    principal_axis: int = 2,
    transverse_axis: int = 0,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return the 2n proper rotations of the rotational dihedral group D_n."""
    if order < 2:
        raise ValueError(f"order must be at least two; got {order}")
    if principal_axis == transverse_axis:
        raise ValueError("principal_axis and transverse_axis must differ")

    device = torch.device(device)
    cyclic = cyclic_group_rotations(
        order,
        axis=principal_axis,
        dtype=dtype,
        device=device,
    )
    transverse = _axis_vector(
        transverse_axis,
        dtype=dtype,
        device=device,
    )
    half_turn = _axis_angle_rotation(
        transverse,
        torch.tensor(math.pi, dtype=dtype, device=device),
    )
    flipped = torch.einsum("gij,jk->gik", cyclic, half_turn)
    return torch.cat((cyclic, flipped), dim=0)


def validate_rotation_group(
    rotations: torch.Tensor,
    *,
    atol: float = 1.0e-7,
) -> torch.Tensor:
    """Validate an explicit finite set of proper body-frame rotations."""
    if rotations.ndim != 3 or rotations.shape[-2:] != (3, 3):
        raise ValueError(
            "rotations must have shape (n_group, 3, 3); "
            f"got {tuple(rotations.shape)}"
        )
    if rotations.shape[0] == 0:
        raise ValueError("rotations must contain at least one group element")
    if not rotations.is_floating_point():
        raise ValueError("rotations must use a floating-point dtype")

    identity = torch.eye(
        3,
        dtype=rotations.dtype,
        device=rotations.device,
    )
    gram = rotations.transpose(-1, -2) @ rotations
    if not torch.allclose(
        gram,
        identity.expand_as(gram),
        atol=atol,
        rtol=0.0,
    ):
        raise ValueError("rotations must be orthogonal")

    determinants = torch.linalg.det(rotations)
    if not torch.allclose(
        determinants,
        torch.ones_like(determinants),
        atol=atol,
        rtol=0.0,
    ):
        raise ValueError("rotations must all be proper rotations with det=+1")
    return rotations


def irrep_group_projector(
    rotations: torch.Tensor,
    irrep: o3.Irrep | str,
) -> torch.Tensor:
    """Return the Reynolds projector onto the G-invariant irrep subspace."""
    rotations = validate_rotation_group(rotations)
    irrep = o3.Irrep(irrep)
    representation = irrep.D_from_matrix(rotations)
    return representation.mean(dim=0)


def invariant_irrep_basis(
    rotations: torch.Tensor,
    irrep: o3.Irrep | str,
    *,
    tolerance: float = 1.0e-7,
) -> torch.Tensor:
    """Return an orthonormal row basis for the G-fixed subspace of an irrep."""
    projector = irrep_group_projector(rotations, irrep)
    projector = 0.5 * (projector + projector.transpose(-1, -2))
    eigenvalues, eigenvectors = torch.linalg.eigh(projector)
    keep = eigenvalues > 1.0 - tolerance
    return eigenvectors[:, keep].transpose(0, 1).contiguous()


def symmetrize_irrep_seeds(
    seeds: torch.Tensor,
    rotations: torch.Tensor,
    irrep: o3.Irrep | str,
) -> torch.Tensor:
    """Project one or more body-frame irrep seeds into the G-fixed subspace."""
    irrep = o3.Irrep(irrep)
    if seeds.ndim == 1:
        seeds = seeds.unsqueeze(0)
    if seeds.ndim != 2 or seeds.shape[-1] != irrep.dim:
        raise ValueError(
            "seeds must have shape (multiplicity, irrep.dim); "
            f"got {tuple(seeds.shape)} for {irrep} with dim={irrep.dim}"
        )
    projector = irrep_group_projector(rotations, irrep)
    return seeds @ projector.transpose(-1, -2)


class SymmetryAdaptedIrrepBlock(torch.nn.Module):
    """One irrep channel built from G-invariant body-frame seed vectors."""

    def __init__(
        self,
        *,
        irrep: o3.Irrep | str,
        seeds: torch.Tensor,
        group_rotations: torch.Tensor,
        project_seeds: bool = True,
        tolerance: float = 1.0e-7,
    ) -> None:
        super().__init__()
        self.irrep = o3.Irrep(irrep)
        group_rotations = validate_rotation_group(group_rotations)
        seeds = torch.as_tensor(
            seeds,
            dtype=group_rotations.dtype,
            device=group_rotations.device,
        )
        if seeds.ndim == 1:
            seeds = seeds.unsqueeze(0)
        if seeds.ndim != 2 or seeds.shape[-1] != self.irrep.dim:
            raise ValueError(
                "seeds must have shape (multiplicity, irrep.dim); "
                f"got {tuple(seeds.shape)} for {self.irrep}"
            )
        if project_seeds:
            seeds = symmetrize_irrep_seeds(
                seeds,
                group_rotations,
                self.irrep,
            )

        projector = irrep_group_projector(group_rotations, self.irrep)
        residual = seeds - seeds @ projector.transpose(-1, -2)
        if torch.max(torch.abs(residual)).item() > tolerance:
            raise ValueError(
                f"seeds for {self.irrep} are not invariant under the supplied group"
            )
        norms = torch.linalg.vector_norm(seeds, dim=-1)
        if torch.any(norms <= tolerance):
            raise ValueError(
                f"the supplied seeds project to zero in irrep {self.irrep}"
            )

        self.register_buffer("seeds", seeds)
        self.multiplicity = int(seeds.shape[0])
        self.irreps = o3.Irreps([(self.multiplicity, self.irrep)])

    def forward(self, rotation_matrices: torch.Tensor) -> torch.Tensor:
        if rotation_matrices.shape[-2:] != (3, 3):
            raise ValueError(
                "rotation_matrices must have shape (..., 3, 3); "
                f"got {tuple(rotation_matrices.shape)}"
            )
        representation = self.irrep.D_from_matrix(rotation_matrices)
        seeds = self.seeds.to(
            dtype=rotation_matrices.dtype,
            device=rotation_matrices.device,
        )
        features = torch.einsum("...ij,mj->...mi", representation, seeds)
        return features.flatten(start_dim=-2)


class SymmetryAdaptedBodyFeatures(torch.nn.Module):
    """Concatenate arbitrary symmetry-adapted body irrep channels."""

    def __init__(
        self,
        *,
        group_rotations: torch.Tensor,
        blocks: Sequence[tuple[o3.Irrep | str, torch.Tensor]],
        project_seeds: bool = True,
    ) -> None:
        super().__init__()
        group_rotations = validate_rotation_group(group_rotations)
        if not blocks:
            raise ValueError("blocks must contain at least one symmetry feature block")
        self.register_buffer("group_rotations", group_rotations)
        self.blocks = torch.nn.ModuleList(
            [
                SymmetryAdaptedIrrepBlock(
                    irrep=irrep,
                    seeds=seeds,
                    group_rotations=group_rotations,
                    project_seeds=project_seeds,
                )
                for irrep, seeds in blocks
            ]
        )
        irreps = o3.Irreps("")
        for block in self.blocks:
            irreps += block.irreps
        self.irreps = irreps

    def forward(self, rotation_matrices: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            tuple(block(rotation_matrices) for block in self.blocks),
            dim=-1,
        )


def symmetry_group_rotations(
    symmetry: str,
    *,
    principal_axis: int = 2,
    transverse_axis: int = 0,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Construct a finite proper rotation group from a compact group name."""
    name = symmetry.strip().upper()

    if name.startswith("C"):
        try:
            order = int(name[1:])
        except ValueError as exc:
            raise ValueError(f"Invalid cyclic symmetry name {symmetry!r}") from exc
        return cyclic_group_rotations(
            order,
            axis=principal_axis,
            dtype=dtype,
            device=device,
        )

    if name.startswith("D"):
        try:
            order = int(name[1:])
        except ValueError as exc:
            raise ValueError(f"Invalid dihedral symmetry name {symmetry!r}") from exc
        return dihedral_group_rotations(
            order,
            principal_axis=principal_axis,
            transverse_axis=transverse_axis,
            dtype=dtype,
            device=device,
        )

    raise ValueError(
        f"Unknown rigid-body symmetry {symmetry!r}; "
        "expected C<n> or D<n>, or supply explicit rotations."
    )


def automatic_symmetry_body_features(
    group_rotations: torch.Tensor,
    *,
    lmax: int,
    include_scalar: bool = False,
) -> SymmetryAdaptedBodyFeatures:
    """Build the complete symmetry-invariant body basis through angular rank lmax."""
    if isinstance(lmax, bool) or not isinstance(lmax, int) or lmax < 0:
        raise ValueError(f"lmax must be a non-negative integer; got {lmax!r}")

    group_rotations = validate_rotation_group(group_rotations)
    blocks = []

    first_l = 0 if include_scalar else 1

    for ell in range(first_l, lmax + 1):
        parity = 1 if ell % 2 == 0 else -1
        irrep = o3.Irrep(ell, parity)
        seeds = invariant_irrep_basis(group_rotations, irrep)

        if seeds.shape[0] == 0:
            continue

        blocks.append((irrep, seeds))

    if not blocks:
        raise ValueError(
            "No non-scalar symmetry-invariant body features exist "
            f"through lmax={lmax}"
        )

    return SymmetryAdaptedBodyFeatures(
        group_rotations=group_rotations,
        blocks=tuple(blocks),
        project_seeds=False,
    )


def automatic_named_symmetry_body_features(
    symmetry: str,
    *,
    lmax: int,
    principal_axis: int = 2,
    transverse_axis: int = 0,
    include_scalar: bool = False,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> SymmetryAdaptedBodyFeatures:
    """Build the complete invariant body basis for a named finite symmetry."""
    rotations = symmetry_group_rotations(
        symmetry,
        principal_axis=principal_axis,
        transverse_axis=transverse_axis,
        dtype=dtype,
        device=device,
    )
    return automatic_symmetry_body_features(
        rotations,
        lmax=lmax,
        include_scalar=include_scalar,
    )
