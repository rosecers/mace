"""Standalone equivariant tensor-product features for rigid-body pairs.

This module is diagnostic only and is not wired into MACE interactions.

A rigid orientation matrix R is represented by its three body axes in the
lab frame. Each axis transforms as an l=1 polar vector under proper global
rotations, giving orientation irreps

    3x1o.

For an edge i -> j we construct

    Y_l(rhat_ij) x frame_i x frame_j

using full e3nn tensor products.

Because FullTensorProduct retains every allowed Clebsch-Gordan path, this
stage does not introduce a learned compression of the angular information.
"""

from __future__ import annotations

import math

import torch
from e3nn import o3

from mace.data.rigid_body import quaternion_to_matrix
from mace.modules.rigid_symmetry import (
    SymmetryAdaptedBodyFeatures,
    cyclic_group_rotations,
    dihedral_group_rotations,
)


class RigidPairTensorProductFeatures(torch.nn.Module):
    """Full equivariant tensor-product basis for an ordered rigid pair."""

    def __init__(self, lmax: int = 2):
        super().__init__()

        if lmax < 0:
            raise ValueError("lmax must be >= 0")

        self.lmax = lmax

        self.edge_irreps = o3.Irreps.spherical_harmonics(lmax)
        self.frame_irreps = o3.Irreps("3x1o")

        # First couple positional angular information to the center frame.
        self.edge_center_tp = o3.FullTensorProduct(
            self.edge_irreps,
            self.frame_irreps,
        )

        # Then couple the neighbor frame.
        self.pair_tp = o3.FullTensorProduct(
            self.edge_center_tp.irreps_out,
            self.frame_irreps,
        )

        self.irreps_out = self.pair_tp.irreps_out

    @staticmethod
    def _frame_features(rotation_matrices: torch.Tensor) -> torch.Tensor:
        """Convert rotation matrices to 3x1o body-axis features.

        ``rotation_matrices[..., :, a]`` is body axis ``a`` expressed in
        lab coordinates.

        e3nn expects multiplicity-major layout

            [axis_0_xyz, axis_1_xyz, axis_2_xyz],

        hence the transpose before flattening.
        """
        return rotation_matrices.transpose(-1, -2).reshape(
            *rotation_matrices.shape[:-2],
            9,
        )

    def forward(
        self,
        quaternions: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vectors: torch.Tensor,
    ) -> torch.Tensor:
        """Construct equivariant rigid-pair features.

        Parameters
        ----------
        quaternions
            ``(num_nodes, 4)`` scalar-first ``[w, x, y, z]`` quaternions.
        edge_index
            ``(2, num_edges)``.
        edge_vectors
            ``(num_edges, 3)``.

        Returns
        -------
        torch.Tensor
            ``(num_edges, irreps_out.dim)`` equivariant pair features.
        """
        if quaternions.ndim != 2 or quaternions.shape[-1] != 4:
            raise ValueError("quaternions must have shape (num_nodes, 4)")

        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, num_edges)")

        if edge_vectors.ndim != 2 or edge_vectors.shape[-1] != 3:
            raise ValueError("edge_vectors must have shape (num_edges, 3)")

        if edge_vectors.shape[0] != edge_index.shape[1]:
            raise ValueError(
                "edge_vectors and edge_index must contain the same number of edges"
            )

        distances = torch.linalg.vector_norm(
            edge_vectors,
            dim=-1,
        )

        if torch.any(distances <= 1.0e-12):
            raise ValueError(
                "RigidPairTensorProductFeatures does not support zero-length edges"
            )

        directions = edge_vectors / distances.unsqueeze(-1)

        edge_features = o3.spherical_harmonics(
            self.edge_irreps,
            directions,
            normalize=True,
            normalization="component",
        )

        rotations = quaternion_to_matrix(quaternions)

        frame_features = self._frame_features(rotations)

        i = edge_index[0]
        j = edge_index[1]

        frame_i = frame_features[i]
        frame_j = frame_features[j]

        edge_center = self.edge_center_tp(
            edge_features,
            frame_i,
        )

        return self.pair_tp(
            edge_center,
            frame_j,
        )


class RigidPairEdgeEmbedding(torch.nn.Module):
    """Learned compressed rigid-pair edge representation.

    The complete pair tensor product is projected to ``multiplicity``
    learned copies of each ordinary edge spherical-harmonic irrep.

    multiplicity=1 preserves the original projected full-frame model.

    The projected rigid block is scaled by 1/sqrt(multiplicity), keeping
    its aggregate norm approximately comparable as multiplicity grows.
    """

    def __init__(
        self,
        lmax: int,
        edge_irreps: o3.Irreps,
        multiplicity: int = 1,
    ):
        super().__init__()

        if isinstance(multiplicity, bool) or multiplicity < 1:
            raise ValueError(
                "rigid pair multiplicity must be a positive integer, "
                f"got {multiplicity!r}"
            )

        self.multiplicity = int(multiplicity)

        self.full_pair = RigidPairTensorProductFeatures(
            lmax=lmax,
        )

        self.base_edge_irreps = o3.Irreps(edge_irreps)

        # Increase multiplicity without changing which irreps appear.
        #
        # Example:
        #
        #   0e + 1o + 2e + 3o
        #
        # becomes, for multiplicity=4,
        #
        #   4x0e + 4x1o + 4x2e + 4x3o
        #
        self.edge_irreps = o3.Irreps(
            [(mul * self.multiplicity, ir) for mul, ir in self.base_edge_irreps]
        )

        # Do not perturb initialization of the surrounding MACE model.
        with torch.random.fork_rng(devices=[]):
            self.projection = o3.Linear(
                self.full_pair.irreps_out,
                self.edge_irreps,
            )

        self.output_scale = 1.0 / math.sqrt(self.multiplicity)

    def forward(
        self,
        quaternions: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vectors: torch.Tensor,
    ) -> torch.Tensor:
        full_pair = self.full_pair(
            quaternions,
            edge_index,
            edge_vectors,
        )

        projected = self.projection(full_pair)

        return projected * self.output_scale


class RigidPairIrrepCompleteEdgeEmbedding(torch.nn.Module):
    """Compact projection retaining every raw rigid-pair irrep type.

    Unlike ``RigidPairEdgeEmbedding``, which projects only onto the
    ordinary edge spherical-harmonic irreps, this module retains one
    learned copy of every distinct (L, parity) irrep occurring in the
    complete

        Y_l(r_hat) x frame_i x frame_j

    tensor product.

    Multiplicity inside the raw tensor product is compressed to one
    learned copy per irrep type, but no angular/parity sector is
    discarded.
    """

    def __init__(self, lmax: int):
        super().__init__()

        self.full_pair = RigidPairTensorProductFeatures(
            lmax=lmax,
        )

        by_type = {}

        for _, ir in self.full_pair.irreps_out:
            by_type[(ir.l, ir.p)] = ir

        # Deterministic order:
        #   even parity, increasing L
        #   odd parity, increasing L
        keys = sorted(
            by_type,
            key=lambda key: (
                0 if key[1] == 1 else 1,
                key[0],
            ),
        )

        self.edge_irreps = o3.Irreps([(1, by_type[key]) for key in keys])

        # Do not perturb initialization of the surrounding MACE model.
        with torch.random.fork_rng(devices=[]):
            self.projection = o3.Linear(
                self.full_pair.irreps_out,
                self.edge_irreps,
            )

    def forward(
        self,
        quaternions: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vectors: torch.Tensor,
    ) -> torch.Tensor:
        full_pair = self.full_pair(
            quaternions,
            edge_index,
            edge_vectors,
        )

        return self.projection(full_pair)


class RigidPairRawEdgeEmbedding(torch.nn.Module):
    """Uncompressed rigid-pair tensor-product edge representation.

    This diagnostic module performs no learned projection back to the
    ordinary spherical-harmonic irreps. Its output is exactly the full

        Y_l(r_hat) x frame_i x frame_j

    tensor-product representation.
    """

    def __init__(self, lmax: int):
        super().__init__()

        self.full_pair = RigidPairTensorProductFeatures(lmax=lmax)
        self.edge_irreps = self.full_pair.irreps_out

    def forward(
        self,
        quaternions: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vectors: torch.Tensor,
    ) -> torch.Tensor:
        return self.full_pair(
            quaternions,
            edge_index,
            edge_vectors,
        )


class RigidPairSymmetryEdgeEmbedding(torch.nn.Module):
    """Projected rigid-pair features for an arbitrary body symmetry."""

    def __init__(
        self,
        *,
        body_features: torch.nn.Module,
        max_ell=None,
        multiplicity=1,
        lmax=None,
        edge_irreps=None,
        restrict_pair_irreps=False,
    ):
        super().__init__()
        if max_ell is None:
            max_ell = lmax
        if max_ell is None:
            raise TypeError("max_ell/lmax must be provided")
        if multiplicity < 1:
            raise ValueError("multiplicity must be >= 1")
        if not hasattr(body_features, "irreps"):
            raise TypeError("body_features must expose an 'irreps' attribute")

        self.max_ell = int(max_ell)
        self.multiplicity = int(multiplicity)
        self.restrict_pair_irreps = bool(restrict_pair_irreps)
        self.body_features = body_features
        self.body_irreps = o3.Irreps(body_features.irreps)

        if edge_irreps is None:
            self.sh_irreps = o3.Irreps.spherical_harmonics(self.max_ell)
        else:
            self.sh_irreps = o3.Irreps(edge_irreps)

        self.edge_body_tp = o3.FullTensorProduct(
            self.sh_irreps,
            self.body_irreps,
        )

        if self.restrict_pair_irreps:
            allowed_irreps = [ir for _, ir in self.sh_irreps]
            self.pair_tp = o3.FullTensorProduct(
                self.edge_body_tp.irreps_out,
                self.body_irreps,
                filter_ir_out=allowed_irreps,
            )
        else:
            self.pair_tp = o3.FullTensorProduct(
                self.edge_body_tp.irreps_out,
                self.body_irreps,
            )

        self.irreps_in = self.pair_tp.irreps_out
        self.edge_irreps = o3.Irreps(
            [(mul * self.multiplicity, ir) for mul, ir in self.sh_irreps]
        )
        self.irreps_out = self.edge_irreps

        with torch.random.fork_rng(devices=[]):
            self.projection = o3.Linear(
                self.irreps_in,
                self.irreps_out,
            )

    def forward(
        self,
        quaternions,
        edge_index,
        edge_vectors,
    ):
        rotations = quaternion_to_matrix(quaternions)
        body = self.body_features(rotations)

        edge_sh = o3.spherical_harmonics(
            self.sh_irreps,
            edge_vectors,
            normalize=True,
            normalization="component",
        )

        senders = edge_index[0]
        receivers = edge_index[1]

        x = self.edge_body_tp(
            edge_sh,
            body[senders],
        )
        x = self.pair_tp(
            x,
            body[receivers],
        )
        x = self.projection(x)

        if self.multiplicity > 1:
            x = x / self.multiplicity**0.5

        return x


def validate_rigid_pair_mode(mode: str) -> str:
    """Validate the rigid-pair edge representation mode."""
    if mode in (
        "c2_frame",
        "d6_frame",
        "d6_frame_compact",
    ):
        return mode
    valid_modes = {
        "none",
        "full_frame",
        "full_frame_compact",
        "full_frame_irrep_complete",
        "full_frame_raw",
        "invariant_radial",
        "symmetry_frame",
        "symmetry_frame_compact",
    }

    if mode not in valid_modes:
        raise ValueError(
            f"Unknown rigid_pair_mode={mode!r}. "
            f"Expected one of {sorted(valid_modes)}."
        )

    return mode


def _c2_symmetry_body_features(c2_axis):
    if c2_axis not in (0, 1, 2):
        raise ValueError("c2_axis must be 0, 1, or 2")

    identity = torch.eye(3, dtype=torch.float64)
    transverse = [axis for axis in range(3) if axis != c2_axis]

    axis_seed = identity[:, c2_axis]
    b = identity[:, transverse[0]]
    c = identity[:, transverse[1]]

    quadrupole_seed = (
        o3.spherical_harmonics(
            2,
            b,
            normalize=True,
            normalization="component",
        )
        - o3.spherical_harmonics(
            2,
            c,
            normalize=True,
            normalization="component",
        )
    ) / math.sqrt(2.0)

    return SymmetryAdaptedBodyFeatures(
        group_rotations=cyclic_group_rotations(
            2,
            axis=c2_axis,
            dtype=identity.dtype,
        ),
        blocks=(
            ("1o", axis_seed),
            ("2e", quadrupole_seed),
        ),
    )


def _d6_symmetry_body_features():
    identity = torch.eye(3, dtype=torch.float64)
    normal = identity[:, 2]

    plane_seed = o3.spherical_harmonics(
        2,
        normal,
        normalize=True,
        normalization="component",
    )

    angles = torch.arange(6, dtype=identity.dtype) * (math.pi / 3.0)
    body_hexagon = torch.stack(
        (
            torch.cos(angles),
            torch.sin(angles),
            torch.zeros_like(angles),
        ),
        dim=-1,
    )

    hexatic_seed = o3.spherical_harmonics(
        6,
        body_hexagon,
        normalize=True,
        normalization="component",
    ).mean(dim=0)

    return SymmetryAdaptedBodyFeatures(
        group_rotations=dihedral_group_rotations(
            6,
            principal_axis=2,
            transverse_axis=0,
            dtype=identity.dtype,
        ),
        blocks=(
            ("2e", plane_seed),
            ("6e", hexatic_seed),
        ),
    )


class RigidPairD6EdgeEmbedding(RigidPairSymmetryEdgeEmbedding):
    """Compatibility wrapper for the historical D6 rigid-pair mode."""

    def __init__(
        self,
        max_ell=None,
        multiplicity=1,
        lmax=None,
        edge_irreps=None,
        **kwargs,
    ):
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs)}")

        super().__init__(
            body_features=_d6_symmetry_body_features(),
            max_ell=max_ell,
            lmax=lmax,
            edge_irreps=edge_irreps,
            multiplicity=multiplicity,
            restrict_pair_irreps=True,
        )


class RigidPairC2EdgeEmbedding(RigidPairSymmetryEdgeEmbedding):
    """Compatibility wrapper for the historical C2 rigid-pair mode."""

    def __init__(
        self,
        max_ell=None,
        multiplicity=1,
        c2_axis=1,
        lmax=None,
        edge_irreps=None,
        **kwargs,
    ):
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs)}")

        self.c2_axis = int(c2_axis)

        super().__init__(
            body_features=_c2_symmetry_body_features(self.c2_axis),
            max_ell=max_ell,
            lmax=lmax,
            edge_irreps=edge_irreps,
            multiplicity=multiplicity,
            restrict_pair_irreps=False,
        )
