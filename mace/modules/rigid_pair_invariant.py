"""Global-rotation-invariant coordinates for ordered rigid molecular pairs."""

import torch

from mace.modules.rigid_pair import quaternion_to_matrix


def rigid_pair_invariant_geometry(
    quaternions: torch.Tensor,
    edge_index: torch.Tensor,
    edge_vectors: torch.Tensor,
    eps: float = 1.0e-12,
) -> torch.Tensor:
    """Return complete redundant invariant coordinates for rigid pairs.

    For each directed edge i <- j, returns

        [
            r_ij,
            R_i^T rhat_ij,          # 3
            R_j^T rhat_ij,          # 3
            vec(R_i^T R_j),         # 9
        ]

    giving 16 scalar components per edge.

    These coordinates are invariant under a common global rotation of
    positions and molecular orientations. They are redundant because the
    vector and rotation-matrix blocks satisfy geometric constraints, but
    they retain the complete ordered rigid-pair relative geometry before
    quotienting by molecular point-group symmetry.
    """

    if edge_vectors.ndim != 2 or edge_vectors.shape[-1] != 3:
        raise ValueError(
            "edge_vectors must have shape [num_edges, 3], "
            f"got {tuple(edge_vectors.shape)}"
        )

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(
            "edge_index must have shape [2, num_edges], "
            f"got {tuple(edge_index.shape)}"
        )

    if quaternions.ndim != 2 or quaternions.shape[-1] != 4:
        raise ValueError(
            "quaternions must have shape [num_nodes, 4], "
            f"got {tuple(quaternions.shape)}"
        )

    centers = edge_index[0]
    neighbors = edge_index[1]

    rotations = quaternion_to_matrix(quaternions)

    R_i = rotations[centers]
    R_j = rotations[neighbors]

    distance = torch.linalg.vector_norm(
        edge_vectors,
        dim=-1,
        keepdim=True,
    )

    rhat = edge_vectors / distance.clamp_min(eps)

    # torch vectors are row vectors here:
    # (R^T rhat)_a = sum_b R_{b a} rhat_b
    body_i_direction = torch.einsum(
        "ebi,eb->ei",
        R_i,
        rhat,
    )

    body_j_direction = torch.einsum(
        "ebi,eb->ei",
        R_j,
        rhat,
    )

    relative_rotation = torch.einsum(
        "eai,eaj->eij",
        R_i,
        R_j,
    )

    return torch.cat(
        (
            distance,
            body_i_direction,
            body_j_direction,
            relative_rotation.reshape(
                relative_rotation.shape[0],
                9,
            ),
        ),
        dim=-1,
    )


class RigidPairInvariantRadialConditioning(torch.nn.Module):
    """Invariant FiLM conditioning of scalar radial edge features.

    The rigid-pair descriptor is globally rotation invariant, so its
    learned scale and shift remain scalar edge features. The final layer
    is initialized to zero, making this module exactly the identity at
    initialization:

        edge_feats_out = edge_feats

    Training may then learn orientation-dependent modulation without
    changing the equivariant edge-attribute representation.
    """

    def __init__(
        self,
        radial_dim: int,
        r_max: float,
        hidden_dim: int = 64,
    ):
        super().__init__()

        if radial_dim < 1:
            raise ValueError(f"radial_dim must be positive, got {radial_dim}")

        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")

        if r_max <= 0.0:
            raise ValueError(f"r_max must be positive, got {r_max}")

        self.radial_dim = int(radial_dim)
        self.r_max = float(r_max)

        self.net = torch.nn.Sequential(
            torch.nn.Linear(16, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(
                hidden_dim,
                2 * self.radial_dim,
            ),
        )

        # Start as exactly ordinary MACE.
        last = self.net[-1]
        torch.nn.init.zeros_(last.weight)
        torch.nn.init.zeros_(last.bias)

    def forward(
        self,
        edge_feats: torch.Tensor,
        quaternions: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vectors: torch.Tensor,
    ) -> torch.Tensor:
        if edge_feats.ndim != 2:
            raise ValueError(
                "edge_feats must have shape [num_edges, radial_dim], "
                f"got {tuple(edge_feats.shape)}"
            )

        if edge_feats.shape[-1] != self.radial_dim:
            raise ValueError(
                "edge_feats final dimension does not match radial_dim: "
                f"{edge_feats.shape[-1]} != {self.radial_dim}"
            )

        invariant = rigid_pair_invariant_geometry(
            quaternions=quaternions,
            edge_index=edge_index,
            edge_vectors=edge_vectors,
        )

        # All orientation entries already lie in [-1, 1].
        # Scale only the dimensional distance coordinate.
        conditioning_input = torch.cat(
            (
                invariant[:, :1] / self.r_max,
                invariant[:, 1:],
            ),
            dim=-1,
        )

        film = self.net(conditioning_input)
        scale, shift = torch.chunk(
            film,
            chunks=2,
            dim=-1,
        )

        return edge_feats * (1.0 + scale) + shift


def reverse_rigid_pair_invariant(
    invariant: torch.Tensor,
) -> torch.Tensor:
    """Return invariant coordinates for the reversed directed pair.

    Input layout:

        [
            r,
            R_i^T rhat_ij,
            R_j^T rhat_ij,
            vec(R_i^T R_j),
        ]

    Under edge reversal i <- j  ->  j <- i,

        r                    -> r
        R_i^T rhat_ij        -> -R_j^T rhat_ij
        R_j^T rhat_ij        -> -R_i^T rhat_ij
        R_i^T R_j            -> (R_i^T R_j)^T
    """

    if invariant.ndim != 2 or invariant.shape[-1] != 16:
        raise ValueError(
            "invariant must have shape [num_edges, 16], "
            f"got {tuple(invariant.shape)}"
        )

    distance = invariant[:, :1]
    body_i = invariant[:, 1:4]
    body_j = invariant[:, 4:7]

    relative_rotation = invariant[:, 7:16].reshape(
        invariant.shape[0],
        3,
        3,
    )

    return torch.cat(
        (
            distance,
            -body_j,
            -body_i,
            relative_rotation.transpose(-1, -2).reshape(
                invariant.shape[0],
                9,
            ),
        ),
        dim=-1,
    )


class RigidPairInvariantEnergyHead(torch.nn.Module):
    """Swap-symmetric scalar energy from complete rigid-pair invariants.

    This head is globally rotation invariant by construction.

    Pair-exchange symmetry is imposed as

        e(q_ij) = 1/2 [g(q_ij) + g(q_ji)]

    rather than asking the network to learn that symmetry from data.

    The final layer is initialized to zero, so the module initially
    contributes exactly zero energy.
    """

    def __init__(
        self,
        r_max: float,
        hidden_dim: int = 32,
    ):
        super().__init__()

        if r_max <= 0.0:
            raise ValueError(f"r_max must be positive, got {r_max}")

        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")

        self.r_max = float(r_max)

        self.net = torch.nn.Sequential(
            torch.nn.Linear(16, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, 1),
        )

        last = self.net[-1]
        torch.nn.init.zeros_(last.weight)
        torch.nn.init.zeros_(last.bias)

    def _normalize(
        self,
        invariant: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            (
                invariant[:, :1] / self.r_max,
                invariant[:, 1:],
            ),
            dim=-1,
        )

    def forward(
        self,
        quaternions: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vectors: torch.Tensor,
    ) -> torch.Tensor:
        invariant = rigid_pair_invariant_geometry(
            quaternions=quaternions,
            edge_index=edge_index,
            edge_vectors=edge_vectors,
        )

        reverse = reverse_rigid_pair_invariant(invariant)

        forward_energy = self.net(self._normalize(invariant))

        reverse_energy = self.net(self._normalize(reverse))

        return 0.5 * (forward_energy + reverse_energy)
