import torch
from e3nn import o3
from scipy.spatial.transform import Rotation

from mace.modules.rigid_pair_tp import (
    RigidPairC2EdgeEmbedding,
)


DTYPE = torch.float64


def matrix_to_wxyz(R):
    q_xyzw = Rotation.from_matrix(
        R.detach().cpu().numpy()
    ).as_quat()

    return torch.tensor(
        [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]],
        dtype=DTYPE,
    )


def pi_about(axis):
    signs = -torch.ones(3, dtype=DTYPE)
    signs[axis] = 1.0
    return torch.diag(signs)


def make_pair():
    R0 = o3.rand_matrix().to(DTYPE)
    R1 = o3.rand_matrix().to(DTYPE)

    q = torch.stack(
        (
            matrix_to_wxyz(R0),
            matrix_to_wxyz(R1),
        )
    )

    edge_index = torch.tensor(
        [[0], [1]],
        dtype=torch.long,
    )

    edge_vector = torch.tensor(
        [[1.3, -0.7, 0.9]],
        dtype=DTYPE,
    )

    return R0, R1, q, edge_index, edge_vector


def test_c2_pair_output_shape():
    sh_irreps = o3.Irreps.spherical_harmonics(3)

    model = RigidPairC2EdgeEmbedding(
        lmax=3,
        edge_irreps=sh_irreps,
        multiplicity=1,
        c2_axis=1,
    ).to(dtype=DTYPE)

    _, _, q, edge_index, edge_vector = make_pair()

    out = model(
        q,
        edge_index,
        edge_vector,
    )

    # Standard edge harmonics through l=3 have dimension
    # 1 + 3 + 5 + 7 = 16.
    assert out.shape == (1, 16)


def test_c2_pair_respects_physical_c2():
    model = RigidPairC2EdgeEmbedding(
        max_ell=3,
        multiplicity=1,
        c2_axis=1,
    ).to(dtype=DTYPE)

    R0, _, q, edge_index, edge_vector = make_pair()

    ref = model(
        q,
        edge_index,
        edge_vector,
    )

    # Water's genuine C2 operation in our fixed bead frame.
    q_alias = q.clone()
    q_alias[0] = matrix_to_wxyz(
        R0 @ pi_about(1)
    )

    aliased = model(
        q_alias,
        edge_index,
        edge_vector,
    )

    torch.testing.assert_close(
        aliased,
        ref,
        atol=1.0e-9,
        rtol=1.0e-9,
    )


def test_c2_pair_is_globally_equivariant():
    model = RigidPairC2EdgeEmbedding(
        max_ell=3,
        multiplicity=1,
        c2_axis=1,
    ).to(dtype=DTYPE)

    R0, R1, q, edge_index, edge_vector = make_pair()

    ref = model(
        q,
        edge_index,
        edge_vector,
    )

    S = o3.rand_matrix().to(DTYPE)

    q_rot = torch.stack(
        (
            matrix_to_wxyz(S @ R0),
            matrix_to_wxyz(S @ R1),
        )
    )

    edge_rot = edge_vector @ S.T

    rotated = model(
        q_rot,
        edge_index,
        edge_rot,
    )

    D = model.irreps_out.D_from_matrix(
        S
    ).to(dtype=DTYPE)

    expected = ref @ D.T

    # Same numerical tolerance required by the underlying
    # e3nn Wigner-D / spherical-harmonic operations.
    torch.testing.assert_close(
        rotated,
        expected,
        atol=1.0e-6,
        rtol=1.0e-6,
    )
