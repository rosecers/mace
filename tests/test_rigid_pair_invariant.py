import numpy as np
import torch
from scipy.spatial.transform import Rotation

from mace.modules.rigid_pair_invariant import (
    rigid_pair_invariant_geometry,
)


DTYPE = torch.float64


def _quat_xyzw_to_wxyz(q):
    q = np.asarray(q)
    return np.concatenate((q[-1:], q[:3]))


def _global_rotate_quaternion(q_wxyz, global_rotation):
    q_xyzw = np.concatenate(
        (
            np.asarray(q_wxyz)[1:],
            np.asarray(q_wxyz)[:1],
        )
    )

    body = Rotation.from_quat(q_xyzw)
    rotated = global_rotation * body

    return _quat_xyzw_to_wxyz(
        rotated.as_quat()
    )


def test_invariant_pair_descriptor_has_16_components():
    q = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=DTYPE,
    )

    edge_index = torch.tensor(
        [[0], [1]],
        dtype=torch.long,
    )

    edge_vectors = torch.tensor(
        [[1.2, -0.4, 0.7]],
        dtype=DTYPE,
    )

    out = rigid_pair_invariant_geometry(
        q,
        edge_index,
        edge_vectors,
    )

    assert out.shape == (1, 16)


def test_invariant_pair_descriptor_is_globally_rotation_invariant():
    R_i = Rotation.from_euler(
        "xyz",
        [0.31, -0.52, 0.77],
    )
    R_j = Rotation.from_euler(
        "xyz",
        [-0.42, 0.63, -0.28],
    )

    q = torch.tensor(
        [
            _quat_xyzw_to_wxyz(R_i.as_quat()),
            _quat_xyzw_to_wxyz(R_j.as_quat()),
        ],
        dtype=DTYPE,
    )

    edge_index = torch.tensor(
        [[0], [1]],
        dtype=torch.long,
    )

    edge_vector = np.array(
        [1.3, -0.8, 0.55],
    )

    edge_vectors = torch.tensor(
        [edge_vector],
        dtype=DTYPE,
    )

    before = rigid_pair_invariant_geometry(
        q,
        edge_index,
        edge_vectors,
    )

    global_rotation = Rotation.from_euler(
        "zyx",
        [0.48, -0.37, 0.91],
    )

    q_rotated = torch.tensor(
        [
            _global_rotate_quaternion(
                q[0].numpy(),
                global_rotation,
            ),
            _global_rotate_quaternion(
                q[1].numpy(),
                global_rotation,
            ),
        ],
        dtype=DTYPE,
    )

    edge_vectors_rotated = torch.tensor(
        [
            global_rotation.apply(
                edge_vector
            )
        ],
        dtype=DTYPE,
    )

    after = rigid_pair_invariant_geometry(
        q_rotated,
        edge_index,
        edge_vectors_rotated,
    )

    torch.testing.assert_close(
        after,
        before,
        atol=1.0e-10,
        rtol=1.0e-10,
    )


def test_invariant_pair_descriptor_detects_neighbor_rotation():
    q0 = Rotation.identity()
    q1a = Rotation.identity()
    q1b = Rotation.from_euler(
        "xyz",
        [0.37, -0.41, 0.29],
    )

    edge_index = torch.tensor(
        [[0], [1]],
        dtype=torch.long,
    )

    edge_vectors = torch.tensor(
        [[1.1, 0.3, -0.6]],
        dtype=DTYPE,
    )

    qa = torch.tensor(
        [
            _quat_xyzw_to_wxyz(q0.as_quat()),
            _quat_xyzw_to_wxyz(q1a.as_quat()),
        ],
        dtype=DTYPE,
    )

    qb = torch.tensor(
        [
            _quat_xyzw_to_wxyz(q0.as_quat()),
            _quat_xyzw_to_wxyz(q1b.as_quat()),
        ],
        dtype=DTYPE,
    )

    a = rigid_pair_invariant_geometry(
        qa,
        edge_index,
        edge_vectors,
    )

    b = rigid_pair_invariant_geometry(
        qb,
        edge_index,
        edge_vectors,
    )

    assert not torch.allclose(a, b)


def test_quaternion_double_cover_does_not_change_invariants():
    q = torch.tensor(
        [
            [0.8, 0.1, -0.2, 0.3],
            [0.7, -0.4, 0.2, 0.1],
        ],
        dtype=DTYPE,
    )

    q = q / torch.linalg.vector_norm(
        q,
        dim=-1,
        keepdim=True,
    )

    edge_index = torch.tensor(
        [[0], [1]],
        dtype=torch.long,
    )

    edge_vectors = torch.tensor(
        [[0.7, -1.1, 0.4]],
        dtype=DTYPE,
    )

    a = rigid_pair_invariant_geometry(
        q,
        edge_index,
        edge_vectors,
    )

    b = rigid_pair_invariant_geometry(
        -q,
        edge_index,
        edge_vectors,
    )

    torch.testing.assert_close(
        a,
        b,
        atol=1.0e-12,
        rtol=1.0e-12,
    )


def test_radial_conditioning_starts_as_identity_and_gets_gradient():
    from mace.modules.rigid_pair_invariant import (
        RigidPairInvariantRadialConditioning,
    )

    torch.manual_seed(17)

    conditioner = RigidPairInvariantRadialConditioning(
        radial_dim=8,
        r_max=5.0,
        hidden_dim=16,
    ).to(dtype=DTYPE)

    q = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.9, 0.1, -0.2, 0.3],
        ],
        dtype=DTYPE,
    )

    q = q / torch.linalg.vector_norm(
        q,
        dim=-1,
        keepdim=True,
    )

    edge_index = torch.tensor(
        [[0], [1]],
        dtype=torch.long,
    )

    edge_vectors = torch.tensor(
        [[1.1, -0.3, 0.6]],
        dtype=DTYPE,
    )

    edge_feats = torch.randn(
        1,
        8,
        dtype=DTYPE,
        requires_grad=True,
    )

    out = conditioner(
        edge_feats=edge_feats,
        quaternions=q,
        edge_index=edge_index,
        edge_vectors=edge_vectors,
    )

    # Exact baseline at initialization.
    torch.testing.assert_close(
        out,
        edge_feats,
        atol=0.0,
        rtol=0.0,
    )

    loss = out.square().sum()
    loss.backward()

    last = conditioner.net[-1]

    assert last.weight.grad is not None
    assert torch.count_nonzero(
        last.weight.grad
    ).item() > 0


def test_reverse_invariant_is_an_involution():
    from mace.modules.rigid_pair_invariant import (
        reverse_rigid_pair_invariant,
    )

    torch.manual_seed(23)

    q = torch.randn(
        7,
        16,
        dtype=DTYPE,
    )

    # Make the distance slot physically sensible; the algebraic
    # involution does not otherwise require a valid rotation matrix.
    q[:, 0] = q[:, 0].abs() + 0.1

    twice = reverse_rigid_pair_invariant(
        reverse_rigid_pair_invariant(q)
    )

    torch.testing.assert_close(
        twice,
        q,
        atol=1.0e-12,
        rtol=1.0e-12,
    )


def test_invariant_energy_head_starts_at_zero():
    from mace.modules.rigid_pair_invariant import (
        RigidPairInvariantEnergyHead,
    )

    head = RigidPairInvariantEnergyHead(
        r_max=5.0,
        hidden_dim=16,
    ).to(dtype=DTYPE)

    q = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.9, 0.1, -0.2, 0.3],
        ],
        dtype=DTYPE,
    )

    q = q / torch.linalg.vector_norm(
        q,
        dim=-1,
        keepdim=True,
    )

    edge_index = torch.tensor(
        [[0], [1]],
        dtype=torch.long,
    )

    edge_vectors = torch.tensor(
        [[1.1, -0.3, 0.6]],
        dtype=DTYPE,
    )

    energy = head(
        quaternions=q,
        edge_index=edge_index,
        edge_vectors=edge_vectors,
    )

    torch.testing.assert_close(
        energy,
        torch.zeros_like(energy),
        atol=0.0,
        rtol=0.0,
    )


def test_invariant_energy_head_is_pair_exchange_symmetric():
    from mace.modules.rigid_pair_invariant import (
        RigidPairInvariantEnergyHead,
    )

    torch.manual_seed(31)

    head = RigidPairInvariantEnergyHead(
        r_max=5.0,
        hidden_dim=16,
    ).to(dtype=DTYPE)

    # Make the head nontrivial for the symmetry test.
    with torch.no_grad():
        head.net[-1].weight.normal_(
            mean=0.0,
            std=0.2,
        )
        head.net[-1].bias.fill_(0.07)

    q = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.8, 0.2, -0.3, 0.4],
        ],
        dtype=DTYPE,
    )

    q = q / torch.linalg.vector_norm(
        q,
        dim=-1,
        keepdim=True,
    )

    vector = torch.tensor(
        [[1.2, -0.5, 0.8]],
        dtype=DTYPE,
    )

    e_ij = head(
        quaternions=q,
        edge_index=torch.tensor(
            [[0], [1]],
            dtype=torch.long,
        ),
        edge_vectors=vector,
    )

    e_ji = head(
        quaternions=q,
        edge_index=torch.tensor(
            [[1], [0]],
            dtype=torch.long,
        ),
        edge_vectors=-vector,
    )

    torch.testing.assert_close(
        e_ij,
        e_ji,
        atol=1.0e-12,
        rtol=1.0e-12,
    )
