import math

import torch
from e3nn import o3

from mace.modules.rigid_pair_tp import RigidPairD6EdgeEmbedding


def _axis_angle_quaternion(axis, angle):
    axis = torch.as_tensor(axis, dtype=torch.float64)
    axis = axis / torch.linalg.vector_norm(axis)
    half = 0.5 * angle
    return torch.cat(
        (
            torch.tensor(
                [math.cos(half)],
                dtype=torch.float64,
            ),
            axis * math.sin(half),
        )
    )


def _quaternion_product(a, b):
    aw, ax, ay, az = a.unbind()
    bw, bx, by, bz = b.unbind()

    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        )
    )


def _rotation_matrix(axis, angle):
    axis = torch.as_tensor(axis, dtype=torch.float64)
    axis = axis / torch.linalg.vector_norm(axis)

    x, y, z = axis
    zero = torch.zeros((), dtype=torch.float64)

    K = torch.stack(
        (
            torch.stack((zero, -z, y)),
            torch.stack((z, zero, -x)),
            torch.stack((-y, x, zero)),
        )
    )

    eye = torch.eye(3, dtype=torch.float64)

    return (
        eye
        + math.sin(angle) * K
        + (1.0 - math.cos(angle)) * (K @ K)
    )


def _d6_quaternions():
    result = []

    for k in range(6):
        result.append(
            _axis_angle_quaternion(
                (0.0, 0.0, 1.0),
                k * math.pi / 3.0,
            )
        )

    for k in range(6):
        phi = k * math.pi / 6.0
        result.append(
            _axis_angle_quaternion(
                (
                    math.cos(phi),
                    math.sin(phi),
                    0.0,
                ),
                math.pi,
            )
        )

    return result


def _example():
    q0 = _axis_angle_quaternion(
        (1.0, 2.0, -1.0),
        0.71,
    )
    q1 = _axis_angle_quaternion(
        (-2.0, 1.0, 3.0),
        -0.93,
    )

    quaternions = torch.stack((q0, q1))

    edge_index = torch.tensor(
        [[0, 1], [1, 0]],
        dtype=torch.long,
    )

    edge_vectors = torch.tensor(
        [
            [1.2, -0.7, 0.9],
            [-1.2, 0.7, -0.9],
        ],
        dtype=torch.float64,
    )

    return quaternions, edge_index, edge_vectors


def test_d6_pair_output_irreps():
    edge_irreps = o3.Irreps.spherical_harmonics(3)

    model = RigidPairD6EdgeEmbedding(
        max_ell=3,
        edge_irreps=edge_irreps,
        multiplicity=1,
    ).double()

    assert model.irreps_out == edge_irreps
    assert model.irreps_out.dim == 16


def test_d6_pair_is_invariant_to_right_d6_actions():
    torch.manual_seed(7)

    model = RigidPairD6EdgeEmbedding(
        max_ell=3,
        multiplicity=1,
    ).double()

    quaternions, edge_index, edge_vectors = _example()

    reference = model(
        quaternions,
        edge_index,
        edge_vectors,
    )

    group = _d6_quaternions()

    for g0 in group:
        for g1 in group:
            transformed = torch.stack(
                (
                    _quaternion_product(
                        quaternions[0],
                        g0,
                    ),
                    _quaternion_product(
                        quaternions[1],
                        g1,
                    ),
                )
            )

            actual = model(
                transformed,
                edge_index,
                edge_vectors,
            )

            torch.testing.assert_close(
                actual,
                reference,
                atol=1.0e-9,
                rtol=1.0e-9,
            )


def test_d6_pair_is_globally_equivariant():
    torch.manual_seed(11)

    model = RigidPairD6EdgeEmbedding(
        max_ell=3,
        multiplicity=1,
    ).double()

    quaternions, edge_index, edge_vectors = _example()

    reference = model(
        quaternions,
        edge_index,
        edge_vectors,
    )

    axis = (1.0, -2.0, 0.5)
    angle = 0.63

    q_global = _axis_angle_quaternion(
        axis,
        angle,
    )
    rotation = _rotation_matrix(
        axis,
        angle,
    )

    transformed_q = torch.stack(
        [
            _quaternion_product(
                q_global,
                q,
            )
            for q in quaternions
        ]
    )

    transformed_vectors = edge_vectors @ rotation.T

    actual = model(
        transformed_q,
        edge_index,
        transformed_vectors,
    )

    D = model.irreps_out.D_from_matrix(
        rotation
    )

    expected = reference @ D.T

    torch.testing.assert_close(
        actual,
        expected,
        atol=2.0e-7,
        rtol=1.0e-6,
    )


def test_d6_pair_changes_under_non_d6_axial_rotation():
    torch.manual_seed(13)

    model = RigidPairD6EdgeEmbedding(
        max_ell=3,
        multiplicity=1,
    ).double()

    quaternions, edge_index, edge_vectors = _example()

    reference = model(
        quaternions,
        edge_index,
        edge_vectors,
    )

    g = _axis_angle_quaternion(
        (0.0, 0.0, 1.0),
        math.pi / 6.0,
    )

    transformed = quaternions.clone()
    transformed[0] = _quaternion_product(
        transformed[0],
        g,
    )

    actual = model(
        transformed,
        edge_index,
        edge_vectors,
    )

    delta = torch.linalg.vector_norm(
        actual - reference
    )

    assert delta.item() > 1.0e-6
