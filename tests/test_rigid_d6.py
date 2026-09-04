import math

import torch

from mace.modules.rigid_d6 import (
    D6_BODY_IRREPS,
    d6_body_features,
    d6_body_features_from_matrix,
)


DTYPE = torch.float64
ATOL = 2.0e-8
RTOL = 2.0e-8


def _rz(angle):
    c = math.cos(angle)
    s = math.sin(angle)

    return torch.tensor(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=DTYPE,
    )


def _c2_in_plane(angle):
    axis = torch.tensor(
        [
            math.cos(angle),
            math.sin(angle),
            0.0,
        ],
        dtype=DTYPE,
    )

    eye = torch.eye(3, dtype=DTYPE)

    return (
        2.0 * torch.outer(axis, axis)
        - eye
    )


def _d6_group():
    rotations = [
        _rz(k * math.pi / 3.0)
        for k in range(6)
    ]

    flips = [
        _c2_in_plane(k * math.pi / 6.0)
        for k in range(6)
    ]

    return rotations + flips


def _axis_angle(axis, angle):
    axis = torch.tensor(axis, dtype=DTYPE)
    axis = axis / torch.linalg.norm(axis)

    x, y, z = axis
    zero = torch.zeros((), dtype=DTYPE)

    K = torch.stack(
        (
            torch.stack((zero, -z, y)),
            torch.stack((z, zero, -x)),
            torch.stack((-y, x, zero)),
        )
    )

    eye = torch.eye(3, dtype=DTYPE)

    return (
        eye
        + math.sin(angle) * K
        + (1.0 - math.cos(angle)) * (K @ K)
    )


def test_d6_body_irreps():
    assert D6_BODY_IRREPS.dim == 18
    assert str(D6_BODY_IRREPS) == "1x2e+1x6e"


def test_d6_body_quaternion_wrapper_matches_identity_matrix():
    q = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0]],
        dtype=DTYPE,
    )

    via_q = d6_body_features(q)

    via_R = d6_body_features_from_matrix(
        torch.eye(3, dtype=DTYPE)[None, :, :]
    )

    torch.testing.assert_close(
        via_q,
        via_R,
        atol=ATOL,
        rtol=RTOL,
    )


def test_d6_body_has_exact_physical_d6_quotient():
    R = _axis_angle(
        [0.4, -0.7, 1.2],
        0.83,
    )

    reference = d6_body_features_from_matrix(
        R[None, :, :]
    )

    for g in _d6_group():
        transformed = d6_body_features_from_matrix(
            (R @ g)[None, :, :]
        )

        torch.testing.assert_close(
            transformed,
            reference,
            atol=ATOL,
            rtol=RTOL,
        )


def test_d6_body_does_not_overquotient_axial_rotation():
    R = _axis_angle(
        [0.4, -0.7, 1.2],
        0.83,
    )

    reference = d6_body_features_from_matrix(
        R[None, :, :]
    )

    rotated_30 = d6_body_features_from_matrix(
        (R @ _rz(math.pi / 6.0))[None, :, :]
    )

    rotated_60 = d6_body_features_from_matrix(
        (R @ _rz(math.pi / 3.0))[None, :, :]
    )

    torch.testing.assert_close(
        rotated_60,
        reference,
        atol=ATOL,
        rtol=RTOL,
    )

    difference = torch.linalg.norm(
        rotated_30 - reference
    ).item()

    assert difference > 1.0e-3


def test_d6_body_is_globally_equivariant():
    R = _axis_angle(
        [0.4, -0.7, 1.2],
        0.83,
    )

    S = _axis_angle(
        [-0.8, 0.3, 0.5],
        1.17,
    )

    before = d6_body_features_from_matrix(
        R[None, :, :]
    )

    after = d6_body_features_from_matrix(
        (S @ R)[None, :, :]
    )

    D = D6_BODY_IRREPS.D_from_matrix(S)

    expected = before @ D.T

    torch.testing.assert_close(
        after,
        expected,
        atol=ATOL,
        rtol=RTOL,
    )
