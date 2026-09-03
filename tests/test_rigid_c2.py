import torch
from e3nn import o3

from mace.modules.rigid_c2 import (
    C2_BODY_IRREPS,
    c2_body_irreducible_features,
)


DTYPE = torch.float64
ATOL = 1.0e-10
RTOL = 1.0e-10


def rotation():
    return o3.rand_matrix().to(dtype=DTYPE)


def pi_about(axis):
    signs = -torch.ones(
        3,
        dtype=DTYPE,
    )
    signs[axis] = 1.0
    return torch.diag(signs)


def test_c2_embedding_has_exact_physical_c2_quotient():
    c2_axis = 0
    R = rotation()

    feature = c2_body_irreducible_features(
        R,
        c2_axis,
    )

    # True physical C2 operation:
    #   a ->  a
    #   b -> -b
    #   c -> -c
    true_alias = (
        R
        @ pi_about(c2_axis)
    )

    true_feature = (
        c2_body_irreducible_features(
            true_alias,
            c2_axis,
        )
    )

    torch.testing.assert_close(
        true_feature,
        feature,
        atol=ATOL,
        rtol=RTOL,
    )

    # The other two D2 operations are MOI aliases,
    # but they are NOT physical C2 symmetries.
    for false_axis in (1, 2):
        false_alias = (
            R
            @ pi_about(false_axis)
        )

        false_feature = (
            c2_body_irreducible_features(
                false_alias,
                c2_axis,
            )
        )

        assert not torch.allclose(
            false_feature,
            feature,
            atol=ATOL,
            rtol=RTOL,
        )


def test_c2_embedding_is_globally_equivariant():
    c2_axis = 0

    R = rotation()
    S = rotation()

    feature = c2_body_irreducible_features(
        R,
        c2_axis,
    )

    rotated_feature = (
        c2_body_irreducible_features(
            S @ R,
            c2_axis,
        )
    )

    D = C2_BODY_IRREPS.D_from_matrix(
        S
    ).to(dtype=DTYPE)

    expected = feature @ D.T

    # e3nn's Wigner-D / spherical-harmonic evaluation can
    # accumulate ~1e-7 numerical error even for float64 inputs.
    torch.testing.assert_close(
        rotated_feature,
        expected,
        atol=1.0e-6,
        rtol=1.0e-6,
    )


def test_c2_embedding_breaks_false_moi_aliases():
    c2_axis = 0
    R = rotation()

    # Generic triaxial body inertia.
    I_body = torch.diag(
        torch.tensor(
            [1.0, 2.0, 3.0],
            dtype=DTYPE,
        )
    )

    I_lab = (
        R
        @ I_body
        @ R.T
    )

    feature = c2_body_irreducible_features(
        R,
        c2_axis,
    )

    for axis in (0, 1, 2):
        G = pi_about(axis)
        R_alias = R @ G

        I_alias = (
            R_alias
            @ I_body
            @ R_alias.T
        )

        # MOI identifies every member of D2.
        torch.testing.assert_close(
            I_alias,
            I_lab,
            atol=ATOL,
            rtol=RTOL,
        )

        alias_feature = (
            c2_body_irreducible_features(
                R_alias,
                c2_axis,
            )
        )

        if axis == c2_axis:
            # The genuine molecular symmetry is retained.
            torch.testing.assert_close(
                alias_feature,
                feature,
                atol=ATOL,
                rtol=RTOL,
            )
        else:
            # The two accidental MOI symmetries are removed.
            assert not torch.allclose(
                alias_feature,
                feature,
                atol=ATOL,
                rtol=RTOL,
            )
