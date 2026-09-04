from __future__ import annotations

import math

import pytest
import torch
from e3nn import o3

import math


def _legacy_c2_body_irreducible_features(
    rotation_matrices: torch.Tensor,
    c2_axis: int,
) -> torch.Tensor:
    if c2_axis not in (0, 1, 2):
        raise ValueError("c2_axis must be one of 0, 1, or 2")

    transverse = [i for i in range(3) if i != c2_axis]

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

    return torch.cat(
        (
            a,
            (y2_b - y2_c) / math.sqrt(2.0),
        ),
        dim=-1,
    )


def _legacy_d6_body_features_from_matrix(
    rotation: torch.Tensor,
) -> torch.Tensor:
    normal = rotation[..., :, 2]

    plane = o3.spherical_harmonics(
        2,
        normal,
        normalize=True,
        normalization="component",
    )

    angles = torch.arange(
        6,
        dtype=rotation.dtype,
        device=rotation.device,
    ) * (math.pi / 3.0)

    body_hexagon = torch.stack(
        (
            torch.cos(angles),
            torch.sin(angles),
            torch.zeros_like(angles),
        ),
        dim=-1,
    )

    space_hexagon = torch.einsum(
        "...ij,kj->...ki",
        rotation,
        body_hexagon,
    )

    hexatic = o3.spherical_harmonics(
        6,
        space_hexagon,
        normalize=True,
        normalization="component",
    ).mean(dim=-2)

    return torch.cat(
        (
            plane,
            hexatic,
        ),
        dim=-1,
    )

from mace.modules.rigid_pair_tp import (
    RigidPairC2EdgeEmbedding,
    RigidPairD6EdgeEmbedding,
    RigidPairSymmetryEdgeEmbedding,
)
from mace.modules.rigid_symmetry import (
    SymmetryAdaptedBodyFeatures,
    cyclic_group_rotations,
    dihedral_group_rotations,
    invariant_irrep_basis,
)

DTYPE = torch.float64
ATOL = 1.0e-7
RTOL = 1.0e-7


def _c2_generic(c2_axis: int) -> SymmetryAdaptedBodyFeatures:
    identity = torch.eye(3, dtype=DTYPE)
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
            dtype=DTYPE,
        ),
        blocks=(
            ("1o", axis_seed),
            ("2e", quadrupole_seed),
        ),
    )


def _d6_generic() -> SymmetryAdaptedBodyFeatures:
    identity = torch.eye(3, dtype=DTYPE)
    normal = identity[:, 2]
    plane_seed = o3.spherical_harmonics(
        2,
        normal,
        normalize=True,
        normalization="component",
    )
    angles = torch.arange(6, dtype=DTYPE) * (math.pi / 3.0)
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
            dtype=DTYPE,
        ),
        blocks=(
            ("2e", plane_seed),
            ("6e", hexatic_seed),
        ),
    )


@pytest.mark.parametrize("c2_axis", (0, 1, 2))
def test_generic_c2_matches_existing_body_features(c2_axis: int) -> None:
    torch.manual_seed(1234 + c2_axis)
    rotations = o3.rand_matrix(16, dtype=DTYPE)
    generic = _c2_generic(c2_axis)
    expected = _legacy_c2_body_irreducible_features(rotations, c2_axis)
    torch.testing.assert_close(
        generic(rotations),
        expected,
        atol=ATOL,
        rtol=RTOL,
    )


def test_generic_d6_matches_existing_body_features() -> None:
    torch.manual_seed(1234)
    rotations = o3.rand_matrix(16, dtype=DTYPE)
    generic = _d6_generic()
    expected = _legacy_d6_body_features_from_matrix(rotations)
    torch.testing.assert_close(
        generic(rotations),
        expected,
        atol=ATOL,
        rtol=RTOL,
    )


def test_generic_features_are_invariant_to_right_group_action() -> None:
    group = dihedral_group_rotations(4, dtype=DTYPE)
    seeds = invariant_irrep_basis(group, "4e")
    assert seeds.shape[0] > 0
    features = SymmetryAdaptedBodyFeatures(
        group_rotations=group,
        blocks=(("4e", seeds),),
        project_seeds=False,
    )
    torch.manual_seed(1234)
    rotations = o3.rand_matrix(5, dtype=DTYPE)
    reference = features(rotations)
    for group_rotation in group:
        transformed = features(rotations @ group_rotation)
        torch.testing.assert_close(
            transformed,
            reference,
            atol=ATOL,
            rtol=RTOL,
        )


def test_generic_features_remain_globally_equivariant() -> None:
    features = _d6_generic()
    torch.manual_seed(1234)
    rotations = o3.rand_matrix(4, dtype=DTYPE)
    global_rotation = o3.rand_matrix(dtype=DTYPE)
    before = features(rotations)
    after = features(global_rotation @ rotations)
    irreps_matrix = features.irreps.D_from_matrix(global_rotation)
    expected = torch.einsum("ij,nj->ni", irreps_matrix, before)
    torch.testing.assert_close(
        after,
        expected,
        atol=ATOL,
        rtol=RTOL,
    )


def _pair_inputs():
    torch.manual_seed(4321)
    quaternions = torch.randn(5, 4, dtype=DTYPE)
    quaternions = quaternions / torch.linalg.vector_norm(
        quaternions,
        dim=-1,
        keepdim=True,
    )
    edge_index = torch.tensor(
        (
            (0, 1, 2, 3, 4, 1),
            (1, 2, 3, 4, 0, 4),
        ),
        dtype=torch.long,
    )
    edge_vectors = torch.randn(edge_index.shape[1], 3, dtype=DTYPE)
    return quaternions, edge_index, edge_vectors


def test_generic_pair_embedding_matches_existing_c2():
    edge_irreps = o3.Irreps.spherical_harmonics(2)

    torch.manual_seed(1234)
    existing = RigidPairC2EdgeEmbedding(
        lmax=2,
        edge_irreps=edge_irreps,
        multiplicity=2,
        c2_axis=1,
    ).to(dtype=DTYPE)

    torch.manual_seed(1234)
    generic = RigidPairSymmetryEdgeEmbedding(
        body_features=_c2_generic(1),
        lmax=2,
        edge_irreps=edge_irreps,
        multiplicity=2,
        restrict_pair_irreps=False,
    ).to(dtype=DTYPE)

    assert generic.body_irreps == existing.edge_body_tp.irreps_in2
    assert generic.irreps_in == existing.irreps_in
    assert generic.edge_irreps == existing.edge_irreps

    torch.testing.assert_close(
        generic.projection.weight,
        existing.projection.weight,
        atol=0.0,
        rtol=0.0,
    )

    quaternions, edge_index, edge_vectors = _pair_inputs()

    torch.testing.assert_close(
        generic(quaternions, edge_index, edge_vectors),
        existing(quaternions, edge_index, edge_vectors),
        atol=ATOL,
        rtol=RTOL,
    )


def test_generic_pair_embedding_matches_existing_d6():
    edge_irreps = o3.Irreps.spherical_harmonics(2)

    torch.manual_seed(1234)
    existing = RigidPairD6EdgeEmbedding(
        lmax=2,
        edge_irreps=edge_irreps,
        multiplicity=2,
    ).to(dtype=DTYPE)

    torch.manual_seed(1234)
    generic = RigidPairSymmetryEdgeEmbedding(
        body_features=_d6_generic(),
        lmax=2,
        edge_irreps=edge_irreps,
        multiplicity=2,
        restrict_pair_irreps=True,
    ).to(dtype=DTYPE)

    assert generic.body_irreps == existing.edge_body_tp.irreps_in2
    assert generic.irreps_in == existing.irreps_in
    assert generic.edge_irreps == existing.edge_irreps

    torch.testing.assert_close(
        generic.projection.weight,
        existing.projection.weight,
        atol=0.0,
        rtol=0.0,
    )

    quaternions, edge_index, edge_vectors = _pair_inputs()

    torch.testing.assert_close(
        generic(quaternions, edge_index, edge_vectors),
        existing(quaternions, edge_index, edge_vectors),
        atol=ATOL,
        rtol=RTOL,
    )


@pytest.mark.parametrize(
    ("symmetry", "lmax"),
    (
        ("C3", 4),
        ("C4", 4),
        ("D3", 4),
        ("D4", 4),
    ),
)
def test_automatic_named_symmetry_features_respect_body_group(
    symmetry,
    lmax,
):
    from mace.modules.rigid_symmetry import (
        automatic_named_symmetry_body_features,
        symmetry_group_rotations,
    )

    features = automatic_named_symmetry_body_features(
        symmetry,
        lmax=lmax,
        dtype=DTYPE,
    )
    group = symmetry_group_rotations(
        symmetry,
        dtype=DTYPE,
    )

    torch.manual_seed(8317)
    rotations = o3.rand_matrix(5, dtype=DTYPE)
    reference = features(rotations)

    for group_rotation in group:
        torch.testing.assert_close(
            features(rotations @ group_rotation),
            reference,
            atol=ATOL,
            rtol=RTOL,
        )


def test_automatic_symmetry_features_remain_globally_equivariant():
    from mace.modules.rigid_symmetry import (
        automatic_named_symmetry_body_features,
    )

    features = automatic_named_symmetry_body_features(
        "D4",
        lmax=4,
        dtype=DTYPE,
    )

    torch.manual_seed(9321)
    rotations = o3.rand_matrix(5, dtype=DTYPE)
    global_rotation = o3.rand_matrix(dtype=DTYPE)

    before = features(rotations)
    after = features(global_rotation @ rotations)
    expected = torch.einsum(
        "ij,nj->ni",
        features.irreps.D_from_matrix(global_rotation),
        before,
    )

    torch.testing.assert_close(
        after,
        expected,
        atol=ATOL,
        rtol=RTOL,
    )


@pytest.mark.parametrize(
    ("embedding_cls", "kwargs"),
    (
        (RigidPairC2EdgeEmbedding, {"c2_axis": 1}),
        (RigidPairD6EdgeEmbedding, {}),
    ),
)
def test_legacy_rigid_pair_wrapper_pickle_is_migrated(
    embedding_cls,
    kwargs,
):
    import io

    torch.manual_seed(1234)
    module = embedding_cls(
        lmax=2,
        edge_irreps=o3.Irreps.spherical_harmonics(2),
        multiplicity=2,
        **kwargs,
    ).to(dtype=DTYPE)

    quaternions, edge_index, edge_vectors = _pair_inputs()
    reference = module(
        quaternions,
        edge_index,
        edge_vectors,
    )

    del module._modules["body_features"]
    module.__dict__.pop("body_irreps", None)
    module.__dict__.pop("restrict_pair_irreps", None)

    buffer = io.BytesIO()
    torch.save(module, buffer)
    buffer.seek(0)

    loaded = torch.load(
        buffer,
        map_location="cpu",
        weights_only=False,
    )

    restored = loaded(
        quaternions,
        edge_index,
        edge_vectors,
    )

    torch.testing.assert_close(
        restored,
        reference,
        atol=ATOL,
        rtol=RTOL,
    )

@pytest.mark.parametrize(
    "embedding_cls,kwargs",
    [
        (RigidPairC2EdgeEmbedding, {"c2_axis": 1}),
        (RigidPairD6EdgeEmbedding, {}),
    ],
)
def test_legacy_rigid_pair_wrapper_state_dict_is_migrated(
    embedding_cls,
    kwargs,
):
    edge_irreps = o3.Irreps.spherical_harmonics(2)

    reference = embedding_cls(
        lmax=2,
        edge_irreps=edge_irreps,
        multiplicity=2,
        **kwargs,
    ).to(dtype=DTYPE)

    legacy_state = {
        key: value.clone()
        for key, value in reference.state_dict().items()
        if not key.startswith("body_features.")
    }

    restored = embedding_cls(
        lmax=2,
        edge_irreps=edge_irreps,
        multiplicity=2,
        **kwargs,
    ).to(dtype=DTYPE)

    restored.load_state_dict(legacy_state, strict=True)

    reference_state = reference.state_dict()
    restored_state = restored.state_dict()

    assert reference_state.keys() == restored_state.keys()

    for key in reference_state:
        torch.testing.assert_close(
            restored_state[key],
            reference_state[key],
        )


def _pi_about_axis(axis: int) -> torch.Tensor:
    signs = -torch.ones(3, dtype=DTYPE)
    signs[axis] = 1.0
    return torch.diag(signs)


def _rz(angle: float) -> torch.Tensor:
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


def test_generic_c2_breaks_false_moi_aliases() -> None:
    c2_axis = 0
    torch.manual_seed(1234)
    rotation = o3.rand_matrix(dtype=DTYPE)

    features = _c2_generic(c2_axis)
    reference = features(rotation)

    inertia_body = torch.diag(
        torch.tensor([1.0, 2.0, 3.0], dtype=DTYPE)
    )
    inertia_lab = rotation @ inertia_body @ rotation.T

    for axis in (0, 1, 2):
        group_rotation = _pi_about_axis(axis)
        alias_rotation = rotation @ group_rotation

        inertia_alias = (
            alias_rotation
            @ inertia_body
            @ alias_rotation.T
        )

        torch.testing.assert_close(
            inertia_alias,
            inertia_lab,
            atol=1.0e-10,
            rtol=1.0e-10,
        )

        alias_feature = features(alias_rotation)

        if axis == c2_axis:
            torch.testing.assert_close(
                alias_feature,
                reference,
                atol=ATOL,
                rtol=RTOL,
            )
        else:
            assert not torch.allclose(
                alias_feature,
                reference,
                atol=ATOL,
                rtol=RTOL,
            )


def test_generic_d6_does_not_overquotient_axial_rotation() -> None:
    torch.manual_seed(1234)
    rotation = o3.rand_matrix(dtype=DTYPE)

    features = _d6_generic()
    reference = features(rotation)

    rotated_30 = features(rotation @ _rz(math.pi / 6.0))
    rotated_60 = features(rotation @ _rz(math.pi / 3.0))

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
