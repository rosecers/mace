from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from ase import Atoms
from e3nn import o3
from scipy.spatial.transform import Rotation

from mace import data, modules, tools

DTYPE = torch.float64
ATOL = 1.0e-7
RTOL = 1.0e-7

TABLE = tools.AtomicNumberTable([0])
CUTOFF = 5.0


def _wxyz(rotation: Rotation) -> np.ndarray:
    x, y, z, w = rotation.as_quat()
    return np.asarray([w, x, y, z], dtype=float)


def _atoms(
    body_i: Rotation,
    body_j: Rotation,
    positions=None,
):
    if positions is None:
        positions = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [2.1, 0.7, -0.4],
            ],
            dtype=float,
        )

    atoms = Atoms(
        "XX",
        positions=np.asarray(positions, dtype=float),
    )

    atoms.arrays["quaternions"] = np.stack(
        [
            _wxyz(body_i),
            _wxyz(body_j),
        ]
    )

    # Supply the same rigid-body fields already used by the existing
    # rank-2 AtomicData tests. Even with rigid_feature_mode="none",
    # the current MACE data path still constructs these tensors.
    atoms.arrays["c_diameter[1]"] = np.asarray([2.0, 2.0])
    atoms.arrays["c_diameter[2]"] = np.asarray([3.0, 3.0])
    atoms.arrays["c_diameter[3]"] = np.asarray([4.0, 4.0])

    return atoms


def _batch(atoms):
    config = data.config_from_atoms(
        atoms,
        config_type_weights={"Default": 1.0},
    )

    graph = data.AtomicData.from_config(
        config,
        z_table=TABLE,
        cutoff=CUTOFF,
        heads=["Default"],
    )

    loader = tools.torch_geometric.dataloader.DataLoader(
        dataset=[graph],
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )

    batch = next(iter(loader)).to_dict()

    # The model below is explicitly converted to DTYPE. AtomicData/PyG
    # may otherwise leave fields such as node_attrs in float32, which
    # causes an unrelated dtype failure in the ordinary MACE embedding
    # before the rigid-pair path is reached.
    for key, value in batch.items():
        if torch.is_tensor(value) and value.is_floating_point():
            batch[key] = value.to(dtype=DTYPE)

    return batch


def _model(rigid_pair_mode: str, rigid_pair_multiplicity: int = 1):
    torch.manual_seed(1234)

    model = modules.ScaleShiftMACE(
        r_max=CUTOFF,
        num_bessel=4,
        num_polynomial_cutoff=5,
        max_ell=2,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticInteractionBlock"
        ],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticInteractionBlock"
        ],
        num_interactions=2,
        num_elements=1,
        hidden_irreps=o3.Irreps(
            "8x0e + 8x1o + 8x2e"
        ),
        MLP_irreps=o3.Irreps("8x0e"),
        gate=F.silu,
        atomic_energies=np.asarray([0.0]),
        avg_num_neighbors=1.0,
        atomic_numbers=TABLE.zs,
        correlation=2,
        radial_type="bessel",
        atomic_inter_scale=1.0,
        atomic_inter_shift=0.0,
        rigid_feature_mode="none",
        rigid_pair_mode=rigid_pair_mode,
        rigid_pair_multiplicity=rigid_pair_multiplicity,
    )

    return model.to(dtype=DTYPE)


def test_none_mode_has_no_rigid_pair_embedding():
    model = _model("none")

    assert model.rigid_pair_mode == "none"
    assert not hasattr(
        model,
        "rigid_pair_edge_embedding",
    )


def test_full_frame_mode_constructs_rigid_pair_embedding():
    model = _model("full_frame")

    assert model.rigid_pair_mode == "full_frame"
    assert hasattr(
        model,
        "rigid_pair_edge_embedding",
    )

    assert (
        model.rigid_pair_edge_embedding.edge_irreps
        == model.spherical_harmonics.irreps_out
    )


def test_none_mode_is_orientation_blind():
    model = _model("none")
    model.eval()

    identity = Rotation.identity()

    axis = np.asarray([0.3, 1.0, -0.2], dtype=float)
    axis /= np.linalg.norm(axis)

    rotated_neighbor = Rotation.from_rotvec(
        axis * 0.83
    )

    out_a = model(
        _batch(_atoms(identity, identity)),
        compute_force=False,
    )

    out_b = model(
        _batch(_atoms(identity, rotated_neighbor)),
        compute_force=False,
    )

    torch.testing.assert_close(
        out_a["energy"],
        out_b["energy"],
        atol=ATOL,
        rtol=RTOL,
    )


def test_full_frame_mode_detects_neighbor_rotation():
    model = _model("full_frame")
    model.eval()

    identity = Rotation.identity()

    axis = np.asarray([0.3, 1.0, -0.2], dtype=float)
    axis /= np.linalg.norm(axis)

    rotated_neighbor = Rotation.from_rotvec(
        axis * 0.83
    )

    out_a = model(
        _batch(_atoms(identity, identity)),
        compute_force=False,
    )

    out_b = model(
        _batch(_atoms(identity, rotated_neighbor)),
        compute_force=False,
    )

    delta = torch.max(
        torch.abs(
            out_a["energy"] - out_b["energy"]
        )
    )

    assert delta > 1.0e-10


def test_full_frame_energy_is_globally_rotation_invariant():
    model = _model("full_frame")
    model.eval()

    body_i = Rotation.from_rotvec(
        np.asarray([0.4, -0.2, 0.7])
    )

    body_j = Rotation.from_rotvec(
        np.asarray([-0.3, 0.6, 0.2])
    )

    global_rotation = Rotation.from_rotvec(
        np.asarray([0.2, 0.5, -0.8])
    )

    positions = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [2.1, 0.7, -0.4],
        ],
        dtype=float,
    )

    out_a = model(
        _batch(
            _atoms(
                body_i,
                body_j,
                positions=positions,
            )
        ),
        compute_force=False,
    )

    out_b = model(
        _batch(
            _atoms(
                global_rotation * body_i,
                global_rotation * body_j,
                positions=global_rotation.apply(positions),
            )
        ),
        compute_force=False,
    )

    torch.testing.assert_close(
        out_a["energy"],
        out_b["energy"],
        atol=ATOL,
        rtol=RTOL,
    )


def test_full_frame_forces_rotate_covariantly():
    model = _model("full_frame")
    model.eval()

    body_i = Rotation.from_rotvec(
        np.asarray([0.4, -0.2, 0.7])
    )

    body_j = Rotation.from_rotvec(
        np.asarray([-0.3, 0.6, 0.2])
    )

    global_rotation = Rotation.from_rotvec(
        np.asarray([0.2, 0.5, -0.8])
    )

    positions = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [2.1, 0.7, -0.4],
        ],
        dtype=float,
    )

    out_a = model(
        _batch(
            _atoms(
                body_i,
                body_j,
                positions=positions,
            )
        ),
        compute_force=True,
    )

    out_b = model(
        _batch(
            _atoms(
                global_rotation * body_i,
                global_rotation * body_j,
                positions=global_rotation.apply(positions),
            )
        ),
        compute_force=True,
    )

    S = torch.tensor(
        global_rotation.as_matrix(),
        dtype=DTYPE,
    )

    expected = out_a["forces"] @ S.T

    torch.testing.assert_close(
        out_b["forces"],
        expected,
        atol=ATOL,
        rtol=RTOL,
    )


def test_rigid_pair_projection_gets_gradient():
    model = _model("full_frame")
    model.train()

    body_i = Rotation.identity()
    body_j = Rotation.from_rotvec(
        np.asarray([0.2, 0.4, -0.1])
    )

    output = model(
        _batch(_atoms(body_i, body_j)),
        compute_force=False,
    )

    loss = output["energy"].sum()
    loss.backward()

    parameters = list(
        model.rigid_pair_edge_embedding
        .projection.parameters()
    )

    assert parameters
    assert all(
        parameter.grad is not None
        for parameter in parameters
    )

    total_gradient = sum(
        parameter.grad.abs().sum()
        for parameter in parameters
    )

    assert total_gradient > 0.0



def test_full_frame_edge_irreps_are_concatenated():
    """Geometric and rigid-pair channels remain separate."""
    model = _model("full_frame")

    sh_irreps = model.spherical_harmonics.irreps_out
    rigid_irreps = model.rigid_pair_edge_embedding.edge_irreps

    expected = sh_irreps + rigid_irreps

    assert model.edge_attrs_irreps == expected
    assert model.edge_attrs_irreps.dim == (
        sh_irreps.dim + rigid_irreps.dim
    )

    assert rigid_irreps == sh_irreps
    assert model.edge_attrs_irreps.dim == 2 * sh_irreps.dim


def test_none_mode_keeps_legacy_edge_irreps():
    """Turning rigid-pair features off leaves MACE edge irreps unchanged."""
    model = _model("none")

    sh_irreps = model.spherical_harmonics.irreps_out

    assert model.edge_attrs_irreps == sh_irreps
    assert model.edge_attrs_irreps.dim == sh_irreps.dim


def test_forward_keeps_geometric_and_rigid_edge_channels_distinct():
    """Changing orientation changes only the rigid half of edge attrs."""
    model = _model("full_frame")
    model.eval()

    identity = Rotation.identity()

    axis = np.asarray(
        [0.3, 1.0, -0.2],
        dtype=float,
    )
    axis /= np.linalg.norm(axis)

    rotated_neighbor = Rotation.from_rotvec(axis * 0.83)

    captured = []

    def capture_edge_attrs(_module, _args, kwargs):
        if "edge_attrs" not in kwargs:
            raise RuntimeError(
                f"Interaction kwargs do not contain edge_attrs: "
                f"{sorted(kwargs)}"
            )

        edge_attrs = kwargs["edge_attrs"]

        if edge_attrs.shape[-1] != model.edge_attrs_irreps.dim:
            raise RuntimeError(
                f"Expected edge_attrs dim "
                f"{model.edge_attrs_irreps.dim}, "
                f"got {edge_attrs.shape[-1]}"
            )

        captured.append(edge_attrs.detach().clone())

    handle = model.interactions[0].register_forward_pre_hook(
        capture_edge_attrs,
        with_kwargs=True,
    )

    try:
        model(
            _batch(_atoms(identity, identity)),
            compute_force=False,
        )

        model(
            _batch(_atoms(identity, rotated_neighbor)),
            compute_force=False,
        )
    finally:
        handle.remove()

    assert len(captured) == 2

    edge_a, edge_b = captured

    sh_dim = model.spherical_harmonics.irreps_out.dim

    assert edge_a.shape[-1] == 2 * sh_dim
    assert edge_b.shape[-1] == 2 * sh_dim

    # Positions did not change, so ordinary SH must not change.
    torch.testing.assert_close(
        edge_a[:, :sh_dim],
        edge_b[:, :sh_dim],
        atol=ATOL,
        rtol=RTOL,
    )

    # Neighbor orientation changed, so rigid-pair channels must change.
    delta = torch.max(
        torch.abs(
            edge_a[:, sh_dim:]
            - edge_b[:, sh_dim:]
        )
    )

    assert delta > 1.0e-10



def test_full_frame_raw_uses_uncompressed_pair_irreps():
    model = _model("full_frame_raw")

    sh_irreps = model.spherical_harmonics.irreps_out
    raw = model.rigid_pair_edge_embedding

    # Raw mode has no projection.
    assert raw.edge_irreps == raw.full_pair.irreps_out

    # Each of the two molecular frames has dimension 9.
    assert raw.edge_irreps.dim == 81 * sh_irreps.dim

    # MACE receives ordinary geometric SH plus the complete raw basis.
    assert model.edge_attrs_irreps == (
        sh_irreps + raw.edge_irreps
    )

    assert model.edge_attrs_irreps.dim == (
        sh_irreps.dim + raw.edge_irreps.dim
    )


def test_full_frame_multiplicity_four_expands_rigid_channels():
    model = _model(
        "full_frame",
        rigid_pair_multiplicity=4,
    )

    sh_irreps = model.spherical_harmonics.irreps_out
    rigid = model.rigid_pair_edge_embedding

    assert model.rigid_pair_multiplicity == 4
    assert rigid.multiplicity == 4
    assert rigid.base_edge_irreps == sh_irreps

    assert rigid.edge_irreps.dim == 4 * sh_irreps.dim

    # ordinary SH + four learned rigid copies
    assert model.edge_attrs_irreps.dim == (
        5 * sh_irreps.dim
    )





def test_irrep_complete_mode_has_all_raw_irrep_types():
    model = _model("full_frame_irrep_complete")

    sh_irreps = model.spherical_harmonics.irreps_out
    rigid = model.rigid_pair_edge_embedding

    raw_types = {
        (ir.l, ir.p)
        for _, ir in rigid.full_pair.irreps_out
    }

    compact_types = {
        (ir.l, ir.p)
        for _, ir in rigid.edge_irreps
    }

    assert compact_types == raw_types

    # Integration responsibility: ordinary geometric SH and the
    # compact rigid branch must remain distinct concatenated blocks.
    assert model.edge_attrs_irreps == (
        sh_irreps + rigid.edge_irreps
    )
    assert model.edge_attrs_irreps.dim == (
        sh_irreps.dim + rigid.edge_irreps.dim
    )


def test_irrep_complete_energy_is_globally_rotation_invariant():
    model = _model("full_frame_irrep_complete")
    model.eval()

    body_i = Rotation.from_rotvec(
        np.asarray([0.4, -0.2, 0.7])
    )

    body_j = Rotation.from_rotvec(
        np.asarray([-0.3, 0.6, 0.2])
    )

    global_rotation = Rotation.from_rotvec(
        np.asarray([0.2, 0.5, -0.8])
    )

    positions = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [2.1, 0.7, -0.4],
        ],
        dtype=float,
    )

    out_a = model(
        _batch(
            _atoms(
                body_i,
                body_j,
                positions=positions,
            )
        ),
        compute_force=False,
    )

    out_b = model(
        _batch(
            _atoms(
                global_rotation * body_i,
                global_rotation * body_j,
                positions=global_rotation.apply(positions),
            )
        ),
        compute_force=False,
    )

    torch.testing.assert_close(
        out_a["energy"],
        out_b["energy"],
        atol=ATOL,
        rtol=RTOL,
    )



def test_invariant_radial_mode_keeps_standard_edge_irreps():
    model = _model("invariant_radial")

    sh_irreps = model.spherical_harmonics.irreps_out

    assert model.edge_attrs_irreps == sh_irreps

    conditioner = model.rigid_pair_radial_conditioning

    assert conditioner is not None
    assert conditioner.radial_dim == model.radial_embedding.out_dim


def test_invariant_radial_mode_detects_neighbor_rotation():
    model = _model("invariant_radial")

    # The FiLM layer intentionally starts at zero so the model is
    # exactly baseline MACE at initialization. Make the conditioning
    # nontrivial for this plumbing/sensitivity regression.
    with torch.no_grad():
        torch.manual_seed(123)
        last = model.rigid_pair_radial_conditioning.net[-1]
        last.weight.normal_(mean=0.0, std=0.1)
        last.bias.zero_()
    model.eval()

    identity = Rotation.identity()

    axis = np.asarray([0.3, 1.0, -0.2], dtype=float)
    axis /= np.linalg.norm(axis)

    rotated_neighbor = Rotation.from_rotvec(
        axis * 0.83
    )

    out_a = model(
        _batch(_atoms(identity, identity)),
        compute_force=False,
    )

    out_b = model(
        _batch(_atoms(identity, rotated_neighbor)),
        compute_force=False,
    )

    delta = torch.max(
        torch.abs(
            out_a["energy"] - out_b["energy"]
        )
    )

    assert delta > 1.0e-10


def test_invariant_radial_does_not_change_baseline_parameter_initialization():
    seed = 1729

    torch.manual_seed(seed)
    baseline = _model("none")

    torch.manual_seed(seed)
    conditioned = _model("invariant_radial")

    baseline_state = baseline.state_dict()
    conditioned_state = conditioned.state_dict()

    conditioner_prefix = "rigid_pair_radial_conditioning."

    common_keys = sorted(
        key
        for key in baseline_state
        if (
            key in conditioned_state
            and not key.startswith(conditioner_prefix)
        )
    )

    assert common_keys

    for key in common_keys:
        a = baseline_state[key]
        b = conditioned_state[key]

        assert a.shape == b.shape, key

        torch.testing.assert_close(
            a,
            b,
            atol=0.0,
            rtol=0.0,
            msg=lambda msg, key=key: (
                f"baseline initialization changed for {key}: {msg}"
            ),
        )


def test_invariant_radial_survives_model_serialization():
    import io

    model = _model("invariant_radial")

    conditioner = model.rigid_pair_radial_conditioning
    assert conditioner is not None

    # Make its state visibly non-default so we verify preservation,
    # not merely recreation.
    with torch.no_grad():
        last = conditioner.net[-1]
        last.weight.fill_(0.123456789)
        last.bias.fill_(-0.03125)

    buffer = io.BytesIO()
    torch.save(model, buffer)
    buffer.seek(0)

    loaded = torch.load(
        buffer,
        map_location="cpu",
        weights_only=False,
    )

    restored = loaded.rigid_pair_radial_conditioning

    assert restored is not None

    torch.testing.assert_close(
        restored.net[-1].weight,
        conditioner.net[-1].weight,
        atol=0.0,
        rtol=0.0,
    )

    torch.testing.assert_close(
        restored.net[-1].bias,
        conditioner.net[-1].bias,
        atol=0.0,
        rtol=0.0,
    )


def test_rigid_pair_modules_are_optimized_and_film_updates():
    from types import SimpleNamespace

    from mace.tools.scripts_utils import get_params_options

    args = SimpleNamespace(
        lr=1.0e-3,
        weight_decay=5.0e-7,
        amsgrad=False,
        beta=0.9,
        lr_params_factors="{}",
        freeze=0,
    )

    # Learned rigid-pair modules must be present exactly once in the
    # optimizer.  This protects both the invariant FiLM branch and the
    # learned equivariant projection branches.
    cases = (
        (
            "invariant_radial",
            "rigid_pair_radial_conditioning",
        ),
        (
            "full_frame",
            "rigid_pair_edge_embedding",
        ),
        (
            "full_frame_irrep_complete",
            "rigid_pair_edge_embedding",
        ),
    )

    for mode, attribute in cases:
        model = _model(mode)
        module = getattr(model, attribute)

        target_ids = {
            id(param)
            for param in module.parameters()
            if param.requires_grad
        }

        assert target_ids, mode

        options = get_params_options(args, model)

        grouped_ids = []

        for group in options["params"]:
            grouped_ids.extend(
                id(param)
                for param in group["params"]
            )

        for parameter_id in target_ids:
            assert grouped_ids.count(parameter_id) == 1, mode

    # Invariant FiLM starts as exactly the identity but must move after
    # a genuine optimizer step.
    torch.manual_seed(17)

    model = _model("invariant_radial")
    options = get_params_options(args, model)
    optimizer = torch.optim.Adam(**options)

    conditioner = model.rigid_pair_radial_conditioning
    last = conditioner.net[-1]

    assert torch.count_nonzero(last.weight).item() == 0
    assert torch.count_nonzero(last.bias).item() == 0

    dtype = last.weight.dtype
    device = last.weight.device

    q = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.9, 0.1, -0.2, 0.3],
        ],
        dtype=dtype,
        device=device,
    )
    q = q / torch.linalg.vector_norm(
        q,
        dim=-1,
        keepdim=True,
    )

    edge_index = torch.tensor(
        [[0], [1]],
        dtype=torch.long,
        device=device,
    )

    edge_vectors = torch.tensor(
        [[1.2, -0.4, 0.7]],
        dtype=dtype,
        device=device,
    )

    edge_feats = torch.randn(
        1,
        model.radial_embedding.out_dim,
        dtype=dtype,
        device=device,
    )

    optimizer.zero_grad()

    out = conditioner(
        edge_feats=edge_feats,
        quaternions=q,
        edge_index=edge_index,
        edge_vectors=edge_vectors,
    )

    out.square().sum().backward()

    assert last.weight.grad is not None
    assert last.weight.grad.norm().item() > 0.0

    optimizer.step()

    assert last.weight.detach().norm().item() > 0.0


def test_learned_rigid_pair_embedding_construction_is_rng_neutral():
    from e3nn import o3

    from mace.modules.rigid_pair_tp import (
        RigidPairEdgeEmbedding,
        RigidPairIrrepCompleteEdgeEmbedding,
    )

    sh_irreps = o3.Irreps.spherical_harmonics(3)

    constructors = (
        lambda: RigidPairEdgeEmbedding(
            lmax=3,
            edge_irreps=sh_irreps,
            multiplicity=1,
        ),
        lambda: RigidPairIrrepCompleteEdgeEmbedding(
            lmax=3,
        ),
    )

    for constructor in constructors:
        torch.manual_seed(1729)

        before = torch.random.get_rng_state().clone()

        module = constructor()

        after = torch.random.get_rng_state().clone()

        assert torch.equal(before, after)

        # The projection must still contain genuine trainable parameters.
        trainable = [
            parameter
            for parameter in module.parameters()
            if parameter.requires_grad
        ]

        assert trainable



def _activate_full_frame_compact(model):
    """Give the compact residual a deterministic nonzero test weight."""
    weight = model.rigid_pair_edge_embedding.projection.weight

    with torch.no_grad():
        values = torch.linspace(
            -0.15,
            0.15,
            weight.numel(),
            dtype=weight.dtype,
            device=weight.device,
        ).reshape_as(weight)

        weight.copy_(values)

    assert torch.count_nonzero(weight).item() > 0


def test_full_frame_compact_projection_gets_gradient_from_zero():
    model = _model("full_frame_compact")
    assert torch.count_nonzero(
        model.rigid_pair_edge_embedding.projection.weight
    ).item() == 0
    model.train()

    body_i = Rotation.identity()
    body_j = Rotation.from_rotvec(
        np.asarray([0.2, 0.4, -0.1])
    )

    output = model(
        _batch(_atoms(body_i, body_j)),
        compute_force=False,
    )

    loss = output["energy"].sum()
    loss.backward()

    parameters = list(
        model.rigid_pair_edge_embedding
        .projection.parameters()
    )

    assert parameters
    assert all(
        parameter.grad is not None
        for parameter in parameters
    )

    total_gradient = sum(
        parameter.grad.abs().sum()
        for parameter in parameters
    )

    assert total_gradient > 0.0


def test_full_frame_compact_detects_neighbor_rotation_after_activation():
    model = _model("full_frame_compact")
    _activate_full_frame_compact(model)
    model.eval()

    identity = Rotation.identity()

    axis = np.asarray([0.3, 1.0, -0.2], dtype=float)
    axis /= np.linalg.norm(axis)

    rotated_neighbor = Rotation.from_rotvec(
        axis * 0.83
    )

    out_a = model(
        _batch(_atoms(identity, identity)),
        compute_force=False,
    )

    out_b = model(
        _batch(_atoms(identity, rotated_neighbor)),
        compute_force=False,
    )

    delta = torch.max(
        torch.abs(
            out_a["energy"] - out_b["energy"]
        )
    )

    assert delta > 1.0e-10


def test_full_frame_compact_energy_is_globally_rotation_invariant():
    model = _model("full_frame_compact")
    _activate_full_frame_compact(model)
    model.eval()

    body_i = Rotation.from_rotvec(
        np.asarray([0.4, -0.2, 0.7])
    )

    body_j = Rotation.from_rotvec(
        np.asarray([-0.3, 0.6, 0.2])
    )

    global_rotation = Rotation.from_rotvec(
        np.asarray([0.2, 0.5, -0.8])
    )

    positions = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [2.1, 0.7, -0.4],
        ],
        dtype=float,
    )

    out_a = model(
        _batch(
            _atoms(
                body_i,
                body_j,
                positions=positions,
            )
        ),
        compute_force=False,
    )

    out_b = model(
        _batch(
            _atoms(
                global_rotation * body_i,
                global_rotation * body_j,
                positions=global_rotation.apply(positions),
            )
        ),
        compute_force=False,
    )

    torch.testing.assert_close(
        out_a["energy"],
        out_b["energy"],
        atol=ATOL,
        rtol=RTOL,
    )


def test_full_frame_compact_forces_rotate_covariantly():
    model = _model("full_frame_compact")
    _activate_full_frame_compact(model)
    model.eval()

    body_i = Rotation.from_rotvec(
        np.asarray([0.4, -0.2, 0.7])
    )

    body_j = Rotation.from_rotvec(
        np.asarray([-0.3, 0.6, 0.2])
    )

    global_rotation = Rotation.from_rotvec(
        np.asarray([0.2, 0.5, -0.8])
    )

    positions = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [2.1, 0.7, -0.4],
        ],
        dtype=float,
    )

    out_a = model(
        _batch(
            _atoms(
                body_i,
                body_j,
                positions=positions,
            )
        ),
        compute_force=True,
    )

    out_b = model(
        _batch(
            _atoms(
                global_rotation * body_i,
                global_rotation * body_j,
                positions=global_rotation.apply(positions),
            )
        ),
        compute_force=True,
    )

    S = torch.tensor(
        global_rotation.as_matrix(),
        dtype=DTYPE,
    )

    expected = out_a["forces"] @ S.T

    torch.testing.assert_close(
        out_b["forces"],
        expected,
        atol=ATOL,
        rtol=RTOL,
    )


def _d6_generator_rotations():
    c6 = Rotation.from_rotvec(
        np.asarray([0.0, 0.0, np.pi / 3.0])
    )
    c2 = Rotation.from_rotvec(
        np.asarray([np.pi, 0.0, 0.0])
    )
    return (c6, c2)


def test_d6_frame_energy_is_invariant_to_body_d6_generators():
    model = _model("d6_frame")
    model.eval()

    body_i = Rotation.from_rotvec(
        np.asarray([0.4, -0.2, 0.7])
    )
    body_j = Rotation.from_rotvec(
        np.asarray([-0.3, 0.6, 0.2])
    )

    reference = model(
        _batch(_atoms(body_i, body_j)),
        compute_force=False,
    )["energy"]

    for generator in _d6_generator_rotations():
        out_i = model(
            _batch(
                _atoms(
                    body_i * generator,
                    body_j,
                )
            ),
            compute_force=False,
        )["energy"]

        out_j = model(
            _batch(
                _atoms(
                    body_i,
                    body_j * generator,
                )
            ),
            compute_force=False,
        )["energy"]

        torch.testing.assert_close(
            out_i,
            reference,
            atol=ATOL,
            rtol=RTOL,
        )

        torch.testing.assert_close(
            out_j,
            reference,
            atol=ATOL,
            rtol=RTOL,
        )


def test_d6_frame_energy_is_globally_rotation_invariant():
    model = _model("d6_frame")
    model.eval()

    body_i = Rotation.from_rotvec(
        np.asarray([0.4, -0.2, 0.7])
    )
    body_j = Rotation.from_rotvec(
        np.asarray([-0.3, 0.6, 0.2])
    )
    global_rotation = Rotation.from_rotvec(
        np.asarray([0.2, 0.5, -0.8])
    )

    positions = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [2.1, 0.7, -0.4],
        ],
        dtype=float,
    )

    out_a = model(
        _batch(
            _atoms(
                body_i,
                body_j,
                positions=positions,
            )
        ),
        compute_force=False,
    )

    out_b = model(
        _batch(
            _atoms(
                global_rotation * body_i,
                global_rotation * body_j,
                positions=global_rotation.apply(
                    positions
                ),
            )
        ),
        compute_force=False,
    )

    torch.testing.assert_close(
        out_a["energy"],
        out_b["energy"],
        atol=ATOL,
        rtol=RTOL,
    )


def test_d6_frame_forces_rotate_covariantly():
    model = _model("d6_frame")
    model.eval()

    body_i = Rotation.from_rotvec(
        np.asarray([0.4, -0.2, 0.7])
    )
    body_j = Rotation.from_rotvec(
        np.asarray([-0.3, 0.6, 0.2])
    )
    global_rotation = Rotation.from_rotvec(
        np.asarray([0.2, 0.5, -0.8])
    )

    positions = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [2.1, 0.7, -0.4],
        ],
        dtype=float,
    )

    out_a = model(
        _batch(
            _atoms(
                body_i,
                body_j,
                positions=positions,
            )
        ),
        compute_force=True,
    )

    out_b = model(
        _batch(
            _atoms(
                global_rotation * body_i,
                global_rotation * body_j,
                positions=global_rotation.apply(
                    positions
                ),
            )
        ),
        compute_force=True,
    )

    S = torch.tensor(
        global_rotation.as_matrix(),
        dtype=DTYPE,
    )

    expected = out_a["forces"] @ S.T

    torch.testing.assert_close(
        out_b["forces"],
        expected,
        atol=ATOL,
        rtol=RTOL,
    )


def _activate_d6_frame_compact(model):
    weight = model.rigid_pair_edge_embedding.projection.weight

    with torch.no_grad():
        values = torch.linspace(
            -0.15,
            0.15,
            weight.numel(),
            dtype=weight.dtype,
            device=weight.device,
        ).reshape_as(weight)

        weight.copy_(values)

    assert torch.count_nonzero(weight).item() > 0


def test_d6_frame_compact_matches_none_at_zero():
    baseline = _model("none")
    compact = _model("d6_frame_compact")

    baseline.eval()
    compact.eval()

    body_i = Rotation.from_rotvec(
        np.asarray([0.4, -0.2, 0.7])
    )
    body_j = Rotation.from_rotvec(
        np.asarray([-0.3, 0.6, 0.2])
    )

    out_baseline = baseline(
        _batch(_atoms(body_i, body_j)),
        compute_force=True,
    )

    out_compact = compact(
        _batch(_atoms(body_i, body_j)),
        compute_force=True,
    )

    torch.testing.assert_close(
        out_compact["energy"],
        out_baseline["energy"],
        atol=1.0e-12,
        rtol=1.0e-12,
    )

    torch.testing.assert_close(
        out_compact["forces"],
        out_baseline["forces"],
        atol=1.0e-12,
        rtol=1.0e-12,
    )


def test_d6_frame_compact_projection_gets_gradient_from_zero():
    model = _model("d6_frame_compact")

    weight = model.rigid_pair_edge_embedding.projection.weight

    assert torch.count_nonzero(weight).item() == 0

    model.train()

    body_i = Rotation.identity()
    body_j = Rotation.from_rotvec(
        np.asarray([0.2, 0.4, -0.1])
    )

    output = model(
        _batch(_atoms(body_i, body_j)),
        compute_force=False,
    )

    output["energy"].sum().backward()

    assert weight.grad is not None
    assert weight.grad.abs().sum() > 0.0


def test_d6_frame_compact_is_invariant_to_body_d6_generators():
    model = _model("d6_frame_compact")
    _activate_d6_frame_compact(model)
    model.eval()

    body_i = Rotation.from_rotvec(
        np.asarray([0.4, -0.2, 0.7])
    )
    body_j = Rotation.from_rotvec(
        np.asarray([-0.3, 0.6, 0.2])
    )

    reference = model(
        _batch(_atoms(body_i, body_j)),
        compute_force=False,
    )["energy"]

    for generator in _d6_generator_rotations():
        out_i = model(
            _batch(
                _atoms(
                    body_i * generator,
                    body_j,
                )
            ),
            compute_force=False,
        )["energy"]

        out_j = model(
            _batch(
                _atoms(
                    body_i,
                    body_j * generator,
                )
            ),
            compute_force=False,
        )["energy"]

        torch.testing.assert_close(
            out_i,
            reference,
            atol=ATOL,
            rtol=RTOL,
        )

        torch.testing.assert_close(
            out_j,
            reference,
            atol=ATOL,
            rtol=RTOL,
        )


def test_d6_frame_compact_detects_non_d6_axial_rotation():
    model = _model("d6_frame_compact")
    _activate_d6_frame_compact(model)
    model.eval()

    body_i = Rotation.from_rotvec(
        np.asarray([0.4, -0.2, 0.7])
    )
    body_j = Rotation.from_rotvec(
        np.asarray([-0.3, 0.6, 0.2])
    )

    half_step = Rotation.from_rotvec(
        np.asarray([0.0, 0.0, np.pi / 6.0])
    )

    out_a = model(
        _batch(_atoms(body_i, body_j)),
        compute_force=False,
    )

    out_b = model(
        _batch(
            _atoms(
                body_i,
                body_j * half_step,
            )
        ),
        compute_force=False,
    )

    delta = torch.max(
        torch.abs(
            out_a["energy"] - out_b["energy"]
        )
    )

    assert delta > 1.0e-10


def test_d6_frame_compact_energy_is_globally_rotation_invariant():
    model = _model("d6_frame_compact")
    _activate_d6_frame_compact(model)
    model.eval()

    body_i = Rotation.from_rotvec(
        np.asarray([0.4, -0.2, 0.7])
    )
    body_j = Rotation.from_rotvec(
        np.asarray([-0.3, 0.6, 0.2])
    )
    global_rotation = Rotation.from_rotvec(
        np.asarray([0.2, 0.5, -0.8])
    )

    positions = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [2.1, 0.7, -0.4],
        ],
        dtype=float,
    )

    out_a = model(
        _batch(
            _atoms(
                body_i,
                body_j,
                positions=positions,
            )
        ),
        compute_force=False,
    )

    out_b = model(
        _batch(
            _atoms(
                global_rotation * body_i,
                global_rotation * body_j,
                positions=global_rotation.apply(
                    positions
                ),
            )
        ),
        compute_force=False,
    )

    torch.testing.assert_close(
        out_a["energy"],
        out_b["energy"],
        atol=ATOL,
        rtol=RTOL,
    )


def test_d6_frame_compact_forces_rotate_covariantly():
    model = _model("d6_frame_compact")
    _activate_d6_frame_compact(model)
    model.eval()

    body_i = Rotation.from_rotvec(
        np.asarray([0.4, -0.2, 0.7])
    )
    body_j = Rotation.from_rotvec(
        np.asarray([-0.3, 0.6, 0.2])
    )
    global_rotation = Rotation.from_rotvec(
        np.asarray([0.2, 0.5, -0.8])
    )

    positions = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [2.1, 0.7, -0.4],
        ],
        dtype=float,
    )

    out_a = model(
        _batch(
            _atoms(
                body_i,
                body_j,
                positions=positions,
            )
        ),
        compute_force=True,
    )

    out_b = model(
        _batch(
            _atoms(
                global_rotation * body_i,
                global_rotation * body_j,
                positions=global_rotation.apply(
                    positions
                ),
            )
        ),
        compute_force=True,
    )

    S = torch.tensor(
        global_rotation.as_matrix(),
        dtype=DTYPE,
    )

    expected = out_a["forces"] @ S.T

    torch.testing.assert_close(
        out_b["forces"],
        expected,
        atol=ATOL,
        rtol=RTOL,
    )
