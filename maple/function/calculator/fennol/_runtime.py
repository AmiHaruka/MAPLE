from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
import functools
from functools import lru_cache
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import serialization
from ase.data import chemical_symbols
from ase.neighborlist import neighbor_list

ANG = 0.52917721
HA_TO_EV = 27.211386024367243


# === FeNNol runtime tables ===================================================

_ORBITAL_CAPACITIES: tuple[tuple[int, int], ...] = (
    (0, 2),   # 1s
    (0, 2),   # 2s
    (1, 6),   # 2p
    (0, 2),   # 3s
    (1, 6),   # 3p
    (0, 2),   # 4s
    (2, 10),  # 3d
    (1, 6),   # 4p
    (0, 2),   # 5s
    (2, 10),  # 4d
    (1, 6),   # 5p
    (0, 2),   # 6s
    (3, 14),  # 4f
    (2, 10),  # 5d
    (1, 6),   # 6p
    (0, 2),   # 7s
    (3, 14),  # 5f
    (2, 10),  # 6d
    (1, 6),   # 7p
)

_D3_COV_RADII = np.asarray(
    [
        1.8897261278504418,
        0.6047123609121414,
        0.8692740188112033,
        2.5133357500410876,
        1.9275206504074507,
        1.6062672086728755,
        1.4172945958878314,
        1.3417055507738136,
        1.1905274605457783,
        1.2094247218242828,
        1.266116505659796,
        2.929075498168185,
        2.626719317712114,
        2.3810549210915566,
        2.1920823083065124,
        2.0975960019139905,
        1.946417911685955,
        1.8708288665719373,
        1.814137082736424,
        3.703863210586866,
        3.2314316786242556,
        2.7967946692186536,
        2.570027533876601,
        2.532233011319592,
        2.305465875977539,
        2.2487740921420256,
        2.1920823083065124,
        2.0975960019139905,
        2.078698740635486,
        2.116493263192495,
        2.229876830863521,
        2.343260398534548,
        2.2865686146990347,
        2.2865686146990347,
        2.1920823083065124,
        2.1542877857495033,
        2.210979569585017,
        3.968424868485928,
        3.4959933365233176,
        3.08025358839622,
        2.9101782368896805,
        2.7778974079401495,
        2.6078220564336094,
        2.4188494436485657,
        2.362157659813052,
        2.362157659813052,
        2.26767135342053,
        2.4188494436485657,
        2.570027533876601,
        2.683411101547627,
        2.6456165789906185,
        2.6456165789906185,
        2.570027533876601,
        2.5133357500410876,
        2.475541227484079,
        4.384164616613025,
        3.703863210586866,
        3.4015070301307953,
        3.08025358839622,
        3.3259179850167775,
        3.288123462459769,
        3.2692262011812643,
        3.2503289399027597,
        3.174739894788742,
        3.1936371560672465,
        3.174739894788742,
        3.155842633510238,
        3.1369453722317333,
        3.1180481109532288,
        3.099150849674724,
        3.212534417345751,
        3.061356327117716,
        2.8723837143326714,
        2.759000146661645,
        2.5889247951551053,
        2.475541227484079,
        2.43774670492707,
        2.305465875977539,
        2.3243631372560434,
        2.343260398534548,
        2.5133357500410876,
        2.7212056241046363,
        2.7212056241046363,
        2.8534864530541673,
        2.7401028853831404,
        2.7778974079401495,
        2.683411101547627,
        1.8897261278504418,
    ],
    dtype=float,
)

_HALOGENS = frozenset({9, 17, 35, 53, 85, 117})
_ALKALI = frozenset({3, 11, 19, 37, 55, 87})
_ALKALINE = frozenset({4, 12, 20, 38, 56, 88})
_TRANSITION_METALS = frozenset(range(21, 31)) | frozenset(range(39, 49)) | frozenset(range(72, 81)) | frozenset(range(104, 113))
_POST_METALS = frozenset({13, 31, 49, 50, 81, 82, 83, 84, 113, 114, 115, 116})
_METALLOIDS = frozenset({5, 14, 32, 33, 51, 52})
_LANTHANIDES = frozenset(range(57, 72))
_ACTINIDES = frozenset(range(89, 104))
_NOBLE_GASES = frozenset({2, 10, 18, 36, 54, 86, 118})
_BIO_NONMETALS = frozenset({6, 7, 8, 15, 16, 34})


def _element_family_name(atomic_number: int) -> str | None:
    z = int(atomic_number)
    if z == 1:
        return "H"
    if z in _BIO_NONMETALS:
        return "CNOPSSe"
    if z in _HALOGENS:
        return "HALOGENS"
    if z in _ALKALI:
        return "ALKALI"
    if z in _ALKALINE:
        return "ALKALINE"
    if z in _TRANSITION_METALS:
        return "TRANSITION_METALS"
    if z in _POST_METALS:
        return "POST_METALS"
    if z in _METALLOIDS:
        return "METALLOIDS"
    if z in _LANTHANIDES:
        return "LANTHANIDES"
    if z in _ACTINIDES:
        return "ACTINIDES"
    if z in _NOBLE_GASES:
        return "NOBLE GASES"
    return None


@lru_cache(maxsize=1)
def _electronic_structure_table() -> np.ndarray:
    table = np.zeros((119, len(_ORBITAL_CAPACITIES)), dtype=float)
    for atomic_number in range(1, table.shape[0]):
        remaining = atomic_number
        for column, (_, capacity) in enumerate(_ORBITAL_CAPACITIES):
            filled = min(remaining, capacity)
            table[atomic_number, column] = filled
            remaining -= filled
            if remaining <= 0:
                break
    return table


@lru_cache(maxsize=1)
def _valence_structure_table() -> np.ndarray:
    table = np.zeros((119, 4), dtype=float)
    shell_order = [
        1, 2, 2, 3, 3, 4, 3, 4, 5, 4, 5, 6, 4, 5, 6, 7, 5, 6, 7
    ]
    for atomic_number in range(1, table.shape[0]):
        occupancies = _electronic_structure_table()[atomic_number]
        occupied_shells = [
            shell for shell, occ in zip(shell_order, occupancies) if occ > 0
        ]
        if not occupied_shells:
            continue
        valence_shell = max(occupied_shells)
        for shell, (angular_momentum, _), occ in zip(
            shell_order, _ORBITAL_CAPACITIES, occupancies
        ):
            if shell == valence_shell:
                table[atomic_number, angular_momentum] += occ
    return table


def _d3_cov_radii() -> np.ndarray:
    return _D3_COV_RADII


@lru_cache(maxsize=1)
def _nlh_coeffs() -> np.ndarray:
    path = Path(__file__).with_name("nlh_coeffs.dat")
    return np.loadtxt(path, usecols=np.arange(0, 8), dtype=np.float32)


def _as_dict(items: Any) -> dict[str, Any]:
    return dict(items) if isinstance(items, list) else dict(items)


def _jaxify_variables(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return jnp.asarray(value)
    if isinstance(value, dict):
        return {key: _jaxify_variables(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jaxify_variables(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_jaxify_variables(item) for item in value)
    return value


@dataclass(frozen=True)
class FennolStep:
    name: str
    fid: str
    params: dict[str, Any]


@dataclass(frozen=True)
class FennolProgram:
    model_path: Path
    cutoff: float | None
    energy_unit: str
    energy_terms: tuple[str, ...]
    preprocessing: tuple[FennolStep, ...]
    modules: tuple[FennolStep, ...]
    variables: dict[str, Any]


@dataclass(frozen=True)
class FENNIXModelInfo:
    model_path: Path
    energy_unit: str
    energy_terms: tuple[str, ...]
    preprocessing_fpids: tuple[str, ...]
    module_fids: tuple[str, ...]


def load_fennol_program(model_path: str | Path) -> FennolProgram:
    path = Path(model_path)
    state = serialization.msgpack_restore(path.read_bytes())
    preprocessing = OrderedDict(_as_dict(state.get("preprocessing", ())))
    modules = OrderedDict(_as_dict(state.get("modules", ())))
    return FennolProgram(
        model_path=path,
        cutoff=state.get("cutoff"),
        energy_unit=str(state.get("energy_unit", "hartree")).strip().lower() or "hartree",
        energy_terms=tuple(state.get("energy_terms", ())),
        preprocessing=tuple(
            FennolStep(name, _registry_key(name, dict(params), "FPID"), dict(params))
            for name, params in preprocessing.items()
        ),
        modules=tuple(
            FennolStep(name, _registry_key(name, dict(params), "FID"), dict(params))
            for name, params in modules.items()
        ),
        variables=state["variables"],
    )


def inspect_fennol_model(model_path: str | Path) -> FENNIXModelInfo:
    program = load_fennol_program(model_path)
    return FENNIXModelInfo(
        model_path=program.model_path,
        energy_unit=program.energy_unit,
        energy_terms=program.energy_terms,
        preprocessing_fpids=tuple(step.fid for step in program.preprocessing),
        module_fids=tuple(step.fid for step in program.modules),
    )


def _swish(x):
    return jax.nn.swish(x)


def _activation(name: str):
    name = str(name).strip().lower()
    if name in {"swish", "silu"}:
        return _swish
    if name == "identity":
        return lambda x: x
    raise NotImplementedError(f"FeNNol activation {name!r} is not supported.")


def _layer_norm(x):
    mu = jnp.mean(x, axis=-1, keepdims=True)
    dx = x - mu
    var = jnp.mean(dx**2, axis=-1, keepdims=True)
    return dx * (1.0e-6 + var) ** (-0.5)


def _safe_sqrt(x):
    return jnp.sqrt(jnp.clip(x, min=1.0e-5))


def _dense(x, features: int, *, name: str, use_bias: bool = True):
    return nn.Dense(features, use_bias=use_bias, name=name)(x)


class FullyConnectedNet(nn.Module):
    neurons: tuple[int, ...]
    activation: str = "swish"
    use_bias: bool = True
    input_key: str | None = None
    output_key: str | None = None
    squeeze: bool = False

    @nn.compact
    def __call__(self, inputs):
        dict_input = isinstance(inputs, dict)
        if dict_input:
            if self.input_key is None:
                raise ValueError(f"{self.name} requires input_key for dict input.")
            x = inputs[self.input_key]
        else:
            x = inputs
        act = _activation(self.activation)
        for i, dim in enumerate(self.neurons[:-1]):
            x = act(_dense(x, dim, name=f"Layer_{i + 1}", use_bias=self.use_bias))
        x = _dense(
            x,
            self.neurons[-1],
            name=f"Layer_{len(self.neurons)}",
            use_bias=self.use_bias,
        )
        if self.squeeze and x.shape[-1] == 1:
            x = jnp.squeeze(x, axis=-1)
        if dict_input:
            output_key = self.output_key if self.output_key is not None else self.name
            return {**inputs, output_key: x}
        return x


class SpeciesEncoding(nn.Module):
    encoding: str = "electronic_structure"
    zmax: int = 86
    output_key: str | None = None
    trainable: bool = False

    @nn.compact
    def __call__(self, inputs):
        species = inputs["species"] if isinstance(inputs, dict) else inputs
        dtype = inputs["coordinates"].dtype if isinstance(inputs, dict) else jnp.float32
        encoding = str(self.encoding).strip().lower()
        if encoding not in {"electronic_structure", "electronic-structure"}:
            raise NotImplementedError(f"FeNNol species encoding {self.encoding!r} is not supported.")
        zmax = self.zmax
        z = np.arange(1, zmax + 1).reshape(-1, 1)
        zref = [zmax]
        eref = [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6]
        vref = [2, 6, 10, 14]
        table = np.concatenate(
            [
                z,
                _electronic_structure_table()[1 : zmax + 1, :15],
                _valence_structure_table()[1 : zmax + 1],
            ],
            axis=1,
        )
        table = table / np.asarray(zref + eref + vref)[None, :]
        table = np.concatenate(
            [np.zeros((1, table.shape[1])), table, np.zeros((1, table.shape[1]))],
            axis=0,
        )
        output = jnp.asarray(table, dtype=dtype)[species]
        if isinstance(inputs, dict):
            output_key = self.output_key if self.output_key is not None else self.name
            return {**inputs, output_key: output}
        return output


class ChargeHypothesis(nn.Module):
    embedding_key: str = "embedding"
    output_key: str | None = None
    total_charge_key: str = "total_charge"
    ncharges: int = 32
    mode: str = "qeq"
    squeeze: bool = True

    @nn.compact
    def __call__(self, inputs):
        if str(self.mode).strip().lower() != "qeq":
            raise NotImplementedError(f"FeNNol charge mode {self.mode!r} is not supported.")
        embedding = inputs[self.embedding_key]
        batch_index = inputs["batch_index"]
        natoms = inputs["natoms"]
        total_charge = inputs[self.total_charge_key]
        wi = jax.nn.softplus(_dense(embedding, self.ncharges, name="wi"))
        wtot = jax.ops.segment_sum(wi, batch_index, natoms.shape[0])
        qtilde = _dense(embedding, self.ncharges, name="qi")
        qtot = jax.ops.segment_sum(qtilde, batch_index, natoms.shape[0])
        qref = jnp.asarray(total_charge, dtype=wi.dtype)
        if qref.ndim == 0:
            qref = qref * jnp.ones(natoms.shape[0], dtype=wi.dtype)
        dq = qref[:, None] - qtot
        output = qtilde + wi * (dq / wtot)[batch_index]
        if self.squeeze and output.shape[-1] == 1:
            output = jnp.squeeze(output, axis=-1)
        output_key = self.output_key if self.output_key is not None else self.name
        return {**inputs, output_key: output}


class SwitchFunction(nn.Module):
    cutoff: float
    switch_type: str = "hard"
    p: float | None = None
    trainable: bool = False

    @nn.compact
    def __call__(self, distances, edge_mask):
        kind = self.switch_type.lower()
        if kind == "hard":
            switch = jnp.where(distances < self.cutoff, 1.0, 0.0)
        elif kind == "polynomial":
            p = 3.0 if self.p is None else float(self.p)
            if self.trainable:
                p = self.param("p", lambda rng: jnp.asarray(p, dtype=jnp.float32))
            d = distances / self.cutoff
            switch = (
                1.0
                - 0.5 * (p + 1) * (p + 2) * d**p
                + p * (p + 2) * d ** (p + 1)
                - 0.5 * p * (p + 1) * d ** (p + 2)
            )
        else:
            raise NotImplementedError(f"FeNNol switch {kind!r} is not supported.")
        return jnp.where(edge_mask, switch, 0.0)


class GraphFilterProcessor(nn.Module):
    cutoff: float = 3.5
    graph_key: str = "graph_embed"
    parent_graph: str = "graph"
    output_key: str | None = None
    switch_params: dict[str, Any] | None = None
    remove_hydrogens: bool = False

    @nn.compact
    def __call__(self, inputs):
        if not isinstance(inputs, dict):
            graph = inputs
            parent_graph = None
        else:
            graph = inputs[self.graph_key]
            parent_graph = inputs[self.parent_graph]
        switch_params = dict(self.switch_params or {})
        switch_type = switch_params.get("switch_type", "polynomial")
        p = switch_params.get("p", 8.0)
        trainable = bool(switch_params.get("trainable", True))
        if parent_graph is not None and "filter_indices" in graph:
            filter_indices = graph["filter_indices"]
            vec = parent_graph["vec"].at[filter_indices].get(
                mode="fill",
                fill_value=self.cutoff,
            )
            distances = parent_graph["distances"].at[filter_indices].get(
                mode="fill",
                fill_value=self.cutoff,
            )
        else:
            vec = graph["vec"]
            distances = graph["distances"]
        edge_mask = distances < self.cutoff
        switch = SwitchFunction(
            cutoff=self.cutoff,
            switch_type=switch_type,
            p=p,
            trainable=trainable,
            name="SwitchFunction_0",
        )(distances, edge_mask)
        graph = {
            **graph,
            "vec": vec,
            "distances": distances,
            "d12": distances * distances,
            "edge_mask": edge_mask,
            "switch": switch,
        }
        if isinstance(inputs, dict):
            output_key = self.output_key if self.output_key is not None else self.graph_key
            return {**inputs, output_key: graph}
        return graph


def _spherical_harmonics_l3(vec):
    x, y, z = [jax.lax.index_in_dim(vec, i, axis=-1, keepdims=False) for i in range(3)]
    sh_0_0 = jnp.ones_like(x)
    sh_1_0 = 1.73205080756888 * x
    sh_1_1 = 1.73205080756888 * y
    sh_1_2 = 1.73205080756888 * z
    sh_2_0 = 1.11803398874989 * sh_1_0 * z + 1.11803398874989 * sh_1_2 * x
    sh_2_1 = 1.11803398874989 * sh_1_0 * y + 1.11803398874989 * sh_1_1 * x
    sh_2_2 = -0.645497224367903 * sh_1_0 * x + 1.29099444873581 * sh_1_1 * y - 0.645497224367903 * sh_1_2 * z
    sh_2_3 = 1.11803398874989 * sh_1_1 * z + 1.11803398874989 * sh_1_2 * y
    sh_2_4 = -1.11803398874989 * sh_1_0 * x + 1.11803398874989 * sh_1_2 * z
    sh_3_0 = 1.08012344973464 * sh_2_0 * z + 1.08012344973464 * sh_2_4 * x
    sh_3_1 = 0.881917103688197 * sh_2_0 * y + 0.881917103688197 * sh_2_1 * z + 0.881917103688197 * sh_2_3 * x
    sh_3_2 = -0.278886675511359 * sh_2_0 * z + 1.11554670204543 * sh_2_1 * y + 0.966091783079296 * sh_2_2 * x + 0.278886675511359 * sh_2_4 * x
    sh_3_3 = -0.683130051063973 * sh_2_1 * x + 1.18321595661992 * sh_2_2 * y - 0.683130051063973 * sh_2_3 * z
    sh_3_4 = -0.278886675511359 * sh_2_0 * x + 0.966091783079296 * sh_2_2 * z + 1.11554670204543 * sh_2_3 * y - 0.278886675511359 * sh_2_4 * z
    sh_3_5 = -0.881917103688197 * sh_2_1 * x + 0.881917103688197 * sh_2_3 * z + 0.881917103688197 * sh_2_4 * y
    sh_3_6 = -1.08012344973464 * sh_2_0 * x + 1.08012344973464 * sh_2_4 * z
    return jnp.stack(
        [
            sh_0_0,
            sh_1_0,
            sh_1_1,
            sh_1_2,
            sh_2_0,
            sh_2_1,
            sh_2_2,
            sh_2_3,
            sh_2_4,
            sh_3_0,
            sh_3_1,
            sh_3_2,
            sh_3_3,
            sh_3_4,
            sh_3_5,
            sh_3_6,
        ],
        axis=-1,
    )


def _radial_bessel(distances, dim: int, cutoff: float):
    x = distances[:, None]
    roots = jnp.asarray(np.arange(1, dim + 1)[None, :] * (np.pi / cutoff), dtype=distances.dtype)
    norm = 1.0 / (dim * np.pi / cutoff)
    return norm * jnp.sin(x * roots) / x


def _radial_basis(distances, params: dict[str, Any], cutoff: float):
    basis = str(params.get("basis", "bessel")).strip().lower()
    if basis != "bessel":
        raise NotImplementedError(f"FeNNol radial basis {basis!r} is not supported.")
    dim = int(params.get("dim", 8))
    start = float(params.get("start", 0.0))
    end = float(params.get("end", cutoff))
    c = end - start
    x = distances[:, None] - start
    roots = jnp.asarray(np.arange(1, dim + 1)[None, :] * (np.pi / c), dtype=distances.dtype)
    if bool(params.get("alt_bessel_norm", False)):
        norm = (2.0 / c) ** 0.5
    else:
        norm = 1.0 / (dim * np.pi / c)
    return norm * jnp.sin(x * roots) / x


@lru_cache(maxsize=1)
def _require_e3nn_jax():
    try:
        import e3nn_jax as e3nn
    except Exception as exc:
        raise ImportError("FeNNol MACE modules require the optional dependency e3nn_jax.") from exc
    return e3nn


class ChannelMixing(nn.Module):
    lmax: int
    nchannels: int

    @nn.compact
    def __call__(self, x):
        weights = self.param(
            "weights",
            jax.nn.initializers.normal(stddev=1.0 / self.nchannels**0.5),
            (self.nchannels, self.nchannels),
        )
        return jnp.einsum("ij,...jk->...ik", weights, x)


class BlockIndexNet(nn.Module):
    output_dim: int
    hidden_neurons: tuple[int, ...]
    activation: str = "swish"
    use_bias: bool = True
    input_key: str | None = None
    output_key: str | None = None
    block_index_key: str = "block_index"
    squeeze: bool = False

    @nn.compact
    def __call__(self, inputs, embedding=None, block_index=None):
        dict_input = isinstance(inputs, dict)
        if dict_input:
            species = inputs["species"]
            if self.input_key is None:
                raise ValueError(f"{self.name} requires input_key for dict input.")
            embedding = inputs[self.input_key]
            block_index = inputs[self.block_index_key]
        else:
            species = inputs
        outputs = []
        indices = []
        for name, idx in block_index.items():
            if idx is None or idx.size == 0:
                continue
            net = FullyConnectedNet(
                (*self.hidden_neurons, self.output_dim),
                activation=self.activation,
                use_bias=self.use_bias,
                name=name,
            )
            outputs.append(net(embedding[idx]))
            indices.append(idx)
        if not outputs:
            out = jnp.zeros((species.shape[0], self.output_dim), dtype=embedding.dtype)
        else:
            values = jnp.concatenate(outputs, axis=0)
            idx = jnp.concatenate(indices, axis=0)
            out = jnp.zeros((species.shape[0], *values.shape[1:]), dtype=values.dtype).at[idx].set(values, mode="drop")
        if self.squeeze and out.shape[-1] == 1:
            out = jnp.squeeze(out, axis=-1)
        if dict_input:
            output_key = self.output_key if self.output_key is not None else self.name
            return {**inputs, output_key: out}
        return out


class RaSTER(nn.Module):
    dim: int = 176
    scal_heads: int = 8
    tens_heads: int = 1
    lode_channels: int = 1
    graph_key: str = "graph_embed"
    graph_lode: str = "graph"
    block_index_key: str = "block_index"
    embedding_key: str = "embedding"
    species_encoding: str = "species_embedding"
    lmax: int = 3
    lmax_lode: int = 2
    lode_extra_powers: tuple[int, ...] = (6, 8, 10)
    lode_rshort: float = 1.75
    lode_dshort: float = 1.75
    radial_basis: dict[str, Any] | None = None
    update_hidden: tuple[int, ...] = (352, 176)
    activation: str = "swish"
    att_activation: str = "swish"
    att_dim: int = 16
    nlayers: int = 2
    ignore_parity: bool = False

    @nn.compact
    def __call__(self, inputs):
        species = inputs["species"]
        graph = inputs[self.graph_key]
        graph_lode = inputs[self.graph_lode]
        block_index = inputs[self.block_index_key]
        zi = inputs[self.species_encoding]
        xi = _layer_norm(_dense(zi, self.dim, name="species_linear", use_bias=False))

        distances = graph["distances"]
        switch = graph["switch"]
        edge_src = graph["edge_src"]
        edge_dst = graph["edge_dst"]
        vec = graph["vec"] / graph["distances"][:, None]

        rc = jnp.asarray(_d3_cov_radii(), dtype=distances.dtype)[species] * ANG
        rcij = rc[edge_src] + rc[edge_dst]
        rstart = rcij * 0.5
        rend = rcij * 0.6
        switch_short = (distances >= rend) + 0.5 * (1 - jnp.cos(jnp.pi * (distances - rstart) / (rend - rstart))) * (distances > rstart) * (distances < rend)
        switch = switch * switch_short

        yij = _spherical_harmonics_l3(vec)[:, None, :]
        nrep = np.array([1, 3, 5, 7])
        ls = np.arange(4).repeat(nrep)
        parity = jnp.array((-1) ** ls[None, None, :])
        vi = 0.0
        radial_dim = int((self.radial_basis or {}).get("dim", 10))
        radial_terms = _radial_bessel(distances, radial_dim, graph["cutoff"]) * switch_short[:, None]

        graph_lr = graph_lode
        edge_src_lr, edge_dst_lr = graph_lr["edge_src"], graph_lr["edge_dst"]
        r = graph_lr["distances"][:, None]
        switch_lode = graph_lr["switch"][:, None]
        rc_lode = graph_lode["cutoff"]
        lmax_lr = self.lmax_lode
        nrep_lr = np.array([2 * l + 1 for l in range(lmax_lr + 1)], dtype=np.int32)
        ls_lr_base = np.arange(lmax_lr + 1)
        extra = np.asarray(self.lode_extra_powers, dtype=np.int32)
        ls_lr = np.concatenate([extra, ls_lr_base])
        a = self.param("a_lr", lambda rng: jnp.asarray([[1.0] * len(ls_lr)], dtype=jnp.float32)) ** 2
        rc2a = rc_lode**2 + a
        ls_exp = 0.5 * (ls_lr[None, :] + 1)
        eij_lr = (
            1.0 / (r**2 + a) ** ls_exp
            - 1.0 / rc2a**ls_exp
            + (r - rc_lode) * rc_lode * (2 * ls_exp) / rc2a ** (ls_exp + 1)
        ) * switch_lode
        rs = self.lode_rshort
        dshort = self.lode_dshort
        short = 0.5 * (1 - jnp.cos(jnp.pi * (r - rs) / dshort)) * (r > rs) * (r < rs + dshort) + (r >= rs + dshort)
        eij_lr = eij_lr * short
        eij_lr_extra = eij_lr[:, :3]
        eij_lr = eij_lr[:, 3:]
        eij_lr = eij_lr.repeat(nrep_lr, axis=-1)
        yij_lr = _spherical_harmonics_l3(graph_lr["vec"] / r)[:, :9]
        eij_lr = eij_lr * yij_lr

        if self.tens_heads > 1:
            vi = jnp.zeros((zi.shape[0], self.tens_heads, yij.shape[-1]), dtype=xi.dtype)

        for layer in range(self.nlayers):
            parts = [radial_terms]
            if layer > 0:
                xij2 = (vi[edge_dst] + (parity * vi)[edge_src]) * yij
                for l in range(4):
                    parts.append((xij2[:, :, l**2 : (l + 1) ** 2]).sum(axis=-1))
            ur = jnp.concatenate(parts, axis=-1)

            nout = 1
            w = FullyConnectedNet(
                (2 * 16, nout * 16),
                activation="swish",
                use_bias=True,
                name=f"positional_encoding_{layer}",
            )(ur).reshape(radial_terms.shape[0], nout, 16)
            nls = 4 if layer == 0 else 8

            q = _dense(xi, (self.scal_heads + nls * self.tens_heads) * 16, name=f"queries_{layer}", use_bias=False).reshape(
                xi.shape[0], self.scal_heads + nls * self.tens_heads, 16
            )
            k = _dense(xi, (self.scal_heads + nls * self.tens_heads) * 16, name=f"keys_{layer}", use_bias=False).reshape(
                xi.shape[0], self.scal_heads + nls * self.tens_heads, 16
            )
            v = _dense(xi, self.scal_heads * 16, name=f"values_{layer}", use_bias=False).reshape(
                xi.shape[0], self.scal_heads, 16
            )

            wk = w * k[edge_dst]
            aij = (_swish((q[edge_src] * wk).sum(axis=-1) / (16**0.5)) * switch[:, None])
            aijl = aij[:, : self.tens_heads * 4].reshape(-1, self.tens_heads, 4).repeat(nrep, axis=-1)
            if layer > 0:
                aijl1 = aij[:, self.tens_heads * 4 : self.tens_heads * nls].reshape(-1, self.tens_heads, 4).repeat(nrep, axis=-1)
            aij_scalar = aij[:, self.tens_heads * nls :, None]
            vij = v[edge_dst]
            vai = jax.ops.segment_sum(aij_scalar * vij, edge_src, num_segments=xi.shape[0]).reshape(xi.shape[0], -1)

            uij = aijl * yij
            if layer > 0:
                uij = uij + aijl1 * vi[edge_dst]
            vi = vi + jax.ops.segment_sum(uij, edge_src, num_segments=zi.shape[0])

            si = _dense(xi, 16, name=f"self_values_{layer}", use_bias=False)
            components = [si, vai]
            if self.tens_heads == 1:
                vi2 = vi**2
            else:
                vi2 = vi * ChannelMixing(3, self.tens_heads, name=f"extract_mixing_{layer}")(vi)
            for l in range(4):
                components.append((vi2[:, :, l**2 : (l + 1) ** 2]).sum(axis=-1) / (2 * l + 1))

            if layer == 1:
                zj = _dense(xi, self.lode_channels * 6, name=f"lode_values_{layer}", use_bias=False).reshape(
                    xi.shape[0], self.lode_channels, 6
                )
                zj_extra = zj[:, :, :3]
                zj_lr = zj[:, :, 3:]
                xi_lr_extra = jax.ops.segment_sum(eij_lr_extra[:, None, :] * zj_extra[edge_dst_lr], edge_src_lr, species.shape[0]).reshape(species.shape[0], -1)
                components.append(xi_lr_extra)
                zj_lr = zj_lr.repeat(nrep_lr, axis=-1)
                vi_lr = jax.ops.segment_sum(eij_lr[:, None, :] * zj_lr[edge_dst_lr], edge_src_lr, species.shape[0])
                components.append(vi_lr[:, :, 0])
                mi_lr = vi[:, : self.lode_channels, :9] * vi_lr
                for l in range(1, 3):
                    components.append((mi_lr[:, :, l**2 : (l + 1) ** 2]).sum(axis=-1) / (2 * l + 1))

            components = jnp.concatenate(components, axis=-1)
            updi = BlockIndexNet(
                output_dim=self.dim + self.tens_heads * 4,
                hidden_neurons=tuple(self.update_hidden),
                activation=self.activation,
                use_bias=True,
                name=f"update_net_{layer}",
            )(species, components, block_index)

            xi = _layer_norm(xi + updi[:, : self.dim])
            vi = vi * (1 + updi[:, self.dim :]).reshape(-1, self.tens_heads, 4).repeat(nrep, axis=-1)
            if self.tens_heads > 1:
                vi = ChannelMixing(3, self.tens_heads, name=f"update_mixing_{layer}")(vi)

        return {**inputs, self.embedding_key: xi}


class SymmetricContraction(nn.Module):
    correlation: int
    keep_irrep_out: Any
    num_species: int
    gradient_normalization: str | float
    symmetric_tensor_product_basis: bool

    @nn.compact
    def __call__(self, input, index):
        e3nn = _require_e3nn_jax()
        gradient_normalization = self.gradient_normalization
        if gradient_normalization is None:
            gradient_normalization = e3nn.config("gradient_normalization")
        if isinstance(gradient_normalization, str):
            gradient_normalization = {"element": 0.0, "path": 1.0}[gradient_normalization]

        keep_irrep_out = self.keep_irrep_out
        if isinstance(keep_irrep_out, str):
            keep_irrep_out = e3nn.Irreps(keep_irrep_out)
        keep_irrep_out = {e3nn.Irrep(ir) for ir in keep_irrep_out}

        input = input.mul_to_axis().remove_nones()
        weights_by_order = []
        bases_by_order = []
        for order in range(1, self.correlation + 1):
            if self.symmetric_tensor_product_basis:
                basis = e3nn.reduced_symmetric_tensor_product_basis(
                    input.irreps,
                    order,
                    keep_ir=keep_irrep_out,
                )
            else:
                basis = e3nn.reduced_tensor_product_basis(
                    [input.irreps] * order,
                    keep_ir=keep_irrep_out,
                )
            bases_by_order.append(basis)
            order_weights = []
            for (mul, ir_out), _ in zip(basis.irreps, basis.list):
                w = self.param(
                    f"w{order}_{ir_out}",
                    nn.initializers.normal(
                        stddev=(mul**-0.5) ** (1.0 - gradient_normalization)
                    ),
                    (self.num_species, mul, input.shape[-2]),
                )
                order_weights.append(w * (mul**-0.5) ** gradient_normalization)
            weights_by_order.append(order_weights)

        def contract_one(value, species_index):
            out = {}
            x = value.array
            for order in range(self.correlation, 0, -1):
                basis = bases_by_order[order - 1]
                for idx, ((mul, ir_out), u) in enumerate(zip(basis.irreps, basis.list)):
                    del mul
                    u = u.astype(x.dtype)
                    w = weights_by_order[order - 1][idx][species_index]
                    if ir_out not in out:
                        out[ir_out] = (
                            "special",
                            jnp.einsum("...jki,kc,cj->c...i", u, w, x),
                        )
                    else:
                        out[ir_out] = out[ir_out] + jnp.einsum("...ki,kc->c...i", u, w)
                for ir_out in out:
                    if isinstance(out[ir_out], tuple):
                        out[ir_out] = out[ir_out][1]
                    else:
                        out[ir_out] = jnp.einsum("c...ji,cj->c...i", out[ir_out], x)
            irreps_out = e3nn.Irreps(sorted(out.keys()))
            return e3nn.IrrepsArray.from_list(
                irreps_out,
                [out[ir][:, None, :] for _, ir in irreps_out],
                (value.shape[0],),
            )

        shape = jnp.broadcast_shapes(input.shape[:-2], index.shape)
        input = input.broadcast_to(shape + input.shape[-2:])
        index = jnp.broadcast_to(index, shape)
        fn = contract_one
        for _ in range(input.ndim - 2):
            fn = jax.vmap(fn)
        return fn(input, index).axis_to_mul()


class MACE(nn.Module):
    _graphs_properties: dict[str, Any]
    output_irreps: Any = "1x0e"
    hidden_irreps: Any = "128x0e + 128x1o"
    readout_mlp_irreps: Any = "16x0e"
    graph_key: str = "graph"
    output_key: str | None = None
    avg_num_neighbors: float = 1.0
    ninteractions: int = 2
    num_features: int | None = None
    radial_basis: dict[str, Any] | None = None
    lmax: int = 1
    correlation: int = 3
    activation: str = "silu"
    symmetric_tensor_product_basis: bool = False
    interaction_irreps: Any = "o3_restricted"
    skip_connection_first_layer: bool = True
    radial_network_hidden: Sequence[int] = (64, 64, 64)
    scalar_output: bool = False
    zmax: int = 86
    convolution_mode: int = 1
    species_encoding_key: str | None = None

    @nn.compact
    def __call__(self, inputs):
        e3nn = _require_e3nn_jax()
        species_indices = inputs["species"]
        graph = inputs[self.graph_key]
        distances = graph["distances"]
        vec = e3nn.IrrepsArray("1o", graph["vec"])
        switch = graph["switch"]
        edge_src = graph["edge_src"]
        edge_dst = graph["edge_dst"]

        output_irreps = e3nn.Irreps(self.output_irreps)
        hidden_irreps = e3nn.Irreps(self.hidden_irreps)
        readout_mlp_irreps = e3nn.Irreps(self.readout_mlp_irreps)

        if self.num_features is None:
            num_features = functools.reduce(math.gcd, (mul for mul, _ in hidden_irreps))
            hidden_irreps = e3nn.Irreps(
                [(mul // num_features, ir) for mul, ir in hidden_irreps]
            )
        else:
            num_features = int(self.num_features)

        if self.interaction_irreps == "o3_restricted":
            interaction_irreps = e3nn.Irreps.spherical_harmonics(self.lmax)
        elif self.interaction_irreps == "o3_full":
            interaction_irreps = e3nn.Irreps(e3nn.Irrep.iterator(self.lmax))
        else:
            interaction_irreps = e3nn.Irreps(self.interaction_irreps)
        convolution_irreps = num_features * interaction_irreps

        num_species = self.zmax + 2
        encoding_irreps = (num_features * hidden_irreps).filter("0e").regroup()
        if self.species_encoding_key is not None:
            species_encoding = nn.Dense(
                encoding_irreps.dim,
                use_bias=False,
            )(inputs[self.species_encoding_key])
        else:
            species_encoding = self.param(
                "species_encoding",
                lambda key, shape: jax.nn.standardize(
                    jax.random.normal(key, shape, dtype=jnp.float32)
                ),
                (num_species, encoding_irreps.dim),
            )
            species_encoding = jnp.take(species_encoding, species_indices, axis=0)
        node_feats = e3nn.IrrepsArray(encoding_irreps, species_encoding)

        cutoff = self._graphs_properties[self.graph_key]["cutoff"]
        radial_embedding = (
            _radial_basis(distances, dict(self.radial_basis or {}), cutoff) * switch[:, None]
        )

        if int(self.convolution_mode) == 0:
            yij = e3nn.spherical_harmonics(range(0, self.lmax + 1), vec, True)
        elif int(self.convolution_mode) == 1:
            yij = e3nn.spherical_harmonics(range(1, self.lmax + 1), vec, True)
        elif int(self.convolution_mode) == 2:
            yij = None
        else:
            raise ValueError("FeNNol MACE convolution_mode must be 0, 1, or 2.")

        outputs = []
        node_feats_all = []
        activation = _activation(self.activation)
        for layer in range(self.ninteractions):
            first = layer == 0
            last = layer == self.ninteractions - 1
            layer_irreps = num_features * (
                hidden_irreps if not last else hidden_irreps.filter(output_irreps)
            )

            skip = None
            if not first or self.skip_connection_first_layer:
                skip = e3nn.flax.Linear(
                    layer_irreps,
                    num_indexed_weights=num_species,
                    name=f"skip_tp_{layer}",
                    force_irreps_out=True,
                )(species_indices, node_feats)

            node_feats = e3nn.flax.Linear(node_feats.irreps, name=f"linear_up_{layer}")(
                node_feats
            )
            messages = node_feats[edge_src]
            if int(self.convolution_mode) == 0:
                messages = e3nn.tensor_product(
                    messages,
                    yij,
                    filter_ir_out=convolution_irreps,
                    regroup_output=True,
                )
            elif int(self.convolution_mode) == 1:
                messages = e3nn.concatenate(
                    [
                        messages.filter(convolution_irreps),
                        e3nn.tensor_product(
                            messages,
                            yij,
                            filter_ir_out=convolution_irreps,
                        ),
                    ]
                ).regroup()
            else:
                messages = e3nn.tensor_product_with_spherical_harmonics(
                    messages,
                    vec,
                    self.lmax,
                ).filter(convolution_irreps).regroup()

            mix = e3nn.flax.MultiLayerPerceptron(
                [*tuple(self.radial_network_hidden), messages.irreps.num_irreps],
                act=activation,
                output_activation=False,
                name=f"radial_network_{layer}",
                gradient_normalization="element",
            )(radial_embedding)
            messages = messages * mix
            node_feats = (
                e3nn.IrrepsArray.zeros(messages.irreps, node_feats.shape[:1], messages.dtype)
                .at[edge_dst]
                .add(messages)
            )
            node_feats = (
                e3nn.flax.Linear(convolution_irreps, name=f"linear_dn_{layer}")(node_feats)
                / self.avg_num_neighbors
            )

            if first and not self.skip_connection_first_layer:
                node_feats = e3nn.flax.Linear(
                    node_feats.irreps,
                    num_indexed_weights=num_species,
                    name=f"skip_tp_{layer}",
                )(species_indices, node_feats)

            node_feats = SymmetricContraction(
                keep_irrep_out={ir for _, ir in layer_irreps},
                correlation=self.correlation,
                num_species=num_species,
                gradient_normalization="element",
                symmetric_tensor_product_basis=self.symmetric_tensor_product_basis,
                name=f"SymmetricContraction_{layer}",
            )(node_feats, species_indices)
            node_feats = e3nn.flax.Linear(
                layer_irreps,
                name=f"linear_contraction_{layer}",
            )(node_feats)

            if skip is not None:
                node_feats = node_feats + skip

            if last:
                num_vectors = readout_mlp_irreps.filter(drop=["0e", "0o"]).num_irreps
                layer_out = e3nn.flax.Linear(
                    (readout_mlp_irreps + e3nn.Irreps(f"{num_vectors}x0e")).simplify(),
                    name="hidden_linear_readout_last",
                )(node_feats)
                layer_out = e3nn.gate(
                    layer_out,
                    even_act=activation,
                    even_gate_act=None,
                )
                layer_out = e3nn.flax.Linear(
                    output_irreps,
                    name="linear_readout_last",
                )(layer_out)
            else:
                layer_out = e3nn.flax.Linear(
                    output_irreps,
                    name=f"linear_readout_{layer}",
                )(node_feats)

            if self.scalar_output:
                layer_out = layer_out.filter("0e").array
            outputs.append(layer_out)
            node_feats_all.append(node_feats.filter("0e").array)

        output = jnp.stack(outputs, axis=1) if self.scalar_output else e3nn.stack(outputs, axis=1)
        node_feats_all = jnp.concatenate(node_feats_all, axis=-1)
        output_key = self.output_key if self.output_key is not None else self.name
        return {
            **inputs,
            output_key: output,
            output_key + "_node_feats": node_feats_all,
        }


class RepulsionNLH(nn.Module):
    graph_key: str = "graph_embed"
    output_key: str | None = None

    @nn.compact
    def __call__(self, inputs):
        graph = inputs[self.graph_key]
        species = inputs["species"]
        edge_src, edge_dst = graph["edge_src"], graph["edge_dst"]
        distances = graph["distances"]
        nlh_coeffs = _nlh_coeffs()
        zmax = int(np.max(nlh_coeffs[:, 0]))
        ab = np.zeros(((zmax + 1) ** 2, 6), dtype=np.float32)
        for row in nlh_coeffs:
            z1, z2 = int(row[0]), int(row[1])
            ab[z1 + zmax * z2] = row[2:8]
            ab[z2 + zmax * z1] = row[2:8]
        ab = ab.reshape((zmax + 1) ** 2, 3, 2)
        cs = jnp.asarray(ab[:, :, 0], dtype=distances.dtype)
        alphas = jnp.asarray(ab[:, :, 1], dtype=distances.dtype)
        s12 = species[edge_src] + zmax * species[edge_dst]
        coeff = cs[s12]
        alpha = alphas[s12]
        phi = (coeff * jnp.exp(-alpha * distances[:, None])).sum(axis=-1)
        zij = species[edge_src].astype(distances.dtype) * species[edge_dst].astype(distances.dtype) * graph["switch"]
        pair = zij * phi / distances
        output = (HA_TO_EV * 0.5 * ANG) * jax.ops.segment_sum(pair, edge_src, species.shape[0])
        output_key = self.output_key if self.output_key is not None else self.name
        return {**inputs, output_key: output}


class Concatenate(nn.Module):
    keys: tuple[str, ...]
    axis: int = -1
    output_key: str | None = None

    @nn.compact
    def __call__(self, inputs):
        output = jnp.concatenate([inputs[key] for key in self.keys], axis=self.axis)
        output_key = self.output_key if self.output_key is not None else self.name
        return {**inputs, output_key: output}


class Add(nn.Module):
    keys: tuple[str, ...]
    output_key: str | None = None

    @nn.compact
    def __call__(self, inputs):
        output = 0
        for key in self.keys:
            output = output + inputs[key]
        output_key = self.output_key if self.output_key is not None else self.name
        return {**inputs, output_key: output}


class Reshape(nn.Module):
    key: str
    shape: tuple[int, ...]
    output_key: str | None = None

    @nn.compact
    def __call__(self, inputs):
        output_key = self.output_key if self.output_key is not None else self.key
        return {**inputs, output_key: jnp.reshape(inputs[self.key], self.shape)}


class EnsembleStat(nn.Module):
    key: str

    @nn.compact
    def __call__(self, inputs):
        values = inputs[self.key]
        if values.ndim > 1:
            mean = jnp.mean(values, axis=-1)
            std = jnp.std(values, axis=-1)
        else:
            mean = values
            std = jnp.zeros_like(values)
        return {
            **inputs,
            f"{self.key}_mean": mean,
            f"{self.key}_std": std,
        }


class ScatterSystem(nn.Module):
    key: str
    output_key: str | None = None

    @nn.compact
    def __call__(self, inputs):
        output = jax.ops.segment_sum(
            inputs[self.key],
            inputs["batch_index"],
            num_segments=inputs["natoms"].shape[0],
        )
        output_key = self.output_key if self.output_key is not None else self.name
        return {**inputs, output_key: output}


class SumAxis(nn.Module):
    key: str
    axis: int | Sequence[int] | None = None
    output_key: str | None = None
    norm: str | None = None

    @nn.compact
    def __call__(self, inputs):
        values = inputs[self.key]
        output = jnp.sum(values, axis=self.axis)
        if self.norm is not None:
            norm = str(self.norm).strip().lower()
            axes = (self.axis,) if isinstance(self.axis, int) else tuple(self.axis or ())
            dim = np.prod([values.shape[axis] for axis in axes]) if axes else values.size
            if norm == "dim":
                output = output / dim
            elif norm == "sqrt":
                output = output / dim**0.5
            elif norm != "none":
                raise ValueError(f"Unknown FeNNol SUM_AXIS norm {self.norm!r}.")
        output_key = self.output_key if self.output_key is not None else self.key
        return {**inputs, output_key: output}


class Activation(nn.Module):
    key: str
    activation: str
    scale_out: float = 1.0
    shift_out: float = 0.0
    output_key: str | None = None

    @nn.compact
    def __call__(self, inputs):
        output = self.scale_out * _activation(self.activation)(inputs[self.key]) + self.shift_out
        output_key = self.output_key if self.output_key is not None else self.key
        return {**inputs, output_key: output}


class ChemicalConstant(nn.Module):
    value: Any
    output_key: str | None = None
    trainable: bool = False

    @nn.compact
    def __call__(self, inputs):
        species = inputs["species"]
        if isinstance(self.value, (int, float)):
            constant = [float(self.value)] * 119
        elif isinstance(self.value, (list, tuple)):
            constant = list(self.value)
        elif hasattr(self.value, "items"):
            try:
                from ase.data import atomic_numbers
            except Exception as exc:
                raise ImportError("FeNNol CHEMICAL_CONSTANT dict values require ase.data.atomic_numbers.") from exc
            constant = [0.0] * 119
            for symbol, value in self.value.items():
                constant[int(atomic_numbers[str(symbol)])] = float(value)
        else:
            raise ValueError(f"Unsupported FeNNol CHEMICAL_CONSTANT value type {type(self.value)!r}.")
        if self.trainable:
            constant = self.param("constant", lambda rng: jnp.asarray(constant, dtype=jnp.float32))
        else:
            constant = jnp.asarray(constant, dtype=inputs["coordinates"].dtype)
        output = jnp.take(constant, species, axis=0)
        output_key = self.output_key if self.output_key is not None else self.name
        return {**inputs, output_key: output}


class FENNIXModules(nn.Module):
    layers: tuple[tuple[type[nn.Module], dict[str, Any]], ...]

    @nn.compact
    def __call__(self, inputs):
        outputs = inputs
        for module_cls, params in self.layers:
            outputs = module_cls(**params)(outputs)
        return outputs


@dataclass(frozen=True)
class GraphSpec:
    key: str
    cutoff: float
    parent_graph: str | None = None
    remove_hydrogens: bool = False
    switch_params: dict[str, Any] | None = None


def _graph_from_positions_bruteforce(positions: np.ndarray, cutoff: float) -> tuple[np.ndarray, np.ndarray]:
    n_atoms = int(positions.shape[0])
    if n_atoms < 2:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)
    p1, p2 = np.triu_indices(n_atoms, 1)
    vec = positions[p2] - positions[p1]
    d2 = np.sum(vec * vec, axis=-1)
    keep = d2 < cutoff * cutoff
    src = np.concatenate([p1[keep], p2[keep]]).astype(np.int32)
    dst = np.concatenate([p2[keep], p1[keep]]).astype(np.int32)
    return src, dst


def _graph_edges_from_atoms(atoms, positions: np.ndarray, cutoff: float) -> tuple[np.ndarray, np.ndarray]:
    try:
        work = atoms.copy()
        work.set_positions(positions)
        work.set_pbc(False)
        src, dst = neighbor_list("ij", work, float(cutoff))
        return np.asarray(src, dtype=np.int32), np.asarray(dst, dtype=np.int32)
    except Exception:
        return _graph_from_positions_bruteforce(positions, cutoff)


def _graph_from_edges(
    src: np.ndarray,
    dst: np.ndarray,
    positions: np.ndarray,
    cutoff: float,
) -> dict[str, Any]:
    src = np.asarray(src, dtype=np.int32)
    dst = np.asarray(dst, dtype=np.int32)
    if src.size:
        vec = positions[dst] - positions[src]
        d2 = np.sum(vec * vec, axis=-1).astype(np.float32)
    else:
        d2 = np.zeros(0, dtype=np.float32)
    return {
        "edge_src": jnp.asarray(src, dtype=jnp.int32),
        "edge_dst": jnp.asarray(dst, dtype=jnp.int32),
        "d12": jnp.asarray(d2),
        "cutoff": float(cutoff),
    }


def _graph_from_atoms(
    atoms,
    positions: np.ndarray,
    cutoff: float,
) -> dict[str, Any]:
    src, dst = _graph_edges_from_atoms(atoms, positions, cutoff)
    return _graph_from_edges(src, dst, positions, cutoff)


def _filter_graph_from_parent(
    parent: dict[str, Any],
    species: np.ndarray,
    positions: np.ndarray,
    cutoff: float,
    *,
    remove_hydrogens: bool = False,
) -> dict[str, Any]:
    parent_src = np.asarray(parent["edge_src"], dtype=np.int32)
    parent_dst = np.asarray(parent["edge_dst"], dtype=np.int32)
    parent_d2 = np.asarray(parent["d12"], dtype=float)
    mask = parent_d2 < cutoff * cutoff
    if remove_hydrogens and parent_src.size:
        mask = np.logical_and(mask, species[parent_src] > 1)
    filter_indices = np.nonzero(mask)[0].astype(np.int32)
    graph = _graph_from_edges(
        parent_src[filter_indices],
        parent_dst[filter_indices],
        positions,
        cutoff,
    )
    graph["filter_indices"] = jnp.asarray(filter_indices, dtype=jnp.int32)
    return graph


def _process_graph(
    graph,
    coordinates,
    switch_params: dict[str, Any] | None = None,
    cutoff: float | None = None,
):
    src = graph["edge_src"]
    dst = graph["edge_dst"]
    vec = coordinates[dst] - coordinates[src]
    d2 = jnp.sum(vec * vec, axis=-1)
    distances = _safe_sqrt(d2)
    cutoff_value = float(cutoff if cutoff is not None else graph["cutoff"])
    edge_mask = d2 < cutoff_value * cutoff_value
    switch_params = dict(switch_params or {})
    switch_type = str(switch_params.get("switch_type", "hard")).strip().lower()
    if switch_type == "hard":
        switch = jnp.where(edge_mask, 1.0, 0.0)
    elif switch_type == "polynomial":
        p = float(switch_params.get("p", 3.0))
        d = distances / cutoff_value
        switch = (
            1.0
            - 0.5 * (p + 1) * (p + 2) * d**p
            + p * (p + 2) * d ** (p + 1)
            - 0.5 * p * (p + 1) * d ** (p + 2)
        )
        switch = jnp.where(edge_mask, switch, 0.0)
    else:
        raise NotImplementedError(f"FeNNol graph switch {switch_type!r} is not supported.")
    return {**graph, "vec": vec, "d12": d2, "distances": distances, "edge_mask": edge_mask, "switch": switch}


@dataclass(frozen=True)
class BlockIndexSpec:
    output_key: str
    block_names: tuple[str, ...]


def _normalise_block_name(name: str) -> str:
    return str(name).strip().upper()


def _element_block_name(atomic_number: int, block_names: tuple[str, ...]) -> str | None:
    if int(atomic_number) >= len(chemical_symbols):
        return None
    symbol = chemical_symbols[int(atomic_number)]
    exact = {name: name for name in block_names}
    folded = {_normalise_block_name(name): name for name in block_names}
    if symbol in exact:
        return symbol
    if _normalise_block_name(symbol) in folded:
        return folded[_normalise_block_name(symbol)]
    family = _element_family_name(int(atomic_number))
    if family is not None and family in exact:
        return family
    if family is not None and _normalise_block_name(family) in folded:
        return folded[_normalise_block_name(family)]
    return None


def _block_index(species: np.ndarray, block_names: tuple[str, ...]) -> dict[str, jnp.ndarray]:
    out: dict[str, jnp.ndarray] = {}
    unmatched: list[str] = []
    for name in block_names:
        idxs = []
        for idx, z in enumerate(species):
            block = _element_block_name(int(z), block_names)
            if block is None and name == block_names[0]:
                symbol = chemical_symbols[int(z)] if int(z) < len(chemical_symbols) else f"Z={int(z)}"
                unmatched.append(symbol)
            if block == name:
                idxs.append(idx)
        out[name] = jnp.asarray(np.asarray(idxs, dtype=np.int32))
    if unmatched:
        raise ValueError(
            "FeNNol block-index modules do not define a block for input elements "
            f"{sorted(set(unmatched))}; available blocks are {list(block_names)}."
        )
    return out


MODULES: dict[str, type[nn.Module]] = {
    "MACE": MACE,
    "SPECIES_ENCODING": SpeciesEncoding,
    "NEURAL_NET": FullyConnectedNet,
    "CHARGE_HYPOTHESIS": ChargeHypothesis,
    "CONCATENATE": Concatenate,
    "RASTER": RaSTER,
    "BLOCK_INDEX_NET": BlockIndexNet,
    "ENSEMBLE_STAT": EnsembleStat,
    "REPULSION_NLH": RepulsionNLH,
    "ADD": Add,
    "RESHAPE": Reshape,
    "SCATTER_SYSTEM": ScatterSystem,
    "SUM_AXIS": SumAxis,
    "ACTIVATION": Activation,
    "CHEMICAL_CONSTANT": ChemicalConstant,
}

PREPROCESSING: dict[str, type[nn.Module] | None] = {
    "GRAPH_FILTER": GraphFilterProcessor,
    "BLOCK_INDEXER": None,
}


def _registry_key(name: str, params: dict[str, Any], field: str) -> str:
    value = params.get(field) or params.get("FID") or params.get("module_name") or name
    return str(value).strip().upper()


def _layer_params(name: str, params: dict[str, Any]) -> dict[str, Any]:
    cleaned = {
        key: value
        for key, value in params.items()
        if key not in {"module_name", "FID", "FPID"}
    }
    for key in (
        "neurons",
        "hidden_neurons",
        "keys",
        "shape",
        "lode_extra_powers",
        "update_hidden",
        "radial_network_hidden",
        "axis",
    ):
        if key in cleaned and isinstance(cleaned[key], list):
            cleaned[key] = tuple(cleaned[key])
    cleaned["name"] = name
    return cleaned


def _build_fennix_layers(
    program: FennolProgram,
):
    graph_specs: list[GraphSpec] = []
    graph_cutoffs: dict[str, float] = {}
    block_index_keys: set[str] = set()
    layers: list[tuple[type[nn.Module], dict[str, Any]]] = []
    base_cutoff = float(program.cutoff or 0.0)

    for step in program.preprocessing:
        name = step.name
        params = dict(step.params)
        if name == "graph" and "module_name" not in params and "FPID" not in params:
            graph_key = str(params.get("graph_key", name))
            cutoff = float(params.get("cutoff", base_cutoff))
            switch_params = dict(params.get("switch_params", {}))
            if bool(switch_params.get("trainable", False)):
                raise ValueError(
                    f"FeNNol base graph {graph_key!r} uses a trainable switch; "
                    "this MAPLE runtime currently supports trainable switches only in GRAPH_FILTER."
                )
            graph_specs.append(GraphSpec(graph_key, cutoff, switch_params=switch_params))
            graph_cutoffs[graph_key] = cutoff
            continue

        key = step.fid
        if key == "GRAPH_FILTER":
            graph_key = str(params.get("graph_key", name))
            parent_graph = str(params.get("parent_graph", "graph"))
            cutoff = float(params["cutoff"])
            graph_specs.append(
                GraphSpec(
                    graph_key,
                    cutoff,
                    parent_graph=parent_graph,
                    remove_hydrogens=bool(params.get("remove_hydrogens", False)),
                )
            )
            graph_cutoffs[graph_key] = cutoff
            layer_name = f"{graph_key}_Filter_{parent_graph}"
            layers.append(
                (
                    GraphFilterProcessor,
                    {
                        "name": layer_name,
                        "graph_key": graph_key,
                        "parent_graph": parent_graph,
                        "output_key": graph_key,
                        "cutoff": float(params["cutoff"]),
                        "switch_params": dict(params.get("switch_params", {})),
                        "remove_hydrogens": bool(params.get("remove_hydrogens", False)),
                    },
                )
            )
            continue
        if key == "BLOCK_INDEXER":
            block_index_keys.add(str(params.get("output_key", "block_index")))
            continue
        supported = ", ".join(["GRAPH", *sorted(PREPROCESSING)])
        raise ValueError(
            f"Unsupported FeNNol preprocessing {key!r} in block {name!r}. "
            f"Supported preprocessing: {supported}"
        )

    for step in program.modules:
        name = step.name
        params = dict(step.params)
        key = step.fid
        module_cls = MODULES.get(key)
        if module_cls is None:
            supported = ", ".join(sorted(MODULES))
            raise ValueError(
                f"Unsupported FeNNol module {key!r} in block {name!r}. "
                f"Supported modules: {supported}"
            )
        if key == "MACE":
            _require_e3nn_jax()
        if key == "BLOCK_INDEX_NET":
            block_index_keys.add(str(params.get("block_index_key", "block_index")))
        layer_params = _layer_params(name, params)
        fields = getattr(module_cls, "__dataclass_fields__", {})
        if "_graphs_properties" in fields:
            layer_params["_graphs_properties"] = {
                graph_key: {"cutoff": cutoff, "directed": True}
                for graph_key, cutoff in graph_cutoffs.items()
            }
        layers.append((module_cls, layer_params))

    if not graph_cutoffs:
        raise ValueError("FeNNol model did not define any supported graph preprocessing.")
    block_index_specs = _infer_block_index_specs(program, tuple(sorted(block_index_keys)))
    return tuple(layers), tuple(graph_specs), block_index_specs


def _infer_block_index_specs(program: FennolProgram, block_index_keys: tuple[str, ...]) -> tuple[BlockIndexSpec, ...]:
    if not block_index_keys:
        return ()
    variables = program.variables.get("params", {})
    names_by_key: dict[str, set[str]] = {key: set() for key in block_index_keys}
    for step in program.modules:
        if step.fid != "BLOCK_INDEX_NET":
            continue
        key = str(step.params.get("block_index_key", "block_index"))
        params = variables.get(step.name, {})
        if isinstance(params, dict):
            for name, value in params.items():
                if isinstance(value, dict):
                    names_by_key.setdefault(key, set()).add(str(name))
    specs = []
    for key in block_index_keys:
        names = tuple(sorted(names_by_key.get(key, ())))
        if names:
            specs.append(BlockIndexSpec(key, names))
    return tuple(specs)


def _aggregate_energy(outputs: dict[str, Any], energy_terms: tuple[str, ...]):
    if not energy_terms:
        raise ValueError("FeNNol model has no energy_terms for potential energy.")
    species = outputs["species"]
    nsys = outputs["natoms"].shape[0]
    atomic_energies = 0.0
    system_energies = 0.0
    for term in energy_terms:
        if term not in outputs:
            raise KeyError(f"FeNNol energy term {term!r} was not produced by the module chain.")
        values = outputs[term]
        if values.ndim > 1 and values.shape[-1] == 1:
            values = jnp.squeeze(values, axis=-1)
        if values.shape[0] == nsys and nsys != species.shape[0]:
            system_energies = system_energies + values
        else:
            if values.shape != species.shape:
                raise ValueError(
                    f"FeNNol energy term {term!r} has shape {values.shape}; "
                    f"expected atomic shape {species.shape} or system shape {(nsys,)}."
                )
            atomic_energies = atomic_energies + values
    if isinstance(atomic_energies, jnp.ndarray):
        energies = jax.ops.segment_sum(
            atomic_energies,
            outputs["batch_index"],
            num_segments=nsys,
        )
    else:
        energies = jnp.zeros(nsys, dtype=outputs["coordinates"].dtype)
    total = energies + system_energies
    return total, {**outputs, "atomic_energies": atomic_energies, "total_energy": total}


############################################
#----------- Runtime Class ----------------
############################################

@dataclass
class FENNIXRuntime:
    variables: dict[str, Any]
    modules: FENNIXModules
    energy_terms: tuple[str, ...]
    energy_unit: str
    graph_specs: tuple[GraphSpec, ...]
    block_index_specs: tuple[BlockIndexSpec, ...]
    dtype: Any

    @property
    def graph_cutoffs(self) -> dict[str, float]:
        return {spec.key: spec.cutoff for spec in self.graph_specs}

    @property
    def _base_graph_keys(self) -> tuple[str, ...]:
        return tuple(spec.key for spec in self.graph_specs if spec.parent_graph is None)

    @property
    def _base_graph_specs(self) -> tuple[GraphSpec, ...]:
        return tuple(spec for spec in self.graph_specs if spec.parent_graph is None)

    def __post_init__(self):
        def apply_total(variables, data):
            out = self.modules.apply(variables, data)
            return _aggregate_energy(out, self.energy_terms)

        self._apply_total = jax.jit(apply_total)

        def scalar_energy(coordinates, data):
            data = {**data, "coordinates": coordinates}
            for spec in self._base_graph_specs:
                data = {
                    **data,
                    spec.key: _process_graph(
                        data[spec.key],
                        coordinates,
                        spec.switch_params,
                        spec.cutoff,
                    ),
                }
            energy, out = apply_total(self.variables, data)
            return energy.sum(), out

        def hvp_energy_forces(coordinates, vector, data):
            def energy_only(x):
                return scalar_energy(x, data)[0]

            (energy, grad), (_, hvp) = jax.jvp(
                jax.value_and_grad(energy_only),
                (coordinates,),
                (vector,),
            )
            return energy, -grad, hvp

        # Basic products：
        self._energy_and_grad = jax.jit(jax.value_and_grad(scalar_energy, has_aux=True))
        self._hessian = jax.jit(jax.hessian(lambda coordinates, data: scalar_energy(coordinates, data)[0]))
        self._hvp = jax.jit(hvp_energy_forces)

    def make_inputs(self, atoms, total_charge: float):
        species = np.asarray(atoms.get_atomic_numbers(), dtype=np.int32)
        positions = np.asarray(atoms.get_positions(), dtype=np.float64 if self.dtype == jnp.float64 else np.float32)
        coordinates = jnp.asarray(positions, dtype=self.dtype)
        inputs = {
            "species": jnp.asarray(species),
            "coordinates": coordinates,
            "natoms": jnp.asarray([len(species)], dtype=jnp.int32),
            "batch_index": jnp.zeros(len(species), dtype=jnp.int32),
            "total_charge": jnp.asarray(total_charge, dtype=self.dtype),
        }
        for spec in self.block_index_specs:
            inputs[spec.output_key] = _block_index(species, spec.block_names)
        for spec in self.graph_specs:
            if spec.parent_graph is None:
                inputs[spec.key] = _graph_from_atoms(
                    atoms,
                    positions,
                    spec.cutoff,
                )
            else:
                if spec.parent_graph not in inputs:
                    raise ValueError(
                        f"FeNNol graph filter {spec.key!r} references missing parent graph "
                        f"{spec.parent_graph!r}."
                    )
                inputs[spec.key] = _filter_graph_from_parent(
                    inputs[spec.parent_graph],
                    species,
                    positions,
                    spec.cutoff,
                    remove_hydrogens=spec.remove_hydrogens,
                )
        return inputs

    def energy_forces(self, atoms, total_charge: float):
        data = self.make_inputs(atoms, total_charge)
        (energy, out), grad = self._energy_and_grad(data["coordinates"], data)
        return float(energy), np.asarray(-grad), out

    def energy(self, atoms, total_charge: float):
        data = self.make_inputs(atoms, total_charge)
        for spec in self._base_graph_specs:
            data = {
                **data,
                spec.key: _process_graph(
                    data[spec.key],
                    data["coordinates"],
                    spec.switch_params,
                    spec.cutoff,
                ),
            }
        energy, out = self._apply_total(self.variables, data)
        return float(energy.sum()), out

    def hessian(self, atoms, total_charge: float):
        data = self.make_inputs(atoms, total_charge)
        hess = self._hessian(data["coordinates"], data)
        return np.asarray(hess).reshape(3 * len(atoms), 3 * len(atoms))

    def hvp(self, atoms, total_charge: float, vector):
        data = self.make_inputs(atoms, total_charge)
        direction = np.asarray(vector, dtype=np.float64 if self.dtype == jnp.float64 else np.float32)
        direction = direction.reshape(len(atoms), 3)
        energy, forces, hvp = self._hvp(data["coordinates"], jnp.asarray(direction, dtype=self.dtype), data)
        return (
            np.asarray(hvp).reshape(3 * len(atoms)),
            np.asarray(forces).reshape(3 * len(atoms)),
            float(energy),
        )

############################################
#----------- Runtime Loader ----------------
############################################

def load_fennol_runtime(model_path: str | Path, *, use_float64: bool = False) -> FENNIXRuntime:
    if use_float64:
        jax.config.update("jax_enable_x64", True)
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"FeNNol model file not found: {path}")
    program = load_fennol_program(path)
    layers, graph_specs, block_index_specs = _build_fennix_layers(program)
    dtype = jnp.float64 if use_float64 else jnp.float32
    raw_unit = program.energy_unit
    if raw_unit in {"ev", "electronvolt", "electronvolts"}:
        energy_unit = "eV"
    elif raw_unit in {"ha", "hartree", "hartrees"}:
        energy_unit = "hartree"
    else:
        raise ValueError(f"Unsupported FeNNol energy unit: {program.energy_unit}")
    return FENNIXRuntime(
        variables=_jaxify_variables(program.variables),
        modules=FENNIXModules(layers),
        energy_terms=program.energy_terms,
        energy_unit=energy_unit,
        graph_specs=graph_specs,
        block_index_specs=block_index_specs,
        dtype=dtype,
    )
