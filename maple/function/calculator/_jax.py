"""Standard JAX exports and coordinate derivatives, without native model libraries."""
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import export
import numpy as np
from .calculator_base import _convert_properties


def load_model(model_path, device='cpu', *, require_vjp=False):
    model = export.deserialize(Path(model_path).read_bytes())
    name = str(device or 'cpu')
    platform = 'cuda' if name.startswith('cuda') else 'cpu'
    if platform not in model.platforms:
        raise ValueError(f'JAX export does not support platform {platform!r}: {model_path}')
    if require_vjp and not model.has_vjp():
        raise ValueError(f'JAX export requires a saved first VJP: {model_path}')
    backend = 'gpu' if platform == 'cuda' else 'cpu'
    index = int(name.split(':')[1]) if ':' in name else 0
    return model, jax.devices(backend)[index]


def prepare_strain(coordinates, cells, strain, *, batch=None):
    """Apply symmetric strain to dense or flat coordinates and row-vector cells."""
    transform = jnp.eye(3, dtype=strain.dtype) + .5 * (strain + strain.swapaxes(-1, -2))
    if coordinates.ndim == 3:
        positions = jnp.einsum('bni,bij->bnj', coordinates, transform)
    elif batch is None:
        positions = coordinates @ transform[0]
    else:
        positions = jnp.einsum('ni,nij->nj', coordinates, transform[batch])
    return positions, cells @ transform


def compile_strain(forward, *, position_key, cell_key, energy_key, batch_key=None):
    """Bind native fields to one energy/strain derivative with native outputs."""
    def strained_energy(data, strain):
        coordinates, cells = prepare_strain(
            data[position_key], data[cell_key], strain,
            batch=data[batch_key] if batch_key is not None else None,
        )
        updated = {**data, position_key: coordinates, cell_key: cells}
        output = forward(updated)
        return output[energy_key].sum(), output
    return jax.jit(jax.value_and_grad(strained_energy, argnums=1, has_aux=True))


def array_properties(energy, members, properties, *, forces=None,
                     strain_gradient=None, volume=None, source_unit='eV'):
    """Split native arrays by member; E/F/virial use Hartree, stress eV/Å³."""
    unknown = set(properties) - {'energy', 'free_energy', 'forces', 'stress', 'virial'}
    if unknown:
        raise NotImplementedError(f'Unsupported array properties: {sorted(unknown)}')
    result = {}
    if 'energy' in properties or 'free_energy' in properties:
        values = np.asarray(energy, dtype=float).reshape(-1)[:len(members)].copy()
        for name in ('energy', 'free_energy'):
            if name in properties:
                result[name] = values
    if 'forces' in properties:
        if forces is None:
            raise NotImplementedError('No forces array supplied')
        array = np.asarray(forces, dtype=float)
        if array.ndim == 3:
            result['forces'] = [array[i, :len(atoms)].copy() for i, atoms in enumerate(members)]
        else:
            offsets = np.cumsum([0, *map(len, members)])
            result['forces'] = [array[offsets[i]:offsets[i + 1]].copy() for i in range(len(members))]
    if set(properties) & {'stress', 'virial'}:
        if strain_gradient is None:
            raise NotImplementedError('No strain derivative supplied')
        derivative = np.asarray(strain_gradient, dtype=float)[:len(members)]
        if 'stress' in properties:
            if volume is None:
                raise ValueError('Stress requires cell volumes')
            volumes = np.asarray(volume, dtype=float).reshape(-1)[:len(members)]
            result['stress'] = derivative / volumes[:, None, None]
        if 'virial' in properties:
            result['virial'] = -derivative
    return _convert_properties(result, source_unit=source_unit)


def compile_hvp(forward, *, position_key, force_key):
    """Differentiate the native forces using the export's saved first VJP."""
    def product(data, vector):
        def contraction(coordinates):
            output = forward({**data, position_key: coordinates})
            return -jnp.vdot(output[force_key], vector), output
        (_, output), value = jax.value_and_grad(contraction, has_aux=True)(data[position_key])
        return value, output
    return jax.jit(product)


def hessian(hvp, data, coordinates, atom_count, device):
    """Assemble real-atom rows from a compiled HVP, excluding padding."""
    rows = []
    for index in range(atom_count * 3):
        vector = np.zeros(coordinates.shape, coordinates.dtype)
        vector.flat[index] = 1.
        product, _ = hvp(data, jax.device_put(vector, device))
        rows.append(np.asarray(product)[:atom_count].flatten())
    return np.stack(rows)
