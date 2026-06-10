from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Any
from mace_jax.nnx_utils import state_to_pure_dict

import ase
import jax
import jax.numpy as jnp
import numpy as np
import torch
from ase.build import bulk
from mace.data.atomic_data import AtomicData
from mace.data.utils import config_from_atoms
from mace.tools import torch_geometric
from mace.tools.multihead_tools import AtomicNumberTable
from mace.tools.scripts_utils import extract_config_mace_model
from mace.tools.torch_geometric.batch import Batch

from mace_jax.cli.mace_jax_from_torch import convert_model
from mace_jax.modules.wrapper_ops import CuEquivarianceConfig
from mace_jax.tools.device import configure_torch_runtime, get_torch_device
from mace_jax.tools.foundation_models import load_foundation_torch_model


@dataclass
class BenchmarkResult:
    mean: float
    std: float
    min_time: float


def load_foundation_model(
    source: str = 'mp',
    variant: str | None = None,
    device: Any | str = 'cpu',
) -> torch.nn.Module:
    """Return a pretrained Torch MACE foundation model on the specified device."""

    loader_kwargs: dict[str, Any] = {'device': device}
    model = load_foundation_torch_model(
        source=source.lower(),
        model=variant,
        device=loader_kwargs['device'],
    )
    return model.float().eval()


def extract_foundation_metadata(
    torch_model: torch.nn.Module,
) -> tuple[dict[str, Any], AtomicNumberTable, float]:
    config = extract_config_mace_model(torch_model)
    config['torch_model_class'] = torch_model.__class__.__name__
    atomic_numbers = tuple(int(z) for z in config['atomic_numbers'])
    z_table = AtomicNumberTable(atomic_numbers)
    cutoff = float(config['r_max'])
    return config, z_table, cutoff


def build_example_atoms(symbol: str = 'Si', repeat: int = 2) -> ase.Atoms:
    """Construct a simple crystalline structure for benchmarking."""
    atoms = bulk(symbol, 'diamond', a=5.43)
    return atoms.repeat((repeat, repeat, repeat))


def batch_to_jax(batch: Batch) -> dict[str, jnp.ndarray]:
    converted: dict[str, Any] = {}
    for key in batch.keys:
        value = batch[key]
        if isinstance(value, torch.Tensor):
            converted[key] = jnp.asarray(value.detach().cpu().numpy())
        else:
            converted[key] = value
    return converted


def prepare_batches(
    torch_model: torch.nn.Module, atoms: ase.Atoms, device: Any
) -> tuple[Batch, dict[str, jnp.ndarray], dict[str, Any]]:
    config, z_table, cutoff = extract_foundation_metadata(torch_model)
    config_atoms = config_from_atoms(atoms)
    config_atoms.pbc = [bool(x) for x in config_atoms.pbc]
    atomic_data = AtomicData.from_config(
        config_atoms,
        z_table=z_table,
        cutoff=cutoff,
    )
    batch_torch = torch_geometric.batch.Batch.from_data_list([atomic_data])
    batch_torch = batch_torch.to(device)
    batch_jax = batch_to_jax(batch_torch)
    return batch_torch, batch_jax, config


class CompiledMaceInference:
    """Minimal MACE inference wrapper with all torch.compile + CUDA-graph optimizations,
    using the benchmark's precomputed (static) neighbor list.

    (1) rebuild via prepare(extract_model) so a loaded foundation model's e3nn modules
        compile (avoids the Irrep dynamo-guard crash);
    (2) configure_autograd_for_compile (allow_in_graph(autograd.grad) + trace_autograd_ops)
        so the inner force autograd.grad traces into the graph, AND mace's retain_graph fix
        (retain_graph kept alive while compiling) so compiled inference (training=False)
        doesn't fail with "backward through the graph a second time";
    (3) torch.compile(mode="reduce-overhead", fullgraph=True);
    (4) per-call fresh input leaves (positions/cell) so the strain-derived intermediates the
        retained backward references can't be overwritten in the static cudagraph pool across
        steps (this also removes any input mutation, so no cudagraph_support_input_mutation);
    (5) cudagraph_mark_step_begin() each step;
    (6) detached+cloned outputs so no grad-requiring tensor is held across steps."""

    def __init__(self, model, compile_mode, device, *,
                 compute_force=True, compute_stress=True, cueq=False):
        import torch._dynamo as dynamo
        from mace.tools.compile import (
            configure_autograd_for_compile,
            disable_e3nn_codegen,
            prepare,
            simplify,
        )
        from mace.tools.scripts_utils import extract_model

        self.compute_force = compute_force
        self.compute_stress = compute_stress
        self.use_cudagraphs = compile_mode in ('reduce-overhead', 'max-autotune')
        # No cudagraph_support_input_mutation needed: per-call fresh input leaves
        # (see __call__) mean the graph never mutates a re-fed input across steps.
        configure_autograd_for_compile(allow_autograd=True)
        dynamo.config.error_on_recompile = True
        with disable_e3nn_codegen():
            # cueq modules can't be rebuilt by extract_model, so just simplify in place
            # (like MACECalculator); plain e3nn rebuilds fresh.
            prepared = (simplify(model) if cueq
                        else prepare(extract_model)(model=model, map_location=device))
        self.model = torch.compile(prepared, mode=compile_mode, fullgraph=True)

    def __call__(self, batch):
        data = batch.to_dict() if hasattr(batch, 'to_dict') else dict(batch)
        # fresh input leaves each step: forces/stress need leaves, and a fresh buffer keeps
        # the strain mutation from clobbering the static cudagraph pool across replays.
        data['positions'] = data['positions'].detach().clone().requires_grad_(True)
        if 'cell' in data and torch.is_tensor(data['cell']):
            data['cell'] = data['cell'].detach().clone()
        if self.use_cudagraphs:
            torch.compiler.cudagraph_mark_step_begin()
        with torch.enable_grad():
            out = self.model(
                data,
                compute_force=self.compute_force,
                compute_stress=self.compute_stress,
                training=False,
            )
        return {k: (v.detach().clone() if torch.is_tensor(v) else v)
                for k, v in out.items()}


def run_torch_inference(
    model: torch.nn.Module,
    batch: Batch,
    device: Any,
    *,
    repeats: int,
    warmup: int,
    compute_force: bool,
    compute_stress: bool,
) -> tuple[BenchmarkResult, dict[str, torch.Tensor]]:
    grad_ctx = torch.enable_grad if (compute_force or compute_stress) else torch.no_grad

    # Current mace (>= commit 38a21f8) expects a dict-like input (data.get(...));
    # the torch_geometric Batch has no .get, so feed the model a plain dict.
    if hasattr(batch, 'to_dict'):
        batch = batch.to_dict()

    for _ in range(warmup):
        with grad_ctx():
            model(batch, compute_force=compute_force, compute_stress=compute_stress)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    timings: list[float] = []
    outputs: dict[str, torch.Tensor] | None = None
    for _ in range(repeats):
        start = time.perf_counter()
        with grad_ctx():
            outputs = model(
                batch,
                compute_force=compute_force,
                compute_stress=compute_stress,
            )
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        timings.append(time.perf_counter() - start)

    assert outputs is not None
    arr = np.array(timings)
    stats = BenchmarkResult(
        mean=float(arr.mean()),
        std=float(arr.std()),
        min_time=float(arr.min()),
    )
    return stats, outputs


def run_jax_inference(
    graphdef,
    params,
    batch_jax: dict[str, jnp.ndarray],
    *,
    repeats: int,
    warmup: int,
    compute_force: bool,
    compute_stress: bool,
) -> tuple[BenchmarkResult, dict[str, Any]]:
    def _apply(p, data):
        outputs, _ = graphdef.apply(p)(
            data, compute_force=compute_force, compute_stress=compute_stress
        )
        return outputs

    apply_fn = jax.jit(_apply)

    for _ in range(warmup):
        outputs = apply_fn(params, batch_jax)
        jax.block_until_ready(outputs)  # block the full output tree (energy+forces+stress),
        # not just energy, so the backward (forces/stress) is included in the timing

    timings: list[float] = []
    outputs: dict[str, Any] | None = None
    for _ in range(repeats):
        start = time.perf_counter()
        outputs = apply_fn(params, batch_jax)
        jax.block_until_ready(outputs)  # block the full output tree (energy+forces+stress),
        # not just energy, so the backward (forces/stress) is included in the timing
        timings.append(time.perf_counter() - start)

    assert outputs is not None
    arr = np.array(timings)
    stats = BenchmarkResult(
        mean=float(arr.mean()),
        std=float(arr.std()),
        min_time=float(arr.min()),
    )
    return stats, outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Benchmark Torch vs JAX MACE inference.'
    )
    parser.add_argument(
        '--foundation', default='mp', help='Foundation source (mp, off, anicc, omol).'
    )
    parser.add_argument(
        '--variant', default='medium-mpa-0', help='Foundation model variant.'
    )
    parser.add_argument(
        '--symbol', default='Si', help='Element symbol for the benchmark crystal.'
    )
    parser.add_argument(
        '--repeat', type=int, default=2, help='Supercell repeat along each axis.'
    )
    parser.add_argument(
        '--repeats', type=int, default=10, help='Number of timed runs for each backend.'
    )
    parser.add_argument(
        '--warmup', type=int, default=3, help='Number of warmup runs before timing.'
    )
    parser.add_argument(
        '--disable-forces',
        action='store_true',
        help='Skip force computation in the benchmark.',
    )
    parser.add_argument(
        '--disable-stress',
        action='store_true',
        help='Skip stress computation in the benchmark.',
    )
    parser.add_argument(
        '--cue-conv-fusion',
        action='store_true',
        help='Enable cuequivariance conv fusion in the converted JAX model.',
    )
    parser.add_argument(
        '--cueq',
        action='store_true',
        help='Use cuequivariance kernels on BOTH backends (torch via run_e3nn_to_cueq, '
        'JAX via enabled CuEquivarianceConfig).',
    )
    parser.add_argument(
        '--torch-compile',
        default='none',
        choices=['none', 'default', 'reduce-overhead', 'max-autotune',
                 'max-autotune-no-cudagraphs'],
        help='Enable torch compile. Needs mace with the retain_graph fix so the '
        'inner force autograd.grad traces under AOTAutograd.',
    )
    parser.add_argument(
        '--tf32',
        action='store_true',
        help='Enable TF32-class matmuls on BOTH frameworks (faster, lower precision). '
        'Default pins full fp32 on both for apples-to-apples precision.',
    )
    args = parser.parse_args()

    compute_force = not args.disable_forces
    compute_stress = not args.disable_stress

    # Precision: keep both frameworks matched. Default pins full fp32 matmuls; --tf32
    # enables TF32-class matmuls on both (Ampere+ tensor cores, ~lower precision).
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
        jax.config.update('jax_default_matmul_precision', 'high')
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        jax.config.update('jax_default_matmul_precision', 'highest')

    torch_device = configure_torch_runtime(get_torch_device(), deterministic=False)
    print(f'Torch device: {torch_device}')
    print(f'JAX devices: {[f"{dev.platform}:{dev.id}" for dev in jax.devices()]}')
    print(f'Computing forces: {compute_force} | stresses: {compute_stress}')

    torch_model = load_foundation_model(args.foundation, args.variant, device='cpu')
    torch_model = torch_model.to(torch_device)

    atoms = build_example_atoms(args.symbol, args.repeat)
    batch_torch, batch_jax, config = prepare_batches(torch_model, atoms, torch_device)

    run_compiled = args.torch_compile != 'none'

    # Build the JAX graph from the ORIGINAL e3nn model first (the cue-jax modules are
    # built from it via cueq_config); only after that may we convert the torch model to
    # cueq in-place for the torch backends.
    cue_config: CuEquivarianceConfig | None = None
    if args.cueq:
        cue_config = CuEquivarianceConfig(
            enabled=True, optimize_all=True, conv_fusion=True,
            group='O3', layout='mul_ir',
        )
    elif args.cue_conv_fusion:
        cue_config = CuEquivarianceConfig(
            enabled=False, optimize_channelwise=True, conv_fusion=True,
            layout='mul_ir',
        )
    graphdef, state, _ = convert_model(torch_model, config, cueq_config=cue_config)
    jax_params = state_to_pure_dict(state)

    if args.cueq:
        from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq
        torch_model = run_e3nn_to_cueq(torch_model, device=torch_device).to(torch_device)

    torch_stats, torch_outputs = run_torch_inference(
        torch_model,
        batch_torch,
        torch_device,
        repeats=args.repeats,
        warmup=args.warmup,
        compute_force=compute_force,
        compute_stress=compute_stress,
    )
    print('Torch (eager) inference (per call):')
    print(
        f'  mean = {torch_stats.mean * 1e3:.2f} ms  std = {torch_stats.std * 1e3:.2f} ms  min = {torch_stats.min_time * 1e3:.2f} ms'
    )

    if run_compiled:
        wrapper = CompiledMaceInference(
            torch_model, args.torch_compile, torch_device,
            compute_force=compute_force, compute_stress=compute_stress,
            cueq=args.cueq,
        )
        for _ in range(max(args.warmup, 8)):  # warmup / compile / cudagraph record
            wrapper(batch_torch)
            if torch_device.type == 'cuda':
                torch.cuda.synchronize(torch_device)
        ctimes: list[float] = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            wrapper(batch_torch)
            if torch_device.type == 'cuda':
                torch.cuda.synchronize(torch_device)
            ctimes.append(time.perf_counter() - start)
        carr = np.array(ctimes)
        compiled_stats = BenchmarkResult(
            mean=float(carr.mean()), std=float(carr.std()), min_time=float(carr.min()),
        )
        print(f'Torch (compiled: {args.torch_compile}, cudagraphs={wrapper.use_cudagraphs}) inference (per call):')
        print(
            f'  mean = {compiled_stats.mean * 1e3:.2f} ms  std = {compiled_stats.std * 1e3:.2f} ms  min = {compiled_stats.min_time * 1e3:.2f} ms'
        )

    jax_stats, jax_outputs = run_jax_inference(
        graphdef,
        jax_params,
        batch_jax,
        repeats=args.repeats,
        warmup=args.warmup,
        compute_force=compute_force,
        compute_stress=compute_stress,
    )
    print('JAX (jitted) inference (per call):')
    print(
        f'  mean = {jax_stats.mean * 1e3:.2f} ms  std = {jax_stats.std * 1e3:.2f} ms  min = {jax_stats.min_time * 1e3:.2f} ms'
    )

    torch_energy = torch_outputs['energy'].detach().cpu().numpy()[0]
    jax_energy = float(np.asarray(jax_outputs['energy'])[0])
    print(f'Energy difference |Torch - JAX|: {abs(torch_energy - jax_energy):.6e} eV')

    if compute_force:
        torch_forces = torch_outputs.get('forces')
        jax_forces = jax_outputs.get('forces')
        if torch_forces is not None and jax_forces is not None:
            torch_forces_np = torch_forces.detach().cpu().numpy()
            jax_forces_np = np.asarray(jax_forces)
            diff = torch_forces_np - jax_forces_np
            print('Force difference:')
            print(
                f'  max |ΔF| = {np.abs(diff).max():.6e} eV/Å  '
                f'RMSE = {np.sqrt(np.mean(diff**2)):.6e} eV/Å'
            )
        else:
            print('Force outputs unavailable for comparison.')

    if compute_stress:
        torch_stress = torch_outputs.get('stress')
        jax_stress = jax_outputs.get('stress')
        if torch_stress is not None and jax_stress is not None:
            torch_stress_np = torch_stress.detach().cpu().numpy()
            jax_stress_np = np.asarray(jax_stress)
            diff = torch_stress_np - jax_stress_np
            print('Stress difference:')
            print(
                f'  max |Δσ| = {np.abs(diff).max():.6e} eV/Å³  '
                f'RMSE = {np.sqrt(np.mean(diff**2)):.6e} eV/Å³'
            )
        else:
            print('Stress outputs unavailable for comparison.')


if __name__ == '__main__':
    main()
