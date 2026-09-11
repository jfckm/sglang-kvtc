#!/usr/bin/env python3
"""Benchmark the production KVTC hybrid pool, without a server or model weights.

Run in the same Ascend/PyTorch environment used to launch SGLang. Setup and
validation are untimed. All measurements include Python and device work; no
model graph is captured. The default modes compare production configurations,
which can retain different PCA ranks, rather than isolated quantizer arithmetic.
"""

from __future__ import annotations

import argparse
import gc
import logging
import math
import re
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "python"))

SINK_TOKENS = 128
WORKER_PATTERN = re.compile(r"tp_(\d+)_pp_(\d+)")
DUMP_PATTERN = re.compile(r"(.+)-([KV])-chunk_(\d+)-layer_(\d+)\.bin")
MODE_NAMES = ("quant", "compressed", "baseline")
logger = logging.getLogger(__name__)


@dataclass
class Dump:
    name: str
    directory: Path
    token_count: int
    k_paths: tuple[Path, ...]
    v_paths: tuple[Path, ...]


@dataclass
class Mode:
    name: str
    page_bytes: int
    k_rank: int
    v_rank: int
    k_groups: int = 0
    v_groups: int = 0


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Example: python scripts/benchmark_kvtc.py --model-dir /models/Qwen3 "
            "--compression-matrix /data/kvtc.pt --dump-dir /data/kv-dumps "
            "--dump-name request-id --tokens 8192 --k-cr 8 --v-cr 8\n"
            "All three modes use hybrid routing with 128 sink tokens. Host pool "
            "construction uses the production 90/10 split (at least 10 GB total). "
            "Logical GiB/s counts original K+V bytes, not physical bus bandwidth."
        ),
    )
    parser.add_argument(
        "--model-dir", type=Path, required=True,
        help="Local model directory; only configuration/RoPE is loaded",
    )
    parser.add_argument(
        "--compression-matrix", type=Path,
        help="PCA/quantization calibration artifact; required unless only baseline is enabled",
    )
    parser.add_argument(
        "--dump-dir", type=Path, required=True,
        help="Dump dataset, worker directory, or parent containing dump datasets",
    )
    parser.add_argument(
        "--tp-worker-name", default="tp_0_pp_0",
        help="Single TP/PP worker whose dumps and calibration entries are used",
    )
    parser.add_argument(
        "--dump-name",
        help="Conversation name from discovery; otherwise select interactively if multiple exist",
    )
    parser.add_argument(
        "--tokens", type=int,
        help="Use the first N tokens (entire dump when omitted); round down to whole pages",
    )
    parser.add_argument(
        "--page-size", type=int, default=128,
        help="Tokens per page; must be a positive divisor of 128",
    )
    parser.add_argument(
        "--iterations", type=int, default=20,
        help="Measured iterations per direction, per enabled mode",
    )
    parser.add_argument(
        "--warmups", type=int, default=5,
        help="Untimed warmup iterations per direction, per enabled mode; zero is allowed",
    )
    descriptions = (
        "PCA plus quantization", "PCA without quantization", "uncompressed hybrid baseline"
    )
    for name, description in zip(MODE_NAMES, descriptions):
        parser.add_argument(
            f"--disable-{name}-benchmark", action="store_true",
            help=f"Skip the {description} benchmark",
        )
    for side in ("k", "v"):
        parser.add_argument(
            f"--{side}-cr", type=int,
            help=(f"{side.upper()} compression ratio; "
                  "infer only if the artifact has exactly one ratio"),
        )
    parser.add_argument(
        "--device", type=int, default=0,
        help="NPU device index (uses the selected device with no distributed process group)",
    )
    parser.add_argument(
        "--host-memory-gb", type=int,
        help=("Production host-pool size in decimal GB; "
              "automatically size for all modes when omitted, minimum 10 GB"),
    )
    return parser


def enabled_modes(args):
    return [name for name in MODE_NAMES if not getattr(args, f"disable_{name}_benchmark")]


def validate_args(parser, args):
    if not enabled_modes(args):
        parser.error("At least one benchmark must be enabled")
    if args.page_size <= 0 or SINK_TOKENS % args.page_size:
        parser.error("--page-size must be a positive divisor of 128")
    if args.iterations <= 0 or args.warmups < 0:
        parser.error("--iterations must be positive and --warmups must be nonnegative")
    if args.tokens is not None and args.tokens <= 0:
        parser.error("--tokens must be positive")
    if args.device < 0 or not WORKER_PATTERN.fullmatch(args.tp_worker_name):
        parser.error("Use a nonnegative --device and --tp-worker-name tp_<N>_pp_<N>")
    if any(ratio is not None and ratio <= 0 for ratio in (args.k_cr, args.v_cr)):
        parser.error("Compression ratios must be positive integers")
    if args.host_memory_gb is not None and args.host_memory_gb < 10:
        parser.error("The production hybrid allocation requires --host-memory-gb >= 10")
    if (
        any(name != "baseline" for name in enabled_modes(args))
        and args.compression_matrix is None
    ):
        parser.error("--compression-matrix is required for quant/compressed benchmarks")


def load_cpu(path):
    import torch

    return torch.load(path, map_location="cpu", weights_only=True)


def discover_dumps(root, worker, loader=load_cpu):
    """Use calibration's request/chunk/layer grouping without sampling filters."""
    if not root.is_dir():
        raise ValueError(f"Dump directory does not exist: {root}")
    if root.name == worker:
        directories = [root]
    elif (root / worker).is_dir():
        directories = [root / worker]
    else:
        directories = sorted(
            path / worker for path in root.iterdir() if (path / worker).is_dir()
        )
    if not directories:
        available = sorted({
            path.name for path in (root, *root.glob("*"), *root.glob("*/*"))
            if path.is_dir() and WORKER_PATTERN.fullmatch(path.name)
        })
        raise ValueError(
            f"Worker {worker} not found in {root}; "
            f"available workers: {', '.join(available) or '(none)'}"
        )

    dumps = []
    for directory in directories:
        requests = defaultdict(lambda: {"K": {}, "V": {}})
        for path in sorted(directory.iterdir()):
            match = DUMP_PATTERN.fullmatch(path.name)
            if path.is_file() and match:
                request, kv, chunk, layer = match.groups()
                requests[request][kv][(int(chunk), int(layer))] = path
        for name, sides in sorted(requests.items()):
            try:
                keys = sorted(sides["K"])
                if not keys or keys != sorted(sides["V"]):
                    raise ValueError("K/V chunk and layer sets differ or are missing")
                chunks = sorted({chunk for chunk, _ in keys})
                layers = sorted({layer for _, layer in keys})
                if (
                    chunks != list(range(len(chunks)))
                    or layers != list(range(len(layers)))
                ):
                    raise ValueError("chunk/layer IDs must be contiguous starting at zero")
                if len(keys) != len(chunks) * len(layers):
                    raise ValueError("incomplete layer set in a chunk")
                token_count = 0
                for chunk in chunks:
                    k = loader(sides["K"][(chunk, 0)])
                    v = loader(sides["V"][(chunk, 0)])
                    if len(k.shape) != 3 or k.shape != v.shape:
                        raise ValueError(
                            "layer-0 K/V must have matching [token, head, head_dim] shapes"
                        )
                    token_count += k.shape[0]
                dumps.append(Dump(
                    name, directory, token_count,
                    tuple(sides["K"][key] for key in keys),
                    tuple(sides["V"][key] for key in keys),
                ))
            except (ValueError, RuntimeError, OSError, EOFError) as error:
                logger.warning("Skipping %s/%s: %s", directory, name, error)
    counts = Counter(dump.name for dump in dumps)
    for dump in dumps:
        if counts[dump.name] > 1:
            dump.name = f"{dump.directory.parent.name}/{dump.name}"
    return dumps


def select_dump(dumps, name, *, interactive=None, input_fn=input):
    if not dumps:
        raise ValueError("No complete K/V dumps found for the selected worker")
    print("Available KV dumps:", flush=True)
    for index, dump in enumerate(dumps, 1):
        print(f"  {index:3d}. {dump.name}  ({dump.token_count:,} tokens)", flush=True)
    if name is not None:
        matches = [dump for dump in dumps if dump.name == name]
        if len(matches) != 1:
            raise ValueError(f"--dump-name {name!r} must match one of the names above")
        return matches[0]
    if len(dumps) == 1:
        return dumps[0]
    if interactive is None:
        interactive = sys.stdin.isatty()
    if not interactive:
        raise ValueError("Multiple dumps found; pass --dump-name when stdin is not interactive")
    while True:
        choice = input_fn("Select a dump by number or name: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(dumps):
            return dumps[int(choice) - 1]
        matches = [dump for dump in dumps if dump.name == choice]
        if len(matches) == 1:
            return matches[0]
        print("Please enter a number or name from the list.", flush=True)


def selected_token_count(total, requested, page_size):
    requested = total if requested is None else requested
    if requested > total:
        raise ValueError(f"Requested {requested} tokens, but the dump contains {total}")
    actual = requested // page_size * page_size
    if actual <= SINK_TOKENS:
        raise ValueError(
            f"Need at least {SINK_TOKENS + page_size} tokens after rounding "
            "(128 sink tokens plus one remaining page)"
        )
    return actual


def load_selected_dump(dump, tokens):
    from scripts.kvtc_calibration_data import load_tensor

    keys, k_count = load_tensor(dump.k_paths)
    values, v_count = load_tensor(dump.v_paths)
    if (
        keys is None or values is None
        or k_count != dump.token_count or v_count != k_count
        or keys.shape != values.shape or keys.dtype != values.dtype
    ):
        raise ValueError("Selected dump has incomplete or mismatched K/V tensors")
    # Unlike load_sample_pool(), preserve the sink and trailing tokens and leave
    # keys rotated: the production hybrid pool performs its own inverse RoPE.
    return keys[:tokens].contiguous(), values[:tokens].contiguous()


def resolve_ratio(params, requested, option):
    if requested is not None:
        return requested
    ratios = params.get("quant", {})
    if len(ratios) == 1:
        ratio = int(next(iter(ratios)))
        if ratio > 0:
            return ratio
    available = ', '.join(map(str, ratios)) or '(none; PCA-only artifact)'
    raise ValueError(f"Pass {option}; available artifact ratios: {available}")


def describe_modes(args, shape, dtype):
    import torch

    from sglang.srt.mem_cache.kvtc_quant import build_quant_layout, quant_group_bits

    _, layers, heads, head_dim = shape
    features = layers * heads * head_dim
    raw_page_bytes = 2 * features * args.page_size * dtype.itemsize
    names = enabled_modes(args)
    params = {}
    if any(name != "baseline" for name in names):
        artifact = load_cpu(args.compression_matrix)
        for side, option in (("keys", "k_cr"), ("values", "v_cr")):
            entry = artifact.get(side, {}).get(args.tp_worker_name)
            if entry is None:
                raise ValueError(f"Artifact is missing {side}/{args.tp_worker_name}")
            basis, mu = entry["basis"], entry["mu"]
            if basis.ndim != 2 or basis.shape[0] != features or mu.shape != (features,):
                raise ValueError(
                    f"Artifact {side} dimensions do not match {features} local KV features"
                )
            if basis.dtype != torch.float32 or mu.dtype != torch.float32:
                raise ValueError(
                    "Expected FP32 PCA means and bases, as produced by kvtc_calibrate.py"
                )
            ratio = resolve_ratio(
                entry, getattr(args, option), "--" + option.replace("_", "-")
            )
            setattr(args, option, ratio)
            params[side] = (entry, ratio)

    modes = []
    for name in names:
        if name == "baseline":
            modes.append(Mode(name, raw_page_bytes, features, features))
            continue
        ranks, groups, page_bytes = [], [], 0
        for side in ("keys", "values"):
            entry, ratio = params[side]
            if name == "compressed":
                rank = min(features // ratio, entry["basis"].shape[1])
                if rank == 0:
                    raise ValueError(f"{side} compression ratio {ratio} retains no features")
                ranks.append(rank)
                groups.append(0)
                page_bytes += args.page_size * rank * dtype.itemsize
            else:
                schema = entry.get("quant", {}).get(str(ratio))
                layout = build_quant_layout(
                    schema, page_size=args.page_size,
                    basis_rank=entry["basis"].shape[1], matrix_name=side,
                )
                ranks.append(layout.feature_count)
                groups.append(len(layout.groups))
                page_bytes += args.page_size * sum(
                    quant_group_bits(size, storage) for size, storage in schema
                ) // 8
                print(f"  quant {side} schema: {schema}", flush=True)
        modes.append(Mode(name, page_bytes, *ranks, *groups))
    return modes, raw_page_bytes


def required_host_gb(tokens, page_size, modes, raw_page_bytes):
    # Match the real constructor's integer decimal-GB 90/10 split. The remainder
    # pool must be larger than the entire device pool, even though it stores only
    # tokens after the sink. Reserve one page beyond that constructor check.
    remainder_bytes = max(mode.page_bytes for mode in modes) * (tokens // page_size + 1)
    sink_bytes = raw_page_bytes * (SINK_TOKENS // page_size + 1)
    size = max(10, math.ceil(max(remainder_bytes / 0.9, sink_bytes / 0.1) / 1e9))
    while int(size * 0.90) * 1e9 < remainder_bytes or int(size * 0.1) * 1e9 < sink_bytes:
        size += 1
    return size


def measure(operation, warmups, iterations, synchronize, clock=time.perf_counter_ns):
    for _ in range(warmups):
        synchronize()
        operation()
        synchronize()
    elapsed = []
    for _ in range(iterations):
        synchronize()
        started = clock()
        operation()
        synchronize()
        elapsed.append((clock() - started) / 1e6)
    return elapsed


def reload_all_layers(pool, request, layer_count):
    for layer in range(layer_count):
        request.layer_id = layer
        pool.load_to_device_per_layer(request)


def git_version():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True
        )
        return f"{commit} (worktree {'dirty' if status else 'clean'})"
    except (OSError, subprocess.CalledProcessError):
        return "unavailable (not a Git checkout)"


def check_reconstruction(device_pool, keys, values, page_size, mode):
    """Validate once, in bounded CPU chunks, after all timed operations."""
    import torch

    checks = []
    for name, buffer, original in (
        ("K", device_pool.k_buffer, keys), ("V", device_pool.v_buffer, values)
    ):
        # Ignore the dummy first page; compare [layer, token, head, head_dim].
        device_view = buffer[:, 1:].flatten(1, 2)
        expected = original.transpose(0, 1)
        error = energy = 0.0
        for start in range(0, len(original), page_size):
            actual = device_view[:, start : start + page_size].cpu()
            reference = expected[:, start : start + page_size]
            if not bool(torch.isfinite(actual).all()):
                raise ValueError(f"{mode} reload contains nonfinite {name} values")
            if (
                mode == "baseline" or start < SINK_TOKENS
            ) and not torch.equal(actual, reference):
                raise ValueError(
                    f"{mode} reload did not exactly restore {name} "
                    f"at tokens {start}:{start + page_size}"
                )
            difference = actual.float() - reference.float()
            error += difference.square().sum(dtype=torch.float64).item()
            energy += reference.float().square().sum(dtype=torch.float64).item()
        relative = math.sqrt(error / energy) if energy else (0.0 if error == 0 else math.inf)
        checks.append(f"{name} relative-L2={relative:.6g}")
    print(f"  {mode} validation: finite, sink exact; " + ", ".join(checks), flush=True)


def run_mode(args, mode, device_pool, keys, values, rotary_emb):
    import torch

    from sglang.srt.mem_cache.memory_pool_host import (
        KVTCHostMemoryRequest,
        NPUMHATokenToKVPoolHybrid,
    )

    tp_rank, pp_rank = map(int, WORKER_PATTERN.fullmatch(args.tp_worker_name).groups())
    baseline = mode.name == "baseline"
    host = NPUMHATokenToKVPoolHybrid(
        device_pool=device_pool,
        host_to_device_ratio=1,
        host_size=args.host_memory_gb,
        page_size=args.page_size,
        layout="page_first_direct",
        kvtc_params_path="" if baseline else str(args.compression_matrix),
        kvtc_k_compression_ratio=0 if baseline else args.k_cr,
        kvtc_v_compression_ratio=0 if baseline else args.v_cr,
        kvtc_quant_disable=mode.name == "compressed",
        kvtc_quant_debug=False,
        rotary_emb=rotary_emb,
        tp_rank=tp_rank,
        pp_rank=pp_rank,
    )
    sink, remainder = host.alloc(SINK_TOKENS, len(keys) - SINK_TOKENS)
    if sink is None or remainder is None:
        raise ValueError("Host pool allocation failed; increase --host-memory-gb")
    indices = torch.arange(args.page_size, args.page_size + len(keys), dtype=torch.int64)
    positions = torch.arange(SINK_TOKENS, len(keys), dtype=torch.int64, device="npu")
    request = KVTCHostMemoryRequest(
        device_memory_pool=device_pool,
        host_indices_compressed=remainder,
        device_indices_compressed=indices[SINK_TOKENS:],
        token_indices_compressed=positions,
        host_indices_sink=sink,
        device_indices_sink=indices[:SINK_TOKENS],
        io_backend="kernel_ascend",
        layer_id=None,
    )
    # Restore the original data once per mode. Offloads cannot alter this source;
    # reloads happen only after every offload, so lossy errors cannot accumulate.
    for buffer, original in ((device_pool.k_buffer, keys), (device_pool.v_buffer, values)):
        buffer[:, 1:].copy_(original.transpose(0, 1).reshape_as(buffer[:, 1:]))
    torch.npu.synchronize()
    print(f"Running {mode.name}: offload, then reload", flush=True)
    # Match the controller's separate write/load streams, including eager Python
    # wrappers. Synchronization at measurement boundaries waits for completion.
    write_stream, load_stream = torch.npu.Stream(), torch.npu.Stream()
    with torch.npu.stream(write_stream):
        offload = measure(
            lambda: host.backup_from_device_all_layer(request),
            args.warmups, args.iterations, torch.npu.synchronize,
        )
    # Erase the destination once outside timing: validation must prove that the
    # reload really wrote the data rather than finding the original input there.
    device_pool.k_buffer[:, 1:].fill_(float("nan"))
    device_pool.v_buffer[:, 1:].fill_(float("nan"))
    torch.npu.synchronize()
    with torch.npu.stream(load_stream):
        reload = measure(
            lambda: reload_all_layers(host, request, device_pool.layer_num),
            args.warmups, args.iterations, torch.npu.synchronize,
        )
    check_reconstruction(device_pool, keys, values, args.page_size, mode.name)
    return {"offload": offload, "reload": reload}


def print_results(results, tokens, logical_bytes):
    print("\nCompleted-operation wall time (setup, warmups and validation excluded)")
    print("Logical GiB/s uses original K+V bytes; speedup = baseline median / mode median.")
    print(
        f"{'Mode':<12} {'Direction':<8} {'Mean ms':>10} {'Median ms':>10} "
        f"{'Min ms':>10} {'P95 ms':>10} {'Tokens/s':>12} {'GiB/s':>10} {'Speedup':>9}"
    )
    for name, directions in results.items():
        for direction, samples in directions.items():
            median = statistics.median(samples)
            baseline = results.get("baseline", {}).get(direction)
            speedup = f"{statistics.median(baseline) / median:.3f}x" if baseline else "n/a"
            p95 = sorted(samples)[math.ceil(0.95 * len(samples)) - 1]
            seconds = median / 1000
            print(
                f"{name:<12} {direction:<8} {statistics.mean(samples):10.3f} "
                f"{median:10.3f} {min(samples):10.3f} {p95:10.3f} "
                f"{tokens / seconds:12.1f} {logical_bytes / 2**30 / seconds:10.3f} "
                f"{speedup:>9}"
            )


def run():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    # Deferred imports keep --help and CPU-only discovery/timing tests lightweight.
    import torch
    import torch_npu

    from scripts.kvtc_calibration_data import Rope
    from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMHATokenToKVPool

    if not torch.npu.is_available():
        raise ValueError("An available NPU and the SGLang Ascend dependencies are required")
    torch.npu.set_device(args.device)
    dump = select_dump(discover_dumps(args.dump_dir, args.tp_worker_name), args.dump_name)
    tokens = selected_token_count(dump.token_count, args.tokens, args.page_size)
    requested = dump.token_count if args.tokens is None else args.tokens
    print(f"\nSGLang commit: {git_version()}")
    print(f"Model: {args.model_dir}\nCompression artifact: {args.compression_matrix}")
    print(f"Dump: {dump.name}\nDump directory: {dump.directory}\nTP worker: {args.tp_worker_name}")
    print(
        f"Tokens: original={dump.token_count}, requested={requested}, actual={tokens}, "
        f"dropped by rounding={requested - tokens}"
    )
    print(
        f"Sink tokens: {SINK_TOKENS}; remaining tokens: {tokens - SINK_TOKENS}; "
        f"page size: {args.page_size}"
    )
    print(f"Iterations: {args.iterations}; warmups: {args.warmups} (each direction, each mode)")
    print(f"Enabled modes: {', '.join(enabled_modes(args))}")
    print(
        f"Device: npu:{args.device} ({torch.npu.get_device_name(args.device)}); "
        f"torch={torch.__version__}; torch_npu={torch_npu.__version__}", flush=True,
    )

    keys, values = load_selected_dump(dump, tokens)
    print(
        f"Selected tensor shape [token, layer, head, head_dim]: {tuple(keys.shape)}; "
        f"dtype={keys.dtype}", flush=True,
    )
    modes, raw_page_bytes = describe_modes(args, keys.shape, keys.dtype)
    required = required_host_gb(tokens, args.page_size, modes, raw_page_bytes)
    if args.host_memory_gb is None:
        args.host_memory_gb = required
    elif args.host_memory_gb < required:
        raise ValueError(
            f"This workload requires --host-memory-gb >= {required} "
            "with the production 90/10 split"
        )
    print(f"Host allocation parameter: {args.host_memory_gb} decimal GB (production 90/10 split)")
    print(f"K compression ratio: {args.k_cr}; V compression ratio: {args.v_cr}")
    for mode in modes:
        stored_bytes = (
            raw_page_bytes * (SINK_TOKENS // args.page_size)
            + mode.page_bytes * ((tokens - SINK_TOKENS) // args.page_size)
        )
        print(
            f"  {mode.name}: retained K/V rank={mode.k_rank}/{mode.v_rank}, "
            f"K/V groups={mode.k_groups}/{mode.v_groups}, "
            f"remaining-page bytes={mode.page_bytes}, stored workload bytes={stored_bytes}"
        )
    print(
        "Assumptions: BF16 MHA dump; positions start at 0; 128 sink tokens; "
        "no inference overlap or graph capture.", flush=True,
    )

    rotary_emb = None
    if any(mode.name != "baseline" for mode in modes):
        Rope.load_model_config(str(args.model_dir))
        rotary_emb = Rope.rotary_emb.to(device="npu")
        if rotary_emb.rotary_dim != keys.shape[-1] or not rotary_emb.is_neox_style:
            raise ValueError("The current hybrid pool requires full-head, NeoX-style RoPE")
        if tokens > rotary_emb.cos_sin_cache.shape[0]:
            raise ValueError("Selected tokens exceed the model's RoPE cache; reduce --tokens")
    _, layers, heads, head_dim = keys.shape
    with torch.inference_mode():
        device_pool = NPUMHATokenToKVPool(
            size=tokens, page_size=args.page_size, dtype=keys.dtype,
            head_num=heads, head_dim=head_dim, layer_num=layers, device="npu",
            enable_memory_saver=False, enable_alt_stream=False,
        )
        if not isinstance(device_pool.k_buffer, torch.Tensor) or device_pool.k_buffer.ndim != 5:
            raise ValueError("Benchmark requires the paged NPU layout; unset ASCEND_USE_FIA")
        results = {}
        for mode in modes:
            results[mode.name] = run_mode(args, mode, device_pool, keys, values, rotary_emb)
            gc.collect()
            torch.npu.empty_cache()
    print_results(results, tokens, raw_page_bytes * (tokens // args.page_size))


if __name__ == "__main__":
    try:
        run()
    except (ValueError, ImportError, EOFError, KeyboardInterrupt) as error:
        raise SystemExit(f"benchmark_kvtc: {error}") from error
