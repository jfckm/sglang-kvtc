from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm.auto import tqdm

from sglang.srt.mem_cache.kvtc_quant import build_quant_layout, quant_group_bits

logger = logging.getLogger(__name__)

KVTC_FILE_VERSION = "v3-worker-quant"
QUANT_DTYPES = ("float32", "bfloat16", "int8", "int4")
QUANT_BLOCK_SIZES = (1, 16, 64, 256, 1024)
INT4_BLOCK_SIZES = (8, 16, 64, 256, 1024)


def simulate_quantization_error(values: torch.Tensor, dtype_name: str) -> float:
    if dtype_name == "float32":
        return 0.0
    if dtype_name == "bfloat16":
        reconstructed = values.to(torch.bfloat16).to(torch.float32)
    else:
        qmin, qmax = (-128, 127) if dtype_name == "int8" else (0, 15)
        minimum = values.amin(dim=1, keepdim=True)
        maximum = values.amax(dim=1, keepdim=True)
        value_range = maximum - minimum
        scale = torch.where(
            value_range > 0,
            value_range / (qmax - qmin),
            torch.ones_like(value_range),
        )
        zero_point = qmin - torch.round(minimum / scale)
        quantized = torch.clamp(torch.round(values / scale + zero_point), qmin, qmax)
        reconstructed = (quantized - zero_point) * scale
        reconstructed = torch.where(value_range > 0, reconstructed, values)
    error_norm = torch.linalg.vector_norm(values - reconstructed)
    return float(error_norm.square().item())


@dataclass(frozen=True)
class DPRecord:
    error: float
    cost: int
    previous: "DPRecord | None"
    group: tuple[int, str] | None


def _pareto_frontier(records: list[DPRecord]) -> dict[int, DPRecord]:
    best_by_cost = {}
    for record in records:
        current = best_by_cost.get(record.cost)
        if current is None or record.error < current.error:
            best_by_cost[record.cost] = record
    frontier = {}
    best_error = math.inf
    for cost in sorted(best_by_cost):
        record = best_by_cost[cost]
        if record.error < best_error:
            frontier[cost] = record
            best_error = record.error
    return frontier


def assign_quantization(
    projected_data: torch.Tensor,
    original_feature_count: int,
    compression_ratio: int,
) -> list[tuple[int, str]]:
    if projected_data.ndim != 2 or projected_data.shape[0] == 0:
        raise ValueError("No held-out projections were provided for quantization")
    rank = projected_data.shape[1]

    budget = math.floor(16 * original_feature_count / compression_ratio)
    frontiers = [[{} for _ in QUANT_DTYPES] for _ in range(rank + 1)]
    error_cache = {}

    def block_error(start: int, size: int, dtype_name: str) -> float:
        key = (start, size, dtype_name)
        if key not in error_cache:
            error_cache[key] = simulate_quantization_error(
                projected_data[:, start : start + size], dtype_name
            )
        return error_cache[key]

    with tqdm(
        total=rank * len(QUANT_DTYPES),
        desc=f"DP {compression_ratio}x",
        unit="state",
        dynamic_ncols=True,
        disable=None,
    ) as progress:
        for end in range(1, rank + 1):
            for dtype_index, dtype_name in enumerate(QUANT_DTYPES):
                block_sizes = (
                    INT4_BLOCK_SIZES if dtype_name == "int4" else QUANT_BLOCK_SIZES
                )
                candidates = []
                for size in block_sizes:
                    start = end - size
                    group_cost = quant_group_bits(size, dtype_name)
                    if start < 0 or group_cost > budget:
                        continue
                    quant_error = block_error(start, size, dtype_name)
                    if start == 0:
                        candidates.append(
                            DPRecord(quant_error, group_cost, None, (size, dtype_name))
                        )
                        continue
                    for previous_dtype in range(dtype_index + 1):
                        for previous in frontiers[start][previous_dtype].values():
                            cost = previous.cost + group_cost
                            if cost <= budget:
                                candidates.append(
                                    DPRecord(
                                        previous.error + quant_error,
                                        cost,
                                        previous,
                                        (size, dtype_name),
                                    )
                                )
                frontiers[end][dtype_index] = _pareto_frontier(candidates)
                progress.update()

    feature_energy = projected_data.to(torch.float64).square().sum(dim=0)
    tail_error = torch.cat(
        (
            torch.flip(torch.cumsum(torch.flip(feature_energy, (0,)), dim=0), (0,)),
            torch.zeros(1, dtype=feature_energy.dtype),
        )
    )

    best = None
    best_total_error = math.inf
    for end in range(1, rank + 1):
        omitted_error = float(tail_error[end].item())
        for dtype_frontier in frontiers[end]:
            for record in dtype_frontier.values():
                total_error = record.error + omitted_error
                if total_error < best_total_error:
                    best = record
                    best_total_error = total_error
    if best is None:
        raise ValueError(
            f"Compression ratio {compression_ratio} has a {budget}-bit budget, "
            "which cannot fit a non-empty quantization schema"
        )

    source_norm = float(torch.linalg.vector_norm(projected_data).item())
    relative_error = math.sqrt(best_total_error) / source_norm if source_norm else 0.0

    schema = []
    while best is not None:
        schema.append(best.group)
        best = best.previous
    schema.reverse()
    merged = []
    for size, dtype_name in schema:
        if (
            merged
            and dtype_name in ("float32", "bfloat16")
            and merged[-1][1] == dtype_name
        ):
            merged[-1] = (merged[-1][0] + size, dtype_name)
        else:
            merged.append((size, dtype_name))
    build_quant_layout(merged, page_size=1, basis_rank=rank, matrix_name="calibrated")
    logger.info(
        "DP ratio=%sx budget=%s bits used=%s relative_error=%s schema=%s",
        compression_ratio,
        budget,
        sum(quant_group_bits(size, dtype_name) for size, dtype_name in merged),
        relative_error,
        merged,
    )
    return merged


def load_pca_artifact(path: Path, workers: list[str]) -> dict:
    if not path.is_file():
        raise ValueError(f"PCA calibration file does not exist: {path}")
    artifact = torch.load(path, map_location="cpu")
    if not isinstance(artifact, dict):
        raise ValueError(f"PCA calibration file must contain a dictionary: {path}")

    output = {"version": KVTC_FILE_VERSION}
    for matrix_name in ("keys", "values"):
        matrix_params = artifact.get(matrix_name)
        if not isinstance(matrix_params, dict):
            raise ValueError(f"PCA calibration file is missing {matrix_name!r}: {path}")
        artifact_workers = sorted(key for key in matrix_params if key != "quant")
        if artifact_workers != sorted(workers):
            raise ValueError(
                f"PCA calibration {matrix_name} workers {artifact_workers} do not match "
                f"dump workers {sorted(workers)}"
            )

        output[matrix_name] = {}
        for worker in workers:
            worker_params = matrix_params.get(worker)
            if not isinstance(worker_params, dict):
                raise ValueError(
                    f"PCA calibration file has invalid {matrix_name}/{worker} parameters"
                )
            mu = worker_params.get("mu")
            basis = worker_params.get("basis")
            if not isinstance(mu, torch.Tensor) or mu.ndim != 1:
                raise ValueError(
                    f"PCA calibration {matrix_name}/{worker}/mu must be a 1-D tensor"
                )
            if (
                not isinstance(basis, torch.Tensor)
                or basis.ndim != 2
                or basis.shape[0] != mu.shape[0]
                or basis.shape[1] == 0
            ):
                raise ValueError(
                    f"PCA calibration {matrix_name}/{worker}/basis must have shape "
                    f"({mu.shape[0]}, rank>0)"
                )
            output[matrix_name][worker] = {"mu": mu, "basis": basis, "quant": {}}

    logger.info(
        "Reusing PCA parameters from %s (source version=%s)",
        path,
        artifact.get("version", "<missing>"),
    )
    return output
