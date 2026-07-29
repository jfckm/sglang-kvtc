#!/usr/bin/env python3

import argparse
import random
import os
import math
import re
import torch
import logging
import sys

from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

sys.path.append("../python")
from sglang.srt.mem_cache.kvtc_quant import (
    build_quant_layout,
    quant_group_bits,
)
from sglang.srt.layers.rotary_embedding.factory import get_rope
from sglang.srt.utils.hf_transformers.common import get_rope_config
from sglang.srt.server_args import (
       ServerArgs,
       get_global_server_args,
       set_global_server_args_for_scheduler
    )
from transformers import AutoConfig
from collections import Counter, defaultdict
from enum import Enum, IntEnum

logger = logging.getLogger()
WORKER_DIR_PATTERN = re.compile(r"^tp_(\d+)_pp_(\d+)$")
KVTC_FILE_VERSION = "v2-quant"
QUANT_DTYPES = ("float32", "bfloat16", "int8", "int4")
QUANT_BLOCK_SIZES = (1, 16, 64, 256, 1024)
INT4_BLOCK_SIZES = (8, 16, 64, 256, 1024)
SINK_TOKENS = 256


class Rope(object):
    rotary_emb = None
    rotary_dim = None
    is_neox_style = None

    @classmethod
    def ensure_server_args(cls):
        try:
            get_global_server_args()
        except ValueError:
            set_global_server_args_for_scheduler(ServerArgs(model_path="DUMMY"))

    @classmethod
    def load_model_config(cls, model_path):
        logger.info(f"Loading config from {model_path}")
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=False)
        rope_theta, rope_scaling = get_rope_config(cfg)
        rope_theta = float(rope_theta)

        if rope_scaling and "mrope_section" in rope_scaling:
            raise ValueError(
                "Calibration does not support mRoPE models because KV dumps do not "
                "include the required multimodal position IDs."
            )

        head_dim = getattr(cfg, "head_dim", None)
        if head_dim is None:
            head_dim = cfg.hidden_size // cfg.num_attention_heads

        cls.ensure_server_args()
        cls.rotary_emb = get_rope(
            head_size=int(head_dim),
            rotary_dim=int(head_dim),
            max_position=int(getattr(cfg, "max_position_embeddings", 32768)),
            base=rope_theta,
            rope_scaling=rope_scaling,
            partial_rotary_factor=float(getattr(cfg, "partial_rotary_factor", 1.0)),
            is_neox_style=getattr(cfg, "rope_is_neox_style", True),
            dtype=torch.float32,
        )
        if not hasattr(cls.rotary_emb, "cos_sin_cache"):
            raise ValueError(
                "Calibration only supports Qwen3-compatible 1-D RoPE variants "
                "with a direct cos/sin cache."
            )

        cls.rotary_dim = cls.rotary_emb.rotary_dim
        cls.is_neox_style = cls.rotary_emb.is_neox_style
        logger.info(
            "model.rope theta=%s rotary_dim=%s neox_style=%s scaling=%s",
            rope_theta,
            cls.rotary_dim,
            cls.is_neox_style,
            rope_scaling,
        )

    @classmethod
    def invert_rope(cls, tensor):
        tokens, _, _, _ = tensor.shape
        rope_cache_len = 64 * 1024

        assert tokens <= rope_cache_len, f"{tokens=}, {rope_cache_len=}"
        assert cls.rotary_emb is not None, "RoPE model config was not loaded"

        cos_sin_cache = cls.rotary_emb.cos_sin_cache
        if tokens > cos_sin_cache.shape[0]:
            raise ValueError(
                f"Dump has {tokens} tokens, but the model RoPE cache has only "
                f"{cos_sin_cache.shape[0]} positions."
            )

        cos, sin = cos_sin_cache[:tokens].chunk(2, dim=-1)
        cos = cos[:, None, None, :].to(tensor.dtype)
        sin = sin[:, None, None, :].to(tensor.dtype)
        denominator = (cos.square() + sin.square()).clamp_min(
            torch.finfo(tensor.dtype).eps
        )

        rotated = tensor[..., : cls.rotary_dim]
        passthrough = tensor[..., cls.rotary_dim :]
        if cls.is_neox_style:
            first, second = torch.chunk(rotated, 2, dim=-1)
            inverted = torch.cat(
                (
                    (first * cos + second * sin) / denominator,
                    (second * cos - first * sin) / denominator,
                ),
                dim=-1,
            )
        else:
            first, second = rotated[..., ::2], rotated[..., 1::2]
            inverted = torch.stack(
                (
                    (first * cos + second * sin) / denominator,
                    (second * cos - first * sin) / denominator,
                ),
                dim=-1,
            ).flatten(-2)

        return torch.cat((inverted, passthrough), dim=-1)


def get_tensors_paths(tensor_dir, prefix):
    tensor_files = [x for x in tensor_dir.iterdir() if x.is_file()]
    tensor_files = [x for x in tensor_files if x.name.startswith(prefix)]
    file_groups = list(set([str(x).split(".part")[0] for x in tensor_files]))
    file_groups = [
        sorted([f for f in tensor_files if g in str(f)]) for g in file_groups
    ]

    logger.info(f"Detected tensor files:\n{file_groups}")

    return file_groups


def tensor_sorting_fn(f):
    name, chunk_id, layer_id = re.search(
        r"(.*)chunk_(\d+)-layer_(\d+)\.bin$", str(f)
    ).groups()
    return (name, int(layer_id), int(chunk_id))


def count_tokens(datasets_list):
    ret = {}

    for kv_cache_paths in datasets_list:
        ret[kv_cache_paths] = {"ignored": 0, "short": 0, "long": 0}
        for paths in datasets_list[kv_cache_paths]:
            token_count = torch.concat([torch.load(p, map_location="cpu") for p in paths]).shape[0]
            if token_count < 1000:
                ret[kv_cache_paths]["ignored"] += 1
                continue
            if token_count < 8000:
                ret[kv_cache_paths]["short"] += 1
            else:
                ret[kv_cache_paths]["long"] += 1

    return ret


def load_tensor(paths):
    """Load one request only when every chunk has the same complete layer set."""
    if not paths:
        logger.warning("Discarding empty dump request")
        return None, None

    request_path = paths[0].parent
    request_name = paths[0].name.split("-K-", 1)[0].split("-V-", 1)[0]
    chunks = defaultdict(dict)
    for path in paths:
        match = re.search(r"chunk_(\d+)-layer_(\d+)\.bin$", path.name)
        if match is None:
            logger.warning("Discarding %s: unrecognized dump filename %s", request_name, path)
            return None, None

        chunk_id, layer_id = map(int, match.groups())
        if layer_id in chunks[chunk_id]:
            logger.warning(
                "Discarding %s: duplicate layer %s in chunk %s at %s",
                request_name,
                layer_id,
                chunk_id,
                request_path,
            )
            return None, None
        chunks[chunk_id][layer_id] = path

    chunk_ids = sorted(chunks)
    if chunk_ids != list(range(len(chunk_ids))):
        logger.warning(
            "Discarding %s: non-contiguous chunk IDs %s at %s",
            request_name,
            chunk_ids,
            request_path,
        )
        return None, None

    expected_layer_ids = None
    chunk_tensors = []
    for chunk_id in chunk_ids:
        layer_ids = sorted(chunks[chunk_id])
        if expected_layer_ids is None:
            expected_layer_ids = layer_ids
        elif layer_ids != expected_layer_ids:
            logger.warning(
                "Discarding %s: chunk %s has layers %s, expected %s at %s",
                request_name,
                chunk_id,
                layer_ids,
                expected_layer_ids,
                request_path,
            )
            return None, None

        try:
            layer_tensors = [
                torch.load(chunks[chunk_id][layer_id], map_location="cpu")
                for layer_id in layer_ids
            ]
            chunk = torch.stack(layer_tensors)
        except Exception as error:
            logger.warning(
                "Discarding %s: cannot load chunk %s at %s: %s",
                request_name,
                chunk_id,
                request_path,
                error,
            )
            return None, None

        if chunk.ndim != 4:
            logger.warning(
                "Discarding %s: chunk %s has shape %s, expected [layer, token, head, head_dim]",
                request_name,
                chunk_id,
                tuple(chunk.shape),
            )
            return None, None
        if chunk_tensors and (
            chunk.shape[0] != chunk_tensors[0].shape[0]
            or chunk.shape[2:] != chunk_tensors[0].shape[2:]
            or chunk.dtype != chunk_tensors[0].dtype
        ):
            logger.warning(
                "Discarding %s: chunk %s shape/dtype %s/%s differs from %s/%s",
                request_name,
                chunk_id,
                tuple(chunk.shape),
                chunk.dtype,
                tuple(chunk_tensors[0].shape),
                chunk_tensors[0].dtype,
            )
            return None, None
        chunk_tensors.append(chunk)

    try:
        ret = torch.concat(chunk_tensors, dim=1).transpose(0, 1)
    except RuntimeError as error:
        logger.warning(
            "Discarding %s: cannot combine validated chunks at %s: %s",
            request_name,
            request_path,
            error,
        )
        return None, None
    if ret.dtype != torch.bfloat16:
        logger.warning(
            "Discarding %s: dtype %s, expected torch.bfloat16", request_name, ret.dtype
        )
        return None, None

    logger.info("Loaded %s from %s", tuple(ret.shape), request_name)

    torch.cpu.synchronize()

    return ret, ret.shape[0]


def trim_sink_tokens(tensor):
    return tensor[128:-128]


def sample_tokens(tensor, sampling_budget, rng):
    token_cnt = tensor.shape[0]
    ids = rng.sample(range(token_cnt), sampling_budget)
    ids.sort()

    ids = torch.Tensor(ids).to(device="cpu").int()
    logger.debug(f"Sample ids\n{ids}")

    ret = tensor[ids, :].to(dtype=torch.float32, copy=True)
    logger.info(f"Original tokens: {tensor.shape}. Sampled tokens: {ret.shape}")

    return ret


def transform_tensors(tensors):
    if not tensors:
        raise ValueError("No valid calibration tensors were loaded")

    feature_shapes = Counter(tensor.shape[1:] for tensor in tensors)
    feature_shape, _ = feature_shapes.most_common(1)[0]
    valid_tensors = [tensor for tensor in tensors if tensor.shape[1:] == feature_shape]
    discarded = len(tensors) - len(valid_tensors)
    if discarded:
        logger.warning(
            "Discarding %s sampled tensors with non-canonical feature shapes; "
            "using %s from %s tensors",
            discarded,
            feature_shape,
            len(valid_tensors),
        )
        logger.warning("Observed sampled feature shapes: %s", dict(feature_shapes))

    ret = torch.concat(valid_tensors, dim=0).flatten(start_dim=1)

    torch.cpu.synchronize()

    return ret


class TensorFileManager(object):
    class SamplingPolicy(Enum):
        STRICT = 1
        RELAXED = 2
        OPEN = 3

    class KV(str, Enum):
        K = "-K-"
        V = "-V-"

    class Sequence(IntEnum):
        IGNORE = 0
        SHORT = 1000
        LONG = 8000

        @classmethod
        def bucket(cls, length):
            if length < cls.SHORT:
                return cls.IGNORE
            elif length < cls.LONG:
                return cls.SHORT
            else:
                return cls.LONG

    def __init__(
        self,
        input_dir_list,
        tp_pp_worker,
        sampling_policy,
        dp_request_ids,
        partition,
        seed,
    ):
        self.datasets = {}
        self.context_groups = {}
        self.total_tokens = {kv: 0 for kv in TensorFileManager.KV}
        self.sampling_policy = sampling_policy
        self.dp_request_ids = dp_request_ids
        self.partition = partition
        self.rng = random.Random(seed)

        for dataset_path in input_dir_list:
            self.datasets[dataset_path] = {}
            self.context_groups[dataset_path] = {}
            for kv in TensorFileManager.KV:
                files, counter = self._get_tensors_paths(dataset_path, tp_pp_worker, kv)
                self.datasets[dataset_path][kv] = files
                self.context_groups[dataset_path][kv] = counter

        self.datasets_list = list(self.datasets.keys())

        self._log_detected_files()
        self._log_detected_files()

    def _get_tensors_paths(self, tensor_dir, tp_pp_worker, kv):
        """
        An example of tensor file name: fa712099aaa24207a1667853e4401503-K-chunk_0-layer_0.bin
        File format explained:
        fa712099aaa24207a1667853e4401503 - request id. All files from one sequence have the same request id
        -K- - file contains dumped keys. For values, it would be '-V-'
        chunk_0 - A sequence might be split into multiple chunks
        layer_0 - Each layer of kv cache is saved as a separate file. The number of layer is model specific

        To put together one full chunk, we need to take all its layers
        To put together one sequence, we to take all reconstructed chunks

        """
        tensor_files = [x for x in (tensor_dir / tp_pp_worker).iterdir() if x.is_file()]
        tensor_files = [x for x in tensor_files if kv in x.name]

        sequences = defaultdict(list)
        for tf in tensor_files:
            seq_prefix = tf.name.split(kv.value, 1)[0]
            sequences[seq_prefix].append(tf)

        for seq in sequences:
            sequences[seq].sort(key=tensor_sorting_fn)

        file_groups = [
            paths
            for sequence_id, paths in sequences.items()
            if (sequence_id in self.dp_request_ids[tensor_dir])
            == (self.partition == "dp")
        ]

        ret = []
        counter = {b: 0 for b in TensorFileManager.Sequence}
        buckets = {b: 0 for b in TensorFileManager.Sequence}
        logger.info(f"Looking for {kv} tensors at {tensor_dir / tp_pp_worker}")
        for fg in file_groups:
            chunks = [
                path
                for path in fg
                if re.search(r"chunk_\d+-layer_0\.bin$", path.name)
            ]
            if not chunks:
                logger.warning(
                    "Skipping request %s during discovery: no layer-0 dump files",
                    fg[0] if fg else "<empty>",
                )
                continue
            try:
                token_count = torch.concat(
                    [torch.load(path, map_location="cpu") for path in chunks]
                ).shape[0]
            except Exception as error:
                logger.warning(
                    "Skipping request %s during discovery: cannot load layer-0 dumps: %s",
                    fg[0] if fg else "<empty>",
                    error,
                )
                continue
            bucket = TensorFileManager.Sequence.bucket(token_count)
            match bucket:
                case TensorFileManager.Sequence.IGNORE:
                    counter[bucket] += 1
                    buckets[bucket] += 1
                    kv = (
                        TensorFileManager.KV.K
                        if TensorFileManager.KV.K in fg
                        else TensorFileManager.KV.V
                    )
                    logger.info(
                        f"Ignoring too short sequence {kv}. min_len=1000, ignored len={token_count}"
                    )
                case TensorFileManager.Sequence.SHORT:
                    counter[bucket] += 1
                    buckets[bucket] += 1
                    self.total_tokens[kv] += token_count
                    ret.append(fg)
                case TensorFileManager.Sequence.LONG:
                    counter[bucket] += 1
                    buckets[bucket] += 1
                    self.total_tokens[kv] += token_count
                    ret.append(fg)

        return ret, counter

    def _log_detected_files(self):
        logger.debug(f"Tensor files:")
        for ds in self.datasets_list:
            for kv in TensorFileManager.KV:
                for path_list in self.datasets[ds][kv]:
                    logger.debug(f"{path_list}")
            logger.debug(f"{self.context_groups[ds][kv]}")

    def get_token_budget(self, kv: KV, dataset_name, N, tensor):
        token_cnt = tensor.shape[0]
        sink_tokens = SINK_TOKENS

        if self.sampling_policy == TensorFileManager.SamplingPolicy.STRICT:
            assert token_cnt >= 1000
            # Divide the total sample count across every nonempty dataset/bucket.
            bucket = TensorFileManager.Sequence.bucket(token_cnt)

            assert bucket != TensorFileManager.Sequence.IGNORE, (
                f"Invalid sequence length {token_cnt}"
            )

            nonempty_groups = sum(
                self.context_groups[dataset][kv][bucket] > 0
                for dataset in self.datasets_list
                for bucket in (
                    TensorFileManager.Sequence.SHORT,
                    TensorFileManager.Sequence.LONG,
                )
            )
            token_budget = math.ceil(
                N
                / nonempty_groups
                / self.context_groups[dataset_name][kv][bucket]
            )

            if token_budget > token_cnt - sink_tokens:
                raise ValueError(
                    f"Too short sequence. Min length {token_budget + sink_tokens}, got {token_cnt}. "
                    "Generate longer KV cache dumps, set lower sample (-N) value, or "
                    "consider switching sampling policy to 'relaxed' (bear in mind it may hurt LLM accuracy)"
                )

            return token_budget

        elif self.sampling_policy == TensorFileManager.SamplingPolicy.RELAXED:
            assert token_cnt >= 1000
            # Equal amount of tokens will be sampled from each dataset, but ignore short vs long contexts

            dataset_token_budget = math.ceil(N / len(self.datasets_list))

            sequences_cnt = (
                self.context_groups[dataset_name][kv][TensorFileManager.Sequence.SHORT]
                + self.context_groups[dataset_name][kv][TensorFileManager.Sequence.LONG]
            )

            token_budget = math.ceil(dataset_token_budget / sequences_cnt)

            if token_budget > token_cnt - sink_tokens:
                raise ValueError(
                    f"Too short sequence. Min length {token_budget + sink_tokens}, got {token_cnt}. "
                    "Generate longer KV cache dumps, set lower sample (-N) value, or "
                    "consider switching sampling policy to 'open' (bear in mind it may hurt LLM accuracy)"
                )

            return token_budget
        elif self.sampling_policy == TensorFileManager.SamplingPolicy.OPEN:
            # Equal amount of tokens will be sampled from each sequence

            sequences_cnt = 0

            for dataset in self.context_groups:
                sequences_cnt += self.context_groups[dataset][kv][
                    TensorFileManager.Sequence.SHORT
                ]
                sequences_cnt += self.context_groups[dataset][kv][
                    TensorFileManager.Sequence.LONG
                ]

            token_budget = math.ceil(N / sequences_cnt)

            if token_budget > token_cnt - sink_tokens:
                raise ValueError(
                    f"Too short sequence. Min length {token_budget + sink_tokens}, got {token_cnt}. "
                    "Generate longer KV cache dumps, set lower sample (-N) value, or "
                    "consider switching sampling policy to 'open' (bear in mind it may hurt LLM accuracy)"
                )

            return token_budget

        else:
            assert False, f"Invalid sampling policy {self.sampling_policy}"


def reserve_dp_requests(input_dirs, worker, dp_sample_tokens, seed):
    """Reserve whole requests per dataset/length bucket using one canonical worker."""
    grouped = defaultdict(list)
    for dataset_path in input_dirs:
        worker_dir = dataset_path / worker
        sequences = defaultdict(list)
        for path in worker_dir.iterdir():
            if path.is_file() and TensorFileManager.KV.K.value in path.name:
                sequences[path.name.split(TensorFileManager.KV.K.value, 1)[0]].append(path)

        for sequence_id, paths in sequences.items():
            layer_zero = [
                path
                for path in paths
                if re.search(r"chunk_\d+-layer_0\.bin$", path.name)
            ]
            if not layer_zero:
                continue
            try:
                token_count = torch.concat(
                    [torch.load(path, map_location="cpu") for path in layer_zero]
                ).shape[0]
            except Exception as error:
                logger.warning(
                    "Skipping %s while reserving DP requests: %s", sequence_id, error
                )
                continue
            bucket = TensorFileManager.Sequence.bucket(token_count)
            if bucket != TensorFileManager.Sequence.IGNORE:
                grouped[(dataset_path, bucket)].append(
                    (sequence_id, token_count - SINK_TOKENS)
                )

    if not grouped:
        raise ValueError("No eligible requests are available for the DP holdout")

    target_per_group = math.ceil(dp_sample_tokens / len(grouped))
    reserved = {dataset_path: set() for dataset_path in input_dirs}
    for group_index, ((dataset_path, bucket), candidates) in enumerate(
        sorted(grouped.items(), key=lambda item: (str(item[0][0]), int(item[0][1])))
    ):
        rng = random.Random(seed + group_index)
        candidates.sort()
        rng.shuffle(candidates)
        capacity = 0
        for sequence_id, usable_tokens in candidates:
            reserved[dataset_path].add(sequence_id)
            capacity += usable_tokens
            if capacity >= target_per_group:
                break
        if capacity < target_per_group:
            logger.error(
                f"DP holdout group {dataset_path}/{bucket.name.lower()} has only "
                f"{capacity} usable tokens; {target_per_group} are required"
            )
        logger.info(
            "Reserved %s requests with %s usable tokens for DP in %s/%s",
            len(reserved[dataset_path]),
            capacity,
            dataset_path,
            bucket.name.lower(),
        )
    return reserved


def collect_sampled_data(
    tensor_manager: TensorFileManager,
    kv: TensorFileManager.KV,
    sample_count: int,
    undo_rope: bool,
    purpose: str,
):
    sampled_data = []
    collected_by_group = Counter()
    for dataset_path in tensor_manager.datasets_list:
        for paths in tensor_manager.datasets[dataset_path][kv]:
            tensor, token_count = load_tensor(paths)
            if tensor is None or token_count is None:
                continue
            if token_count < TensorFileManager.Sequence.SHORT:
                logger.warning(
                    "Skipping %s: only %s tokens after reconstruction",
                    paths[0],
                    token_count,
                )
                continue
            try:
                sampling_budget = tensor_manager.get_token_budget(
                    kv, dataset_path, sample_count, tensor
                )
                if sampling_budget > token_count - SINK_TOKENS:
                    raise ValueError(
                        f"sampling budget {sampling_budget} exceeds the non-sink tokens "
                        f"in {token_count}-token request"
                    )
                if undo_rope:
                    tensor = Rope.invert_rope(tensor)
                tensor = trim_sink_tokens(tensor)
                sampled_data.append(
                    sample_tokens(tensor, sampling_budget, tensor_manager.rng)
                )
                collected_by_group[
                    (dataset_path, TensorFileManager.Sequence.bucket(token_count))
                ] += sampling_budget
            except (AssertionError, RuntimeError, ValueError) as error:
                logger.warning("Skipping %s: %s", paths[0], error)

    if not sampled_data:
        raise RuntimeError(f"No valid {kv} dump requests remain for {purpose}")
    data = transform_tensors(sampled_data)
    if data.shape[0] < sample_count:
        raise RuntimeError(
            f"Collected only {data.shape[0]}/{sample_count} {kv} tokens for {purpose}"
        )
    if data.shape[0] > sample_count:
        data = data[:sample_count]
    for (dataset_path, bucket), collected in sorted(
        collected_by_group.items(), key=lambda item: (str(item[0][0]), int(item[0][1]))
    ):
        logger.info(
            "%s %s/%s sampled tokens from %s/%s",
            purpose,
            collected,
            sample_count,
            dataset_path,
            bucket.name.lower(),
        )
    logger.info("Collected %s/%s %s tokens for %s", data.shape[0], sample_count, kv, purpose)
    return data


def SVD(
    tensor_manager: TensorFileManager,
    svd_dim: int,
    svd_iter: int,
    kv: TensorFileManager.KV,
    N: int,
    undo_rope: bool,
):
    logger.info(
        f"START SVD for data at {tensor_manager.datasets_list}/{kv}* q={svd_dim} iter={svd_iter}"
    )

    if undo_rope:
        logger.info(f"Load tensors, undo rope and sample")
    else:
        logger.info(f"Load tensors and sample")

    input_tensor = collect_sampled_data(
        tensor_manager, kv, N, undo_rope, "PCA"
    )

    n, p = input_tensor.shape
    dtype = input_tensor.dtype
    logger.info(f"SVD input tensor shape (n={n})x(p={p}). Dtype {dtype}")

    per_feature_mean = input_tensor.mean(dim=0)

    logger.info(f"Calculate lowrank SVD for q={svd_dim} iter={svd_iter}")
    U, S, Vh = torch.svd_lowrank(
        input_tensor, q=svd_dim, niter=svd_iter, M=per_feature_mean
    )

    logger.info(f"DONE SVD for data at")

    return per_feature_mean, U, S, Vh


def simulate_quantization_error(values, dtype_name):
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
    return float((values - reconstructed).square().sum().item())


@dataclass(frozen=True)
class DPRecord:
    error: float
    cost: int
    previous: "DPRecord | None"
    group: tuple[int, str] | None


def _pareto_frontier(records):
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


def assign_quantization(projected_data, original_feature_count, compression_ratio):
    if not projected_data:
        raise ValueError("No held-out projections were provided for quantization")
    rank = projected_data[0].shape[1]
    if any(data.shape[1] != rank for data in projected_data):
        raise ValueError("Workers have inconsistent PCA ranks")

    budget = math.floor(16 * original_feature_count / compression_ratio)
    frontiers = [[{} for _ in QUANT_DTYPES] for _ in range(rank + 1)]
    error_cache = {}

    def block_error(start, size, dtype_name):
        key = (start, size, dtype_name)
        if key not in error_cache:
            error_cache[key] = sum(
                simulate_quantization_error(
                    data[:, start : start + size], dtype_name
                )
                for data in projected_data
            )
        return error_cache[key]

    for end in range(1, rank + 1):
        for dtype_index, dtype_name in enumerate(QUANT_DTYPES):
            block_sizes = INT4_BLOCK_SIZES if dtype_name == "int4" else QUANT_BLOCK_SIZES
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

    feature_energy = torch.zeros(rank, dtype=torch.float64)
    for data in projected_data:
        feature_energy += data.to(torch.float64).square().sum(dim=0)
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

    schema = []
    while best is not None:
        schema.append(best.group)
        best = best.previous
    schema.reverse()

    merged = []
    for size, dtype_name in schema:
        if merged and dtype_name in ("float32", "bfloat16") and merged[-1][1] == dtype_name:
            merged[-1] = (merged[-1][0] + size, dtype_name)
        else:
            merged.append((size, dtype_name))
    build_quant_layout(
        merged,
        page_size=1,
        basis_rank=rank,
        matrix_name="calibrated",
    )
    logger.info(
        "DP ratio=%sx budget=%s bits used=%s error=%s schema=%s",
        compression_ratio,
        budget,
        sum(quant_group_bits(size, dtype_name) for size, dtype_name in merged),
        best_total_error,
        merged,
    )
    return merged


def init_logger(log_dir, filename, log_level):
    levels = {
        "critical": logging.CRITICAL,
        "error": logging.ERROR,
        "warn": logging.WARNING,
        "warning": logging.WARNING,
        "info": logging.INFO,
        "debug": logging.DEBUG,
    }
    if log_level not in levels:
        raise ValueError(f"Invalid log level: {log_level}")

    log_dir.mkdir(parents=True, exist_ok=True)

    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file_path = log_dir / filename

    logging.basicConfig(
        level=levels[log_level],
        format=fmt,
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file_path)],
    )

    logger.info(f"Logging to {log_file_path}")


def discover_dump_directories(input_dir: Path) -> tuple[list[Path], list[str]]:
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    logger.info("Scanning %s for calibration dump directories", input_dir)
    dump_dirs = []
    worker_sets = {}
    for candidate in sorted(input_dir.iterdir()):
        if not candidate.is_dir():
            continue

        workers = sorted(
            (
                entry.name
                for entry in candidate.iterdir()
                if entry.is_dir() and WORKER_DIR_PATTERN.fullmatch(entry.name)
            ),
            key=lambda worker: tuple(map(int, WORKER_DIR_PATTERN.fullmatch(worker).groups())),
        )
        if not workers:
            logger.debug("Skipping %s: no tp_<X>_pp_<Y> worker directories", candidate)
            continue

        dump_dirs.append(candidate)
        worker_sets[candidate] = workers
        logger.info("Discovered dump directory %s with workers: %s", candidate, workers)

    if not dump_dirs:
        raise ValueError(
            f"No dump directories with tp_<X>_pp_<Y> workers found in {input_dir}"
        )

    workers = worker_sets[dump_dirs[0]]
    for dump_dir in dump_dirs[1:]:
        if worker_sets[dump_dir] != workers:
            raise ValueError(
                "Dump directories must have the same worker set. "
                f"Expected {workers} from {dump_dirs[0]}, but found "
                f"{worker_sets[dump_dir]} in {dump_dir}."
            )

    logger.info(
        "Using %d dump directories and %d workers: %s",
        len(dump_dirs),
        len(workers),
        workers,
    )
    return dump_dirs, workers


def load_pca_artifact(path: Path, workers: list[str]):
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

        output[matrix_name] = {"quant": {}}
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
            output[matrix_name][worker] = {"mu": mu, "basis": basis}

    logger.info(
        "Reusing PCA parameters from %s (source version=%s)",
        path,
        artifact.get("version", "<missing>"),
    )
    return output


def run():
    parser = argparse.ArgumentParser(
        usage=(
            f"\n{os.path.basename(__file__)}"
            " (-N <token_sample_count> --niter <svd iterations> -q <svd rank>"
            " | --reuse-pca <calibration_file>)"
            " -i <dump-parent-directory>"
            " -o <output_path> -m <model_path>\n"
        )
    )
    parser.add_argument(
        "--kvtc-version",
        action="version",
        version=KVTC_FILE_VERSION
    )
    parser.add_argument(
        "-N",
        "--sample-tokens",
        type=int,
        help="Number of PCA fitting tokens; required unless --reuse-pca is used",
    )
    parser.add_argument(
        "--reuse-pca",
        type=Path,
        help="Existing calibration file whose PCA means and bases are reused; skips PCA fitting",
    )
    parser.add_argument(
        "--dp-sample-tokens",
        type=int,
        default=32768,
        help="Number of tokens reserved from whole requests for DP quantization (default=32768)",
    )
    parser.add_argument(
        "--compression-ratios",
        type=int,
        nargs="+",
        required=True,
        help="Positive integer compression ratios for which quantization schemas are generated",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for deterministic request partitioning and token sampling (default=0)",
    )
    parser.add_argument(
        "--niter",
        type=int,
        help="Please refer to 'niter' in torch.svd_lowrank in pytorch documentation",
    )
    parser.add_argument(
        "-q",
        "--svd_dim",
        type=int,
        help="Please refer to 'q' in torch.svd_lowrank in pytorch documentation",
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        required=True,
        help="Parent directory containing calibration dump directories. Tokens are sampled across all discovered dump directories.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Calibration file output",
    )
    parser.add_argument(
        "--log-dir",
        required=True,
        help="Calibration file output",
    )
    parser.add_argument(
        "-m", "--model-dir", required=True, help="Path to the target model directory"
    )
    parser.add_argument("-log", "--log-level", required=False, default="info")
    parser.add_argument(
        "-s",
        "--sampling-policy",
        default="strict",
        choices=list(str(p.name).lower() for p in TensorFileManager.SamplingPolicy),
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    log_dir = Path(args.log_dir)
    N = args.sample_tokens
    dp_sample_tokens = args.dp_sample_tokens
    compression_ratios = list(dict.fromkeys(args.compression_ratios))
    svd_iter = args.niter
    svd_dim = args.svd_dim
    log_level = args.log_level
    sampling_policy = TensorFileManager.SamplingPolicy[args.sampling_policy.upper()]
    if args.reuse_pca is None:
        if N is None or svd_iter is None or svd_dim is None:
            parser.error(
                "-N/--sample-tokens, --niter, and -q/--svd_dim are required "
                "for PCA fitting"
            )
        if N <= 0:
            parser.error("PCA sample token count must be positive")
    elif N is not None or svd_iter is not None or svd_dim is not None:
        parser.error("PCA fitting options cannot be combined with --reuse-pca")
    if dp_sample_tokens <= 0:
        parser.error("DP sample token count must be positive")
    if any(ratio <= 0 for ratio in compression_ratios):
        parser.error("Compression ratios must be positive integers")

    Rope.load_model_config(args.model_dir)

    init_logger(
        log_dir,
        (
            f"dp-reuse_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}.log"
            if args.reuse_pca is not None
            else f"svd-q{svd_dim}_iter{svd_iter}_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}.log"
        ),
        log_level,
    )

    input_dir_list, workers = discover_dump_directories(input_dir)
    reused_output = (
        load_pca_artifact(args.reuse_pca, workers)
        if args.reuse_pca is not None
        else None
    )
    dp_request_ids = reserve_dp_requests(
        input_dir_list, workers[0], dp_sample_tokens, args.seed
    )

    if reused_output is not None:
        output_dict = reused_output
    else:
        output_dict = {
            "version": KVTC_FILE_VERSION,
            "keys": {
                "quant": {},
                **{worker: {"mu": None, "basis": None} for worker in workers},
            },
            "values": {
                "quant": {},
                **{worker: {"mu": None, "basis": None} for worker in workers},
            },
        }
    projected_dp = {kv: [] for kv in TensorFileManager.KV}
    original_feature_counts = {kv: None for kv in TensorFileManager.KV}

    logger.info(
        "-------------------- model=%s PCA=%s DP_N=%s ratios=%s --------------------",
        args.model_dir,
        (
            args.reuse_pca
            if args.reuse_pca is not None
            else f"N={N} q={svd_dim} iter={svd_iter}"
        ),
        dp_sample_tokens,
        compression_ratios,
    )

    for worker in workers:
        pca_tensor_manager = (
            None
            if args.reuse_pca is not None
            else TensorFileManager(
                input_dir_list,
                worker,
                sampling_policy,
                dp_request_ids,
                "pca",
                args.seed,
            )
        )
        dp_tensor_manager = TensorFileManager(
            input_dir_list,
            worker,
            sampling_policy,
            dp_request_ids,
            "dp",
            args.seed,
        )
        for kv in TensorFileManager.KV:
            undo_rope = kv == TensorFileManager.KV.K
            matrix_name = "keys" if kv == TensorFileManager.KV.K else "values"

            if args.reuse_pca is not None:
                mu = output_dict[matrix_name][worker]["mu"]
                V = output_dict[matrix_name][worker]["basis"]
                logger.info(
                    "Reusing %s/%s PCA tensors mu=%s basis=%s",
                    matrix_name,
                    worker,
                    mu.shape,
                    V.shape,
                )
            else:
                mu, U, S, V = SVD(
                    pca_tensor_manager, svd_dim, svd_iter, kv, N, undo_rope
                )
                logger.info(f"{mu.shape=}\n{U.shape=}\n{S.shape=}\n{V.shape=}")

            dp_data = collect_sampled_data(
                dp_tensor_manager,
                kv,
                dp_sample_tokens,
                undo_rope,
                "DP quantization",
            )

            projected_dp[kv].append((dp_data - mu) @ V)
            feature_count = dp_data.shape[1]

            if original_feature_counts[kv] not in (None, feature_count):
                raise RuntimeError(
                    f"Workers have inconsistent {kv} feature counts: "
                    f"{original_feature_counts[kv]} and {feature_count}"
                )
            original_feature_counts[kv] = feature_count

            output_dict[matrix_name][worker]["basis"] = V
            output_dict[matrix_name][worker]["mu"] = mu

    for kv in TensorFileManager.KV:
        matrix_name = "keys" if kv == TensorFileManager.KV.K else "values"
        for compression_ratio in compression_ratios:
            output_dict[matrix_name]["quant"][str(compression_ratio)] = assign_quantization(
                projected_dp[kv],
                original_feature_counts[kv],
                compression_ratio,
            )

    torch.save(output_dict, output_path)


if __name__ == "__main__":
    run()
