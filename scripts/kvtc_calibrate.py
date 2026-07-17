#!/usr/bin/env python3

import argparse
import random
import os
import math
import re
import torch
import pprint
import logging
import sys

from pathlib import Path
from datetime import datetime

sys.path.append("../python")
from sglang.srt.mem_cache.allocator import token
from sglang.srt.layers.rotary_embedding.factory import get_rope
from sglang.srt.utils.hf_transformers.common import get_rope_config
from sglang.srt.server_args import (
       ServerArgs,
       get_global_server_args,
       set_global_server_args_for_scheduler
    )
from transformers import AutoConfig
from collections import defaultdict
from enum import Enum, IntEnum

logger = logging.getLogger()
WORKER_DIR_PATTERN = re.compile(r"^tp_(\d+)_pp_(\d+)$")


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
            token_count = torch.concat([torch.load(p, map_locations="cpu") for p in paths]).shape[0]
            if token_count < 1000:
                ret[kv_cache_paths]["ignored"] += 1
                continue
            if token_count < 8000:
                ret[kv_cache_paths]["short"] += 1
            else:
                ret[kv_cache_paths]["long"] += 1

    return ret


def load_tensor(paths):
    # Each prefix is chunked in two dimensions: by layer and by tokens
    # To make sure we don't mix up any dimensions, we first stack all layers that belong
    # to one chunk and then we concatenate them by tokens
    chunk_count = len([p for p in paths if "layer_0" in str(p)])

    ret = None
    for chunk_id in range(chunk_count):
        layers = [p for p in paths if f"chunk_{chunk_id}" in str(p)]
        layers.sort(key=tensor_sorting_fn)

        layer_tensors = []
        all_layers_loaded = True
        for l in layers:
            try:
                layer_tensors.append(torch.load(l, map_locations="cpu"))
            except:
                logger.warning(f"Skipping {l} -- file corrupted")
                all_layers_loaded = False

        if not all_layers_loaded:
            continue

        chunk = torch.stack(layer_tensors)
        # Now chunk is 4d tensor [layer, token, head, h_dim]

        if ret == None:
            ret = chunk
        else:
            ret = torch.concat([ret, chunk], dim=1)

    # Make the output tensor token first
    ret = ret.transpose(0, 1)

    assert ret.dtype == torch.bfloat16

    logger.info(f"Loaded {ret.shape}")

    torch.cpu.synchronize()

    return ret, ret.shape[0]


def trim_sink_tokens(tensor):
    return tensor[128:-128]


def sample_tokens(tensor, sampling_budget):
    token_cnt = tensor.shape[0]
    ids = random.sample(list(range(token_cnt)), sampling_budget)
    ids.sort()

    ids = torch.Tensor(ids).to(device="cpu").int()
    logger.debug(f"Sample ids\n{ids}")

    ret = tensor[ids, :].to(dtype=torch.float32, copy=True)
    logger.info(f"Original tokens: {tensor.shape}. Sampled tokens: {ret.shape}")

    return ret


def transform_tensors(tensors):
    ret = torch.concat(tensors)
    token_count = ret.shape[0]
    ret = ret.view(token_count, -1)

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

    def __init__(self, input_dir_list, tp_pp_worker, sampling_policy):
        self.datasets = {}
        self.context_groups = {}
        self.total_tokens = {kv: 0 for kv in TensorFileManager.KV}
        self.sampling_policy = sampling_policy

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

        file_groups = list(sequences.values())

        ret = []
        counter = {b: 0 for b in TensorFileManager.Sequence}
        buckets = {b: 0 for b in TensorFileManager.Sequence}
        logger.info(f"Looking for {kv} tensors at {tensor_dir / tp_pp_worker}")
        for fg in file_groups:
            chunks = [f for f in fg if "layer_0" in str(f)]
            token_count = torch.concat([torch.load(p, map_locations="cpu") for p in chunks]).shape[0]
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
        sink_tokens = 256

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

    sampled_data = []

    for kv_cache_paths in tensor_manager.datasets_list:
        for paths in tensor_manager.datasets[kv_cache_paths][kv]:
            tensor, token_count = load_tensor(paths)
            assert token_count >= 1000

            sampling_budget = tensor_manager.get_token_budget(
                kv, kv_cache_paths, N, tensor
            )
            assert sampling_budget < token_count - 2 * 128

            if undo_rope:
                tensor = Rope.invert_rope(tensor)
            tensor = trim_sink_tokens(tensor)
            sampled_data.append(sample_tokens(tensor, sampling_budget))

    pprint.pp([d.shape for d in sampled_data])

    input_tensor = transform_tensors(sampled_data)

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


def run():
    parser = argparse.ArgumentParser(
        usage=(
            f"\n{os.path.basename(__file__)}"
            " -N <token_sample_count> --niter <the number of subspace iterations for svd_lowrank>"
            " -q <a slightly overestimated rank of svd matrix>"
            " -i <dump-parent-directory>"
            " -o <output_path> -m <model_path>\n"
        )
    )
    parser.add_argument(
        "-N",
        "--sample-tokens",
        action="append",
        required=True,
        help="The total number of tokens to sample from the calibration dataset (default=200,000)",
    )
    parser.add_argument(
        "--niter",
        required=True,
        help="Please refer to 'niter' in torch.svd_lowrank in pytorch documentation",
    )
    parser.add_argument(
        "-q",
        "--svd_dim",
        required=True,
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
        "--output-dir",
        required=True,
        help="Directory to save the compression matrix and the logs",
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
    output_dir = Path(args.output_dir)
    N_list = [int(n) for n in args.sample_tokens]
    N_list.sort()
    svd_iter = int(args.niter)
    svd_dim = int(args.svd_dim)
    log_level = args.log_level
    sampling_policy = TensorFileManager.SamplingPolicy[args.sampling_policy.upper()]

    Rope.load_model_config(args.model_dir)

    init_logger(
        output_dir,
        f"svd-q{svd_dim}_iter{svd_iter}_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}.log",
        log_level,
    )

    input_dir_list, workers = discover_dump_directories(input_dir)

    output_dict = {
        "keys": {worker: {"mu": None, "basis": None} for worker in workers},
        "values": {worker: {"mu": None, "basis": None} for worker in workers},
    }

    for N in N_list:
        logger.info(
            f"-------------------- N={N} q={svd_dim} iter={svd_iter} --------------------"
        )

        for worker in workers:
            tensor_manager = TensorFileManager(input_dir_list, worker, sampling_policy)
            for kv in TensorFileManager.KV:
                undo_rope = kv == TensorFileManager.KV.K
                try:
                    mu, U, S, V = SVD(
                        tensor_manager, svd_dim, svd_iter, kv, N, undo_rope
                    )
                    logger.info(f"{mu.shape=}\n{U.shape=}\n{S.shape=}\n{V.shape=}")
                    if kv == TensorFileManager.KV.K:
                        output_dict["keys"][worker]["basis"] = V
                        output_dict["keys"][worker]["mu"] = mu
                    elif kv == TensorFileManager.KV.V:
                        output_dict["values"][worker]["basis"] = V
                        output_dict["values"][worker]["mu"] = mu

                except RuntimeError as e:
                    logger.error(f"RuntimeError: {e}")
                    logger.error(f"Skip q={svd_dim} iter={svd_iter} prefix={kv}")
                    return

        output_dict_path = output_dir / f"svd_n_{N}_iter_{svd_iter}.pt"
        torch.save(output_dict, output_dict_path)


if __name__ == "__main__":
    run()
