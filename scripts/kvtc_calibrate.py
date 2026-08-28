#!/usr/bin/env python3

import argparse
import json
import random
import os
import math
import re
import torch
import pprint
import logging
import sys
import threading

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime, timezone

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
from collections import Counter, defaultdict
from enum import Enum, IntEnum

logger = logging.getLogger()
WORKER_DIR_PATTERN = re.compile(r"^tp_(\d+)_pp_(\d+)$")
KVTC_FILE_VERSION="v1-noquant"
TOKEN_SELECTION_FILE_VERSION = 1
SINK_TOKENS = 128
SVD_WORKERS = 4


class TokenSelectionError(ValueError):
    pass


class TokenSelectionStore:
    """Record or replay token positions selected from KV dump requests."""

    def __init__(
        self,
        *,
        mode,
        path,
        input_dir,
        model_dir,
        sample_tokens,
        sampling_policy,
        dump_directories,
        workers,
    ):
        if mode not in (None, "load", "save"):
            raise TokenSelectionError(f"Invalid token selection mode: {mode}")
        self.mode = mode
        self.path = path
        self.input_dir = input_dir.absolute()
        self.metadata = None
        self._lock = threading.Lock()
        self._selections = {}
        self._used_selections = set()

        if self.mode is not None:
            self.metadata = {
                "kvtc_version": KVTC_FILE_VERSION,
                "model_dir": str(model_dir.resolve()),
                "sample_tokens": sample_tokens,
                "sampling_policy": sampling_policy.name.lower(),
                "dump_directories": [
                    self._relative_path(path) for path in dump_directories
                ],
                "workers": workers,
                "sink_tokens_per_side": SINK_TOKENS,
            }
        if self.mode == "load":
            self._load()

    def _relative_path(self, path):
        try:
            return path.absolute().relative_to(self.input_dir).as_posix()
        except ValueError as error:
            raise TokenSelectionError(
                f"Selection source {path} is outside input directory {self.input_dir}"
            ) from error

    def _selection_key(self, worker, kv, dataset_path, request_id):
        return (
            worker,
            kv.name,
            self._relative_path(dataset_path),
            request_id,
        )

    def _load(self):
        try:
            with self.path.open(encoding="utf-8") as file:
                document = json.load(file)
        except FileNotFoundError as error:
            raise TokenSelectionError(
                f"Token selection file does not exist: {self.path}"
            ) from error
        except (OSError, json.JSONDecodeError) as error:
            raise TokenSelectionError(
                f"Cannot load token selection file {self.path}: {error}"
            ) from error

        if not isinstance(document, dict):
            raise TokenSelectionError(
                f"Token selection file {self.path} must contain a JSON object"
            )
        if document.get("version") != TOKEN_SELECTION_FILE_VERSION:
            raise TokenSelectionError(
                f"Unsupported token selection file version in {self.path}: "
                f"{document.get('version')!r}; expected {TOKEN_SELECTION_FILE_VERSION}"
            )

        saved_metadata = document.get("metadata")
        if not isinstance(saved_metadata, dict):
            raise TokenSelectionError(
                f"Token selection file {self.path} has no valid metadata object"
            )
        if saved_metadata != self.metadata:
            differing_fields = sorted(
                key
                for key in set(saved_metadata) | set(self.metadata)
                if saved_metadata.get(key) != self.metadata.get(key)
            )
            raise TokenSelectionError(
                f"Token selection file {self.path} is incompatible with this run; "
                f"different metadata fields: {', '.join(differing_fields)}"
            )

        selections = document.get("selections")
        if not isinstance(selections, list):
            raise TokenSelectionError(
                f"Token selection file {self.path} has no valid selections list"
            )

        for selection in selections:
            try:
                key = (
                    selection["worker"],
                    selection["kv"],
                    selection["dump_directory"],
                    selection["request_id"],
                )
            except (KeyError, TypeError) as error:
                raise TokenSelectionError(
                    f"Malformed selection entry in {self.path}: {selection!r}"
                ) from error
            if key in self._selections:
                raise TokenSelectionError(
                    f"Duplicate selection entry in {self.path}: {key}"
                )
            self._selections[key] = selection

        logger.info(
            "Loaded %d token selections from %s", len(self._selections), self.path
        )

    def select(
        self,
        *,
        worker,
        kv,
        dataset_path,
        request_id,
        source_paths,
        token_count,
        sampling_budget,
    ):
        if self.mode is None:
            return sorted(
                random.sample(
                    range(SINK_TOKENS, token_count - SINK_TOKENS), sampling_budget
                )
            )

        key = self._selection_key(worker, kv, dataset_path, request_id)
        relative_sources = [self._relative_path(path) for path in source_paths]

        if self.mode == "load":
            with self._lock:
                selection = self._selections.get(key)
                if selection is None:
                    raise TokenSelectionError(
                        "No saved token selection for "
                        f"worker={worker}, kv={kv.name}, dump={key[2]}, "
                        f"request={request_id}"
                    )
                if key in self._used_selections:
                    raise TokenSelectionError(
                        f"Token selection was requested more than once: {key}"
                    )
                self._used_selections.add(key)

            expected = {
                "source_files": relative_sources,
                "token_count": token_count,
                "sampling_budget": sampling_budget,
            }
            differing_fields = [
                field
                for field, value in expected.items()
                if selection.get(field) != value
            ]
            if differing_fields:
                raise TokenSelectionError(
                    f"Saved selection for {key} does not match the current dump; "
                    f"different fields: {', '.join(differing_fields)}"
                )

            indices = selection.get("selected_token_indices")
            if (
                not isinstance(indices, list)
                or len(indices) != sampling_budget
                or any(type(index) is not int for index in indices)
                or indices != sorted(set(indices))
                or any(
                    index < SINK_TOKENS or index >= token_count - SINK_TOKENS
                    for index in indices
                )
            ):
                raise TokenSelectionError(
                    f"Saved selection for {key} contains invalid token indices"
                )
            return indices

        indices = sorted(
            random.sample(
                range(SINK_TOKENS, token_count - SINK_TOKENS), sampling_budget
            )
        )
        if self.mode == "save":
            selection = {
                "worker": worker,
                "kv": kv.name,
                "dump_directory": key[2],
                "request_id": request_id,
                "source_files": relative_sources,
                "token_count": token_count,
                "sampling_budget": sampling_budget,
                "selected_token_indices": indices,
            }
            with self._lock:
                if key in self._selections:
                    raise TokenSelectionError(
                        f"Duplicate token selection generated for {key}"
                    )
                self._selections[key] = selection
        return indices

    def finalize(self):
        if self.mode == "load":
            unused = set(self._selections) - self._used_selections
            if unused:
                examples = sorted(unused)[:3]
                raise TokenSelectionError(
                    f"Token selection file {self.path} contains {len(unused)} "
                    f"unused selections, including: {examples}"
                )
            logger.info("Reused all token selections from %s", self.path)
            return

        if self.mode != "save":
            return

        document = {
            "version": TOKEN_SELECTION_FILE_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metadata": self.metadata,
            "selections": sorted(
                self._selections.values(),
                key=lambda selection: (
                    selection["worker"],
                    selection["kv"],
                    selection["dump_directory"],
                    selection["request_id"],
                ),
            ),
        }
        temporary_path = self.path.with_name(f".{self.path.name}.tmp-{os.getpid()}")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with temporary_path.open("w", encoding="utf-8") as file:
                json.dump(document, file, indent=2)
                file.write("\n")
            os.replace(temporary_path, self.path)
        except OSError as error:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise TokenSelectionError(
                f"Cannot save token selections to {self.path}: {error}"
            ) from error
        logger.info("Saved %d token selections to %s", len(self._selections), self.path)


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


def sample_tokens(tensor, token_indices):
    ids = torch.tensor(token_indices, device="cpu", dtype=torch.long)
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
        sink_tokens = 2 * SINK_TOKENS

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
    selection_store: TokenSelectionStore,
    worker: str,
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
                    kv, kv_cache_paths, N, tensor
                )
                if sampling_budget >= token_count - 2 * SINK_TOKENS:
                    raise ValueError(
                        f"sampling budget {sampling_budget} leaves no non-sink tokens "
                        f"in {token_count}-token request"
                    )

                if undo_rope:
                    tensor = Rope.invert_rope(tensor)
                request_id = paths[0].name.split(kv.value, 1)[0]
                token_indices = selection_store.select(
                    worker=worker,
                    kv=kv,
                    dataset_path=kv_cache_paths,
                    request_id=request_id,
                    source_paths=paths,
                    token_count=token_count,
                    sampling_budget=sampling_budget,
                )
                sampled_data.append(sample_tokens(tensor, token_indices))
            except TokenSelectionError:
                raise
            except (AssertionError, RuntimeError, ValueError) as error:
                logger.warning("Skipping %s: %s", paths[0], error)

    pprint.pp([d.shape for d in sampled_data])

    if not sampled_data:
        raise RuntimeError(f"No valid {kv} dump requests remain for SVD")

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


def run_svd_job(
    tensor_manager: TensorFileManager,
    selection_store: TokenSelectionStore,
    worker: str,
    kv: TensorFileManager.KV,
    svd_dim: int,
    svd_iter: int,
    N: int,
):
    mu, U, S, V = SVD(
        tensor_manager,
        selection_store,
        worker,
        svd_dim,
        svd_iter,
        kv,
        N,
        undo_rope=kv == TensorFileManager.KV.K,
    )
    logger.info(
        "%s/%s: mu=%s U=%s S=%s V=%s",
        worker,
        kv,
        mu.shape,
        U.shape,
        S.shape,
        V.shape,
    )

    # U and S are not part of the calibration output. Do not retain them in the
    # Future result because U can be large.
    return worker, kv, mu, V


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
            " -o <output_path> -m <model_path>"
            " [--save-selected-tokens <path> | --load-selected-tokens <path> |"
            " --selected-tokens-cache <path>]\n"
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
    selection_group = parser.add_mutually_exclusive_group()
    selection_group.add_argument(
        "--save-selected-tokens",
        metavar="PATH",
        help="Save the token positions selected during this run as JSON",
    )
    selection_group.add_argument(
        "--load-selected-tokens",
        metavar="PATH",
        help="Load and reuse token positions from a previous run",
    )
    selection_group.add_argument(
        "--selected-tokens-cache",
        metavar="PATH",
        help=(
            "Load token positions when PATH exists; otherwise select tokens and "
            "save them to PATH"
        ),
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    log_dir = Path(args.log_dir)
    N = args.sample_tokens
    svd_iter = int(args.niter)
    svd_dim = int(args.svd_dim)
    log_level = args.log_level
    sampling_policy = TensorFileManager.SamplingPolicy[args.sampling_policy.upper()]

    init_logger(
        log_dir,
        f"svd-q{svd_dim}_iter{svd_iter}_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}.log",
        log_level,
    )

    input_dir_list, workers = discover_dump_directories(input_dir)
    svd_workers = min(SVD_WORKERS, len(workers) * len(TensorFileManager.KV))

    selection_mode = None
    selection_path = None
    if args.save_selected_tokens:
        selection_mode = "save"
        selection_path = Path(args.save_selected_tokens)
    elif args.load_selected_tokens:
        selection_mode = "load"
        selection_path = Path(args.load_selected_tokens)
    elif args.selected_tokens_cache:
        selection_path = Path(args.selected_tokens_cache)
        selection_mode = "load" if selection_path.exists() else "save"
        logger.info(
            "Token selection cache %s; %s selections at %s",
            "exists" if selection_mode == "load" else "does not exist",
            "loading" if selection_mode == "load" else "saving new",
            selection_path,
        )
    if (
        selection_path is not None
        and selection_path.absolute() == output_path.absolute()
    ):
        parser.error("The token selection path must differ from --output")

    selection_store = TokenSelectionStore(
        mode=selection_mode,
        path=selection_path,
        input_dir=input_dir,
        model_dir=Path(args.model_dir),
        sample_tokens=N,
        sampling_policy=sampling_policy,
        dump_directories=input_dir_list,
        workers=workers,
    )

    available_cpus = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else (os.cpu_count() or 1)
    )
    torch.set_num_threads(available_cpus)

    Rope.load_model_config(args.model_dir)

    output_dict = {
        "version": KVTC_FILE_VERSION,
        "keys": {worker: {"mu": None, "basis": None} for worker in workers},
        "values": {worker: {"mu": None, "basis": None} for worker in workers},
    }

    logger.info(
        f"-------------------- model={args.model_dir} N={N} q={svd_dim} iter={svd_iter} --------------------"
    )
    logger.info(
        "Running up to %d SVD jobs concurrently with up to %d PyTorch intra-op threads",
        svd_workers,
        available_cpus,
    )

    tensor_managers = {
        worker: TensorFileManager(input_dir_list, worker, sampling_policy)
        for worker in workers
    }

    with ThreadPoolExecutor(
        max_workers=svd_workers, thread_name_prefix="svd"
    ) as executor:
        futures = [
            executor.submit(
                run_svd_job,
                tensor_managers[worker],
                selection_store,
                worker,
                kv,
                svd_dim,
                svd_iter,
                N,
            )
            for worker in workers
            for kv in TensorFileManager.KV
        ]

        try:
            for future in as_completed(futures):
                worker, kv, mu, basis = future.result()
                section = "keys" if kv == TensorFileManager.KV.K else "values"
                output_dict[section][worker]["mu"] = mu
                output_dict[section][worker]["basis"] = basis
        except Exception:
            for future in futures:
                future.cancel()
            logger.exception("SVD calibration failed")
            raise

    selection_store.finalize()
    torch.save(output_dict, output_path)


if __name__ == "__main__":
    run()
