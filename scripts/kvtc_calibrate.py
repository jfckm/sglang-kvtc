#!/usr/bin/env python3

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "python"))

from scripts.kvtc_calibration_data import (  # noqa: E402
    KV,
    Rope,
    SamplingPolicy,
    discover_dump_directories,
    fit_pca,
    load_sample_pool,
    sample_pool,
    scan_dump_manifest,
)
from scripts.kvtc_calibration_quant import (  # noqa: E402
    KVTC_FILE_VERSION,
    assign_quantization,
    load_pca_artifact,
)

logger = logging.getLogger()


def init_logger(log_dir: Path, filename: str, log_level: str) -> None:
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
    log_file_path = log_dir / filename
    logging.basicConfig(
        level=levels[log_level],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file_path)],
    )
    logger.info("Logging to %s", log_file_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage=(
            f"\n{os.path.basename(__file__)}"
            " (-N <token_sample_count> --niter <svd iterations> -q <svd rank>"
            " | --reuse-pca <calibration_file>)"
            " -i <dump-parent-directory>"
            " -o <output_path> -m <model_path>\n"
        )
    )
    parser.add_argument("--kvtc-version", action="version", version=KVTC_FILE_VERSION)
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
        "--niter", type=int, help="torch.svd_lowrank subspace iterations"
    )
    parser.add_argument(
        "-q", "--svd_dim", type=int, help="torch.svd_lowrank approximation rank"
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        required=True,
        help="Parent directory containing calibration dump directories",
    )
    parser.add_argument("-o", "--output", required=True, help="Calibration file output")
    parser.add_argument("--log-dir", required=True, help="Calibration log directory")
    parser.add_argument(
        "-m", "--model-dir", required=True, help="Path to the target model directory"
    )
    parser.add_argument("-log", "--log-level", default="info")
    parser.add_argument(
        "-s",
        "--sampling-policy",
        default=SamplingPolicy.STRICT.value,
        choices=[policy.value for policy in SamplingPolicy],
    )
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.reuse_pca is None:
        if args.sample_tokens is None or args.niter is None or args.svd_dim is None:
            parser.error(
                "-N/--sample-tokens, --niter, and -q/--svd_dim are required "
                "for PCA fitting"
            )
        if args.sample_tokens <= 0:
            parser.error("PCA sample token count must be positive")
    elif any(
        value is not None for value in (args.sample_tokens, args.niter, args.svd_dim)
    ):
        parser.error("PCA fitting options cannot be combined with --reuse-pca")
    if args.dp_sample_tokens <= 0:
        parser.error("DP sample token count must be positive")
    if any(ratio <= 0 for ratio in args.compression_ratios):
        parser.error("Compression ratios must be positive integers")


def empty_output(workers: list[str]) -> dict:
    return {
        "version": KVTC_FILE_VERSION,
        "keys": {
            worker: {"mu": None, "basis": None, "quant": {}} for worker in workers
        },
        "values": {
            worker: {"mu": None, "basis": None, "quant": {}} for worker in workers
        },
    }


def run() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)

    input_dir = Path(args.input_dir)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log_dir = Path(args.log_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

    compression_ratios = list(dict.fromkeys(args.compression_ratios))
    sampling_policy = SamplingPolicy(args.sampling_policy)

    Rope.load_model_config(args.model_dir)
    init_logger(
        log_dir,
        f"KVTC_q{args.svd_dim}_iter{args.niter}_{timestamp}.log",
        args.log_level,
    )

    dump_dirs, workers = discover_dump_directories(input_dir)
    manifest = scan_dump_manifest(dump_dirs, workers)
    output = (
        load_pca_artifact(args.reuse_pca, workers)
        if args.reuse_pca is not None
        else empty_output(workers)
    )
    logger.info(
        "model=%s PCA={%s} DP_N=%s ratios=%s manifest_entries=%s",
        args.model_dir,
        (
            args.reuse_pca
            if args.reuse_pca is not None
            else f"N={args.sample_tokens} q={args.svd_dim} iter={args.niter}"
        ),
        args.dp_sample_tokens,
        compression_ratios,
        len(manifest),
    )

    for worker_index, worker in enumerate(workers):
        for kv_index, kv in enumerate(KV):
            logger.info(f"Calibrating {worker} - {kv}")

            matrix_name = kv.matrix_name
            sample_seed = args.seed + worker_index * 4 + kv_index * 2
            records = [
                tensor_set
                for tensor_set in manifest
                if tensor_set.worker == worker and tensor_set.kv is kv
            ]
            sample_pool_data = load_sample_pool(records, kv is KV.K)

            if args.reuse_pca is not None:
                mean = output[matrix_name][worker]["mu"]
                basis = output[matrix_name][worker]["basis"]
                logger.info(
                    "Reusing %s/%s PCA tensors mu=%s basis=%s",
                    matrix_name,
                    worker,
                    mean.shape,
                    basis.shape,
                )
                pca_samples = None
            else:
                pca_samples = sample_pool(
                    sample_pool_data,
                    args.sample_tokens,
                    sampling_policy,
                    sample_seed,
                    "PCA",
                )

            dp_samples = sample_pool(
                sample_pool_data,
                args.dp_sample_tokens,
                sampling_policy,
                sample_seed + 1,
                "DP quantization",
            )
            del sample_pool_data

            if pca_samples is not None:
                mean, basis = fit_pca(pca_samples, args.svd_dim, args.niter)
                del pca_samples
                output[matrix_name][worker] = {
                    "mu": mean,
                    "basis": basis,
                    "quant": {},
                }

            projected_dp = (dp_samples - mean) @ basis
            feature_count = dp_samples.shape[1]
            for compression_ratio in compression_ratios:
                output[matrix_name][worker]["quant"][str(compression_ratio)] = (
                    assign_quantization(
                        projected_dp,
                        feature_count,
                        compression_ratio,
                    )
                )
            del dp_samples, projected_dp
    torch.save(output, output_path)


if __name__ == "__main__":
    run()
