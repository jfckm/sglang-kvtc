from __future__ import annotations

from dataclasses import dataclass

import torch


KVTC_QUANT_STORAGE_DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8,
    "int4": torch.int32,
}
KVTC_QUANT_PRECISION_BITS = {
    "float32": 32,
    "bfloat16": 16,
    "int8": 8,
    "int4": 4,
}
KVTC_QUANT_METADATA_DTYPE = torch.float16
KVTC_QUANTIZED_DTYPES = frozenset(("int8", "int4"))


@dataclass(frozen=True)
class KVTCQuantGroup:
    feature_start: int
    feature_end: int
    dtype_name: str
    payload_start: int
    payload_end: int
    metadata_index: int | None


@dataclass(frozen=True)
class KVTCQuantLayout:
    groups: tuple[KVTCQuantGroup, ...]
    feature_count: int
    payload_elements: dict[str, int]
    metadata_count: int


def quant_group_bits(group_size: int, dtype_name: str) -> int:
    """Return per-token storage, including one FP16 scale/offset pair."""
    bits = group_size * KVTC_QUANT_PRECISION_BITS[dtype_name]
    if dtype_name in KVTC_QUANTIZED_DTYPES:
        bits += 2 * KVTC_QUANT_METADATA_DTYPE.itemsize * 8
    return bits


def build_quant_layout(
    schema: object,
    *,
    page_size: int,
    basis_rank: int,
    matrix_name: str,
) -> KVTCQuantLayout:
    if not isinstance(schema, (list, tuple)) or not schema:
        raise ValueError(
            f"{matrix_name} KVTC quantization schema must be a non-empty list"
        )

    groups = []
    feature_offset = 0
    metadata_count = 0
    previous_precision = None
    payload_offsets = {name: 0 for name in KVTC_QUANT_STORAGE_DTYPES}
    for group_index, entry in enumerate(schema):
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise ValueError(
                f"{matrix_name} KVTC quantization entry {group_index} must be (group_size, dtype)"
            )

        group_size, dtype_name = entry
        if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size <= 0:
            raise ValueError(
                f"{matrix_name} KVTC quantization group {group_index} has invalid size {group_size!r}"
            )
        if dtype_name not in KVTC_QUANT_STORAGE_DTYPES:
            raise ValueError(
                f"{matrix_name} KVTC quantization group {group_index} has unsupported dtype {dtype_name!r}"
            )
        precision = KVTC_QUANT_PRECISION_BITS[dtype_name]
        if previous_precision is not None and precision > previous_precision:
            raise ValueError(
                f"{matrix_name} KVTC quantization schema must use non-increasing precision; "
                f"group {group_index} changes from {previous_precision} to {precision} bits"
            )
        previous_precision = precision

        if dtype_name == "int4":
            if group_size < 8 or group_size % 8 != 0:
                raise ValueError(
                    f"{matrix_name} KVTC int4 group {group_index} has size {group_size}; "
                    "packed int4 requires a group size of at least 8 and a multiple of 8"
                )
            payload_elements = page_size * group_size // 8
        else:
            payload_elements = page_size * group_size

        metadata_index = None
        if dtype_name in KVTC_QUANTIZED_DTYPES:
            metadata_index = metadata_count
            metadata_count += 1

        payload_start = payload_offsets[dtype_name]
        payload_end = payload_start + payload_elements
        groups.append(
            KVTCQuantGroup(
                feature_start=feature_offset,
                feature_end=feature_offset + group_size,
                dtype_name=dtype_name,
                payload_start=payload_start,
                payload_end=payload_end,
                metadata_index=metadata_index,
            )
        )
        feature_offset += group_size
        payload_offsets[dtype_name] = payload_end

    if feature_offset > basis_rank:
        raise ValueError(
            f"{matrix_name} KVTC quantization schema retains {feature_offset} features, "
            f"but basis rank is only {basis_rank}"
        )

    return KVTCQuantLayout(
        groups=tuple(groups),
        feature_count=feature_offset,
        payload_elements={name: count for name, count in payload_offsets.items() if count},
        metadata_count=metadata_count,
    )
