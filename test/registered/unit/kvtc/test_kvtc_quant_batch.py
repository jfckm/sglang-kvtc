"""CPU checks for batched KVTC packing, using a row-wise NPU operator stand-in.

Run: python -m unittest discover -s test/registered/unit/kvtc -p test_kvtc_quant_batch.py
The production class is loaded without its accelerator-only module imports.
Actual Ascend kernel numerics and transfer timings require an NPU run.
"""

import ast
import contextlib
import runpy
import unittest
from collections import defaultdict
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[4]
CACHE = ROOT / "python/sglang/srt/mem_cache"
QUANT = runpy.run_path(str(CACHE / "kvtc_quant.py"))


def dynamic_quant(x, *, dst_type):
    """Independent per-row quantization, including eight INT4 values per int32."""
    x = x.float()
    int4 = dst_type == torch.quint4x2
    low, high = (-8, 7) if int4 else (-128, 127)
    minimum, maximum = x.amin(dim=-1), x.amax(dim=-1)
    scale = ((maximum - minimum) / (high - low)).clamp_min(1e-8)
    offset = low - minimum / scale
    q = (x / scale[:, None] + offset[:, None]).round().clamp(low, high)
    if int4:
        nibbles = (q.to(torch.int64) & 15).reshape(x.shape[0], -1, 8)
        q = (nibbles << (torch.arange(8) * 4)).sum(dim=-1).to(torch.int32)
    else:
        q = q.to(torch.int8)
    return q, scale, offset


def anti_quant(q, scale, *, offset, dst_dtype, **kwargs):
    if q.dtype == torch.int32:
        nibbles = (q.to(torch.int64).unsqueeze(-1) >> (torch.arange(8) * 4)) & 15
        q = torch.where(nibbles >= 8, nibbles - 16, nibbles).reshape(1, -1)
    return ((q.float() + offset) * scale).to(dst_dtype)


def load_pool():
    path = CACHE / "memory_pool_host.py"
    tree = ast.parse(path.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "NPUMHATokenToKVPoolCompressed"
    )
    cls.bases = []
    module = ast.parse("from __future__ import annotations")
    module.body.append(cls)
    namespace = {
        **QUANT,
        "defaultdict": defaultdict,
        "torch": torch,
        "torch_npu": SimpleNamespace(
            npu_dynamic_quant_asymmetric=Mock(side_effect=dynamic_quant),
            npu_anti_quant=Mock(side_effect=anti_quant),
        ),
    }
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[cls.name], namespace["torch_npu"]


Pool, NPU = load_pool()


def buffers(pool, layout, page_count=8):
    payloads = {
        name: torch.full((page_count, size), -1, dtype=pool._QUANT_STORAGE_DTYPES[name])
        for name, size in layout.payload_elements.items()
    }
    shape = (page_count, pool.page_size, layout.metadata_count)
    return (
        payloads,
        torch.full(shape, -1, dtype=torch.float16),
        torch.full(shape, -1, dtype=torch.float16),
    )


def reference_pages(pool, source, host_pages, layout, destination):
    """Write independent pages directly into the decoder's prescribed slices."""
    payloads, scales, offsets = destination
    for page, host_page in zip(source, host_pages):
        for group in layout.groups:
            values = page[:, group.feature_start : group.feature_end]
            if group.metadata_index is None:
                q = values.to(pool._QUANT_STORAGE_DTYPES[group.dtype_name])
            else:
                dst_type = torch.quint4x2 if group.dtype_name == "int4" else torch.int8
                q, scale, offset = dynamic_quant(
                    values.to(pool.dtype), dst_type=dst_type
                )
                scales[host_page, :, group.metadata_index] = scale
                offsets[host_page, :, group.metadata_index] = -offset
            payloads[group.dtype_name][
                host_page, group.payload_start : group.payload_end
            ] = q.flatten()


class TestKVTCQuantBatch(unittest.TestCase):
    def make_pool(self, dtype=torch.bfloat16):
        pool = Pool.__new__(Pool)
        pool.page_size = 2
        pool.dtype = dtype
        pool.device = "cpu"
        pool.device_pool = SimpleNamespace(device="cpu")
        pool._log_quant_activity = Mock()
        pool._log_quant_page_diagnostics = Mock()
        return pool

    def assert_buffers_equal(self, actual, expected):
        for name in expected[0]:
            torch.testing.assert_close(
                actual[0][name], expected[0][name], rtol=0, atol=0
            )
        for a, e in zip(actual[1:], expected[1:]):
            torch.testing.assert_close(a, e, rtol=0, atol=0)

    def test_payload_metadata_and_existing_decoder(self):
        schemas = (
            [
                (3, "float32"), (8, "int4"), (5, "int8"), (2, "bfloat16"),
                (16, "int4"), (7, "int8"), (1, "float32"),
            ],
            [(3, "float32"), (2, "bfloat16"), (1, "float32")],
        )
        for schema, dtype, count, profile in product(
            schemas, (torch.float16, torch.bfloat16), (0, 1, 3), (False, True)
        ):
            with self.subTest(schema=schema, dtype=dtype, pages=count, profile=profile):
                layout = QUANT["build_quant_layout"](
                    schema, page_size=2, basis_rank=64, matrix_name="test"
                )
                pool = self.make_pool(dtype)
                pool._profile_kvtc = profile
                source = torch.randn(
                    count, 2, layout.feature_count,
                    generator=torch.Generator().manual_seed(42),
                )
                host_pages = torch.tensor([5, 1, 3][:count], dtype=torch.int64)
                actual, expected = buffers(pool, layout), buffers(pool, layout)
                reference_pages(pool, source, host_pages, layout, expected)
                NPU.npu_dynamic_quant_asymmetric.reset_mock()
                with patch.object(
                    torch.profiler, "record_function",
                    side_effect=lambda _: contextlib.nullcontext(),
                ) as record:
                    pool._quantize_pages(source, host_pages, layout, *actual)
                if not profile or not count:
                    record.assert_not_called()
                self.assert_buffers_equal(actual, expected)
                self.assertEqual(
                    NPU.npu_dynamic_quant_asymmetric.call_count,
                    layout.metadata_count if count else 0,
                )
                for call in NPU.npu_dynamic_quant_asymmetric.call_args_list:
                    self.assertEqual(call.args[0].shape[0], count * pool.page_size)
                # Exercise the unchanged decoder, including later INT4 groups.
                for host_page in host_pages:
                    torch.testing.assert_close(
                        pool._dequantize_page(host_page, layout, *actual),
                        pool._dequantize_page(host_page, layout, *expected),
                        rtol=0, atol=0,
                    )

    def test_single_page_entry_point(self):
        pool = self.make_pool()
        layout = QUANT["build_quant_layout"](
            [(8, "int4")], page_size=2, basis_rank=8, matrix_name="test"
        )
        # Reconstruction tools can supply a full basis beyond the retained rank.
        source = torch.randn(1, 2, 12)
        actual, expected = buffers(pool, layout), buffers(pool, layout)
        reference_pages(pool, source, [3], layout, expected)
        pool._quantize_page(source[0], 3, layout, *actual)
        self.assert_buffers_equal(actual, expected)

    def test_dequantize_batch_matches_individual_pages(self):
        pool = self.make_pool()
        layout = QUANT["build_quant_layout"](
            [
                (3, "float32"),
                (8, "int4"),
                (5, "int8"),
                (16, "int4"),
                (2, "bfloat16"),
            ],
            page_size=pool.page_size,
            basis_rank=34,
            matrix_name="test",
        )
        source = torch.randn(3, pool.page_size, layout.feature_count)
        host_pages = torch.tensor([5, 1, 3], dtype=torch.int64)
        stored = buffers(pool, layout)
        pool._quantize_pages(source, host_pages, layout, *stored)

        expected = torch.stack(
            [pool._dequantize_page(page, layout, *stored) for page in host_pages]
        )
        NPU.npu_anti_quant.reset_mock()
        pool._profile_kvtc = True
        with patch.object(
            torch.profiler,
            "record_function",
            side_effect=lambda _: contextlib.nullcontext(),
        ) as record:
            actual = pool._dequantize_pages(host_pages, layout, *stored)

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        quantized_dtype_names = list(
            dict.fromkeys(
                group.dtype_name
                for group in layout.groups
                if group.metadata_index is not None
            )
        )
        self.assertEqual(
            NPU.npu_anti_quant.call_count, len(quantized_dtype_names)
        )
        for call, dtype_name in zip(
            NPU.npu_anti_quant.call_args_list, quantized_dtype_names
        ):
            dtype_feature_count = sum(
                group.feature_end - group.feature_start
                for group in layout.groups
                if group.dtype_name == dtype_name
            )
            self.assertEqual(
                call.args[1].numel(),
                len(host_pages)
                * pool.page_size
                * dtype_feature_count,
            )
        labels = [call.args[0] for call in record.call_args_list]
        self.assertEqual(labels.count("kvtc/dequant/host_gather"), 1)
        self.assertEqual(labels.count("kvtc/dequant/payload_metadata_h2d"), 1)
        self.assertEqual(
            labels.count("kvtc/dequant/group_payload_pack"),
            len(quantized_dtype_names),
        )
        self.assertEqual(
            labels.count("kvtc/dequant/expand_metadata"),
            len(quantized_dtype_names),
        )
        self.assertEqual(
            labels.count("kvtc/dequant/anti_quant"),
            len(quantized_dtype_names),
        )

    def test_reload_batches_quantized_pages(self):
        pool = self.make_pool(torch.float32)
        pool._QUANT_BATCH_MAX_PAGES = 2
        pool._profile_kvtc = True
        pool.k_kvtc = pool.v_kvtc = True
        pool.kvtc_quant_disable = False
        pool.layer_num = 2
        pool.head_num = 1
        pool.head_dim = 2
        pool.device_page_shape = (2, pool.page_size, 1, 2)
        generator = torch.Generator().manual_seed(17)
        pool.kvtc_k_mu = torch.randn(4, generator=generator)
        pool.kvtc_v_mu = torch.randn(4, generator=generator)
        pool.kvtc_k_V = torch.randn(4, 4, generator=generator)
        pool.kvtc_v_V = torch.randn(4, 4, generator=generator)
        layout = QUANT["build_quant_layout"](
            [(4, "int8")],
            page_size=pool.page_size,
            basis_rank=4,
            matrix_name="test",
        )
        pool.k_quant_layout = pool.v_quant_layout = layout
        k_source = torch.randn(3, pool.page_size, 4)
        v_source = torch.randn(3, pool.page_size, 4)
        host_pages = torch.tensor([5, 1, 3], dtype=torch.int64)
        device_pages = torch.tensor([2, 4, 0], dtype=torch.int64)
        (
            pool.k_quant_buffers,
            pool.k_quant_scales,
            pool.k_quant_offsets,
        ) = buffers(pool, layout)
        (
            pool.v_quant_buffers,
            pool.v_quant_scales,
            pool.v_quant_offsets,
        ) = buffers(pool, layout)
        pool._quantize_pages(
            k_source,
            host_pages,
            layout,
            pool.k_quant_buffers,
            pool.k_quant_scales,
            pool.k_quant_offsets,
        )
        pool._quantize_pages(
            v_source,
            host_pages,
            layout,
            pool.v_quant_buffers,
            pool.v_quant_scales,
            pool.v_quant_offsets,
        )
        restored_k = pool._dequantize_pages(
            host_pages,
            layout,
            pool.k_quant_buffers,
            pool.k_quant_scales,
            pool.k_quant_offsets,
        )
        restored_v = pool._dequantize_pages(
            host_pages,
            layout,
            pool.v_quant_buffers,
            pool.v_quant_scales,
            pool.v_quant_offsets,
        )
        positions = torch.arange(10, 10 + 3 * pool.page_size)

        def forward_rope(page_positions, values):
            return values + page_positions[:, None, None, None]

        pool.rotary_emb = SimpleNamespace(
            forward_native_keys_batch=Mock(side_effect=forward_rope)
        )
        device = SimpleNamespace(
            device="cpu",
            k_buffer=torch.full((2, 6, 2, 1, 2), -1.0),
            v_buffer=torch.full((2, 6, 2, 1, 2), -1.0),
        )
        expected_k = device.k_buffer.clone()
        expected_v = device.v_buffer.clone()
        for page, device_page in enumerate(device_pages):
            page_positions = positions[
                page * pool.page_size : (page + 1) * pool.page_size
            ]
            reconstructed_k = (
                restored_k[page] @ pool.kvtc_k_V.T + pool.kvtc_k_mu
            )
            reconstructed_v = (
                restored_v[page] @ pool.kvtc_v_V.T + pool.kvtc_v_mu
            )
            expected_k[:, device_page] = forward_rope(
                page_positions,
                reconstructed_k.reshape(pool.page_size, 2, 1, 2),
            ).transpose(1, 0)
            expected_v[:, device_page] = (
                reconstructed_v
                .reshape(pool.page_size, pool.layer_num, -1)
                .transpose(1, 0)
                .reshape(*pool.device_page_shape)
            )

        NPU.npu_anti_quant.reset_mock()
        with patch.object(
            torch.profiler,
            "record_function",
            side_effect=lambda _: contextlib.nullcontext(),
        ) as record:
            pool.load_to_device_per_layer(
                device,
                (host_pages[:, None] * pool.page_size + torch.arange(2)).flatten(),
                (device_pages[:, None] * pool.page_size + torch.arange(2)).flatten(),
                0,
                positions,
                "direct",
            )

        torch.testing.assert_close(device.k_buffer, expected_k)
        torch.testing.assert_close(device.v_buffer, expected_v)
        self.assertEqual(NPU.npu_anti_quant.call_count, 4)
        labels = [call.args[0] for call in record.call_args_list]
        self.assertEqual(labels.count("kvtc/K/dequantize"), 2)
        self.assertEqual(labels.count("kvtc/V/dequantize"), 2)
        self.assertEqual(labels.count("kvtc/K/pca_reconstruct"), 2)
        self.assertEqual(labels.count("kvtc/V/pca_reconstruct"), 2)
        self.assertEqual(labels.count("kvtc/K/rope_and_device_write"), 2)
        self.assertEqual(labels.count("kvtc/K/rope"), 2)
        self.assertEqual(labels.count("kvtc/K/device_write"), 2)
        self.assertEqual(labels.count("kvtc/V/device_write"), 2)
        self.assertEqual(pool.rotary_emb.forward_native_keys_batch.call_count, 2)

    def test_reload_preserves_pca_only_and_raw_paths(self):
        host_pages = torch.tensor([5, 1], dtype=torch.int64)
        device_pages = torch.tensor([2, 0], dtype=torch.int64)
        host_indices = (
            host_pages[:, None] * 2 + torch.arange(2, dtype=torch.int64)
        ).flatten()
        device_indices = (
            device_pages[:, None] * 2 + torch.arange(2, dtype=torch.int64)
        ).flatten()
        positions = torch.arange(4)

        for compressed in (True, False):
            with self.subTest(compressed=compressed):
                NPU.npu_anti_quant.reset_mock()
                pool = self.make_pool(torch.float32)
                pool._QUANT_BATCH_MAX_PAGES = 1
                pool.k_kvtc = pool.v_kvtc = compressed
                pool.kvtc_quant_disable = compressed
                pool.layer_num = 2
                pool.device_page_shape = (2, 2, 1, 2)
                pool.rotary_emb = SimpleNamespace(
                    forward_native_keys_batch=Mock(side_effect=lambda _, x: x)
                )
                device = SimpleNamespace(
                    device="cpu",
                    k_buffer=torch.full((2, 3, 2, 1, 2), -1.0),
                    v_buffer=torch.full((2, 3, 2, 1, 2), -1.0),
                )
                if compressed:
                    pool.kvtc_k_mu = pool.kvtc_v_mu = torch.zeros(4)
                    pool.kvtc_k_V = pool.kvtc_v_V = torch.eye(4)
                    pool.k_buffer = torch.randn(8, 2, 4)
                    pool.v_buffer = torch.randn(8, 2, 4)
                    expected_k = device.k_buffer.clone()
                    expected_v = device.v_buffer.clone()
                    for host_page, device_page in zip(host_pages, device_pages):
                        expected_k[:, device_page] = pool.k_buffer[host_page].reshape(
                            2, 2, 1, 2
                        ).transpose(1, 0)
                        expected_v[:, device_page] = pool.v_buffer[host_page].reshape(
                            2, 2, 1, 2
                        ).transpose(1, 0)
                else:
                    pool.k_buffer = torch.randn(2, 8, 2, 1, 2)
                    pool.v_buffer = torch.randn(2, 8, 2, 1, 2)
                    expected_k = device.k_buffer.clone()
                    expected_v = device.v_buffer.clone()
                    for host_page, device_page in zip(host_pages, device_pages):
                        expected_k[:, device_page] = pool.k_buffer[:, host_page]
                        expected_v[:, device_page] = pool.v_buffer[:, host_page]

                with patch.object(
                    torch.profiler,
                    "record_function",
                    side_effect=AssertionError("disabled profiler context"),
                ):
                    pool.load_to_device_per_layer(
                        device,
                        host_indices,
                        device_indices,
                        0,
                        positions,
                        "direct",
                    )

                torch.testing.assert_close(
                    device.k_buffer, expected_k, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    device.v_buffer, expected_v, rtol=0, atol=0
                )
                self.assertEqual(NPU.npu_anti_quant.call_count, 0)

    def test_backup_batches_and_fallback_paths(self):
        modes = (
            (True, True, False), (True, False, False), (False, True, False),
            (True, True, True), (False, False, False),
        )
        for (k_enabled, v_enabled, disabled), count, profile in product(
            modes, (0, 1, 3), (False, True)
        ):
            with self.subTest(
                k=k_enabled, v=v_enabled, disabled=disabled, pages=count, profile=profile
            ):
                pool = self.make_pool(torch.float32)
                pool._QUANT_BATCH_MAX_PAGES = 2
                pool._profile_kvtc = profile
                pool.k_kvtc, pool.v_kvtc = k_enabled, v_enabled
                pool.kvtc_quant_disable = disabled
                device = SimpleNamespace(
                    device="cpu",
                    k_buffer=torch.randn(2, 6, 2, 1, 4),
                    v_buffer=torch.randn(2, 6, 2, 1, 4),
                )
                # A position-dependent stand-in verifies ordering is retained.
                def inverse_rope(positions, values):
                    return values + positions[:, None, None, None]

                pool.rotary_emb = SimpleNamespace(
                    invert_native_keys_batch=Mock(side_effect=inverse_rope)
                )
                layout = QUANT["build_quant_layout"](
                    [(8, "int8")], page_size=2, basis_rank=8, matrix_name="test"
                )
                host_pages = torch.tensor([5, 1, 3][:count], dtype=torch.int64)
                device_pages = torch.tensor([2, 4, 0][:count], dtype=torch.int64)
                positions = list(range(10, 10 + 2 * count))
                expected = {}
                for side, enabled in (("k", k_enabled), ("v", v_enabled)):
                    setattr(pool, f"kvtc_{side}_mu", torch.randn(8))
                    setattr(pool, f"kvtc_{side}_V", torch.randn(8, 8))
                    if enabled and not disabled:
                        payloads, scales, offsets = buffers(pool, layout)
                        setattr(pool, f"{side}_quant_layout", layout)
                        setattr(pool, f"{side}_quant_buffers", payloads)
                        setattr(pool, f"{side}_quant_scales", scales)
                        setattr(pool, f"{side}_quant_offsets", offsets)
                        expected[side] = buffers(pool, layout)
                    else:
                        shape = (8, 2, 8) if enabled else (2, 8, 2, 1, 4)
                        setattr(pool, f"{side}_buffer", torch.full(shape, -1.0))
                        expected[side] = torch.full(shape, -1.0)
                    for page, (host_page, device_page) in enumerate(
                        zip(host_pages, device_pages)
                    ):
                        raw = getattr(device, f"{side}_buffer")[:, device_page]
                        if enabled:
                            x = raw.transpose(1, 0)
                            if side == "k":
                                x = inverse_rope(
                                    torch.tensor(positions[2 * page : 2 * page + 2]), x
                                )
                            projected = (
                                x.reshape(2, -1) - getattr(pool, f"kvtc_{side}_mu")
                            ) @ getattr(pool, f"kvtc_{side}_V")
                            if disabled:
                                expected[side][host_page] = projected
                            else:
                                reference_pages(
                                    pool, projected.unsqueeze(0), [host_page],
                                    layout, expected[side],
                                )
                        else:
                            expected[side][:, host_page] = raw
                NPU.npu_dynamic_quant_asymmetric.reset_mock()
                with patch.object(
                    torch.profiler, "record_function",
                    side_effect=lambda _: contextlib.nullcontext(),
                ) as record:
                    pool.backup_from_device_all_layer(
                        device,
                        (host_pages[:, None] * 2 + torch.arange(2)).flatten(),
                        (device_pages[:, None] * 2 + torch.arange(2)).flatten(),
                        positions,
                        "direct",
                    )
                if not profile:
                    record.assert_not_called()
                batches = (count + 1) // 2
                sides = (int(k_enabled) + int(v_enabled)) if not disabled else 0
                self.assertEqual(
                    NPU.npu_dynamic_quant_asymmetric.call_count, batches * sides
                )
                self.assertEqual(
                    pool.rotary_emb.invert_native_keys_batch.call_count,
                    count if k_enabled else 0,
                )
                labels = [call.args[0] for call in record.call_args_list]
                for side, enabled in (("k", k_enabled), ("v", v_enabled)):
                    if enabled and not disabled:
                        self.assert_buffers_equal(
                            (
                                getattr(pool, f"{side}_quant_buffers"),
                                getattr(pool, f"{side}_quant_scales"),
                                getattr(pool, f"{side}_quant_offsets"),
                            ),
                            expected[side],
                        )
                        if profile:
                            self.assertEqual(
                                labels.count(f"kvtc/{side.upper()}/quantize"), batches
                            )
                            self.assertEqual(
                                labels.count(f"kvtc/{side.upper()}/pca_project"), count
                            )
                    else:
                        torch.testing.assert_close(
                            getattr(pool, f"{side}_buffer"), expected[side],
                            rtol=0, atol=0,
                        )


if __name__ == "__main__":
    unittest.main()
