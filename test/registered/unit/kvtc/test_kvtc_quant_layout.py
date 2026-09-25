"""CPU checks for the independent dtype-grouped KVTC layout builder."""

import runpy
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
QUANT = runpy.run_path(
    str(ROOT / "python/sglang/srt/mem_cache/kvtc_quant.py")
)


class TestKVTCQuantGroupedLayout(unittest.TestCase):
    def build_both(self, schema, *, page_size=128, basis_rank=64):
        kwargs = {
            "page_size": page_size,
            "basis_rank": basis_rank,
            "matrix_name": "test",
        }
        return (
            QUANT["build_quant_layout"](schema, **kwargs),
            QUANT["build_quant_layout_new"](schema, **kwargs),
        )

    def test_interleaved_dtypes_keep_feature_and_metadata_positions(self):
        schema = [
            (3, "float32"),
            (8, "int4"),
            (5, "int8"),
            (16, "int4"),
            (2, "bfloat16"),
            (4, "int8"),
        ]
        old, new = self.build_both(schema)

        self.assertEqual(list(new.direct_storage_groups), ["float32", "bfloat16"])
        self.assertEqual(list(new.integer_quant_groups), ["int8", "int4"])
        self.assertEqual(
            {
                dtype: [group.feature_start for group in groups]
                for groups_by_dtype in (
                    new.direct_storage_groups,
                    new.integer_quant_groups,
                )
                for dtype, groups in groups_by_dtype.items()
            },
            {
                "float32": [0],
                "bfloat16": [32],
                "int8": [11, 34],
                "int4": [3, 16],
            },
        )
        self.assertEqual(new.feature_count, 38)
        self.assertEqual(new.metadata_count, 4)
        self.assertEqual(
            new.payload_elements,
            {"float32": 384, "bfloat16": 256, "int8": 1152, "int4": 384},
        )
        self.assertEqual(new.feature_count, old.feature_count)
        self.assertEqual(new.metadata_count, old.metadata_count)
        self.assertEqual(new.payload_elements, old.payload_elements)
        self.assertEqual(
            sorted(
                (
                    group
                    for groups_by_dtype in (
                        new.direct_storage_groups,
                        new.integer_quant_groups,
                    )
                    for groups in groups_by_dtype.values()
                    for group in groups
                ),
                key=lambda group: group.feature_start,
            ),
            list(old.groups),
        )

    def test_single_dtype_keeps_separate_groups(self):
        old, new = self.build_both([(5, "int8"), (7, "int8")], page_size=2)
        self.assertEqual(new.direct_storage_groups, {})
        self.assertEqual(list(new.integer_quant_groups), ["int8"])
        self.assertEqual(new.integer_quant_groups["int8"], old.groups)
        self.assertEqual(new.payload_elements, {"int8": 24})
        self.assertEqual(new.metadata_count, 2)

    def test_float_groups_need_no_metadata(self):
        old, new = self.build_both([(3, "bfloat16"), (2, "float32")])
        self.assertEqual(new.metadata_count, 0)
        self.assertEqual(new.integer_quant_groups, {})
        self.assertEqual(
            [
                group.metadata_index
                for groups in new.direct_storage_groups.values()
                for group in groups
            ],
            [None, None],
        )
        self.assertEqual(new.payload_elements, old.payload_elements)

    def test_validation_matches_existing_builder(self):
        invalid = [
            ([], 64),
            ([(3, "float32", "extra")], 64),
            ([(True, "int8")], 64),
            ([(0, "int8")], 64),
            ([(3, "unsupported")], 64),
            ([(7, "int4")], 64),
            ([(10, "int4")], 64),
            ([(16, "int4")], 8),
        ]
        for schema, basis_rank in invalid:
            with self.subTest(schema=schema, basis_rank=basis_rank):
                errors = []
                for builder_name in ("build_quant_layout", "build_quant_layout_new"):
                    with self.assertRaises(ValueError) as caught:
                        QUANT[builder_name](
                            schema,
                            page_size=128,
                            basis_rank=basis_rank,
                            matrix_name="test",
                        )
                    errors.append(str(caught.exception))
                self.assertEqual(*errors)


if __name__ == "__main__":
    unittest.main()
