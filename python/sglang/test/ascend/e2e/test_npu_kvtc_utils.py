import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
import logging

logger = logging.getLogger("kvtc_utils")

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    TestAscendAccuracyTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    TestAscendPerformanceTestCaseBase,
)

KVTC_CACHE_PATH = Path("/root/.cache/KVTC")
KVTC_DATASET_PATH = KVTC_CACHE_PATH / "datasets"

OPENMATH_PARTS = 10
KVTC_DATASET_CONFIG = {
    "openmath": {
        "prompt_column": "problem",
    },
    "fineweb": {
        "prompt_column": "text",
    },
}

class _AscendKvtcTestCaseBase:
    kvtc_dataset_config = KVTC_DATASET_CONFIG
    kvtc_dataset_name = None

    @classmethod
    def setUpClass(cls):
        dataset_name = cls.kvtc_dataset_name
        if dataset_name is None:
            raise ValueError("kvtc_dataset_name must be set for a KVTC test")

        cls.kvtc_dataset_path = KVTC_DATASET_PATH / dataset_name
        cls.dataset_path = cls.kvtc_dataset_path
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

    def load_kvtc_dataset(self, dataset_name=None):
        dataset_name = dataset_name or self.kvtc_dataset_name
        dataset_path = KVTC_DATASET_PATH / dataset_name
        logger.info(f"Loading KVTC calibration dataset: {dataset_name}...")

        return pd.read_parquet(dataset_path)


    def get_kvtc_prompts(self, dataset_name=None):
        dataset_name = dataset_name or self.kvtc_dataset_name
        dataset = self.load_kvtc_dataset(dataset_name)
        prompt_column = self.kvtc_dataset_config[dataset_name]["prompt_column"]
        if prompt_column not in dataset.columns:
            raise ValueError(
                f"Dataset {dataset_name} does not contain prompt column "
                f"{prompt_column!r}; available columns: {list(dataset.columns)}"
            )
        prompts = [entry for entry in dataset[prompt_column]]

        logger.info(f"Found {len(prompts)} calibration prompts for {dataset_name}")

        return prompts


class TestAscendPerformanceKvtcTestCaseBase(
    _AscendKvtcTestCaseBase, TestAscendPerformanceTestCaseBase
):
    pass


class TestAscendAccuracyKvtcTestCaseBase(
    _AscendKvtcTestCaseBase, TestAscendAccuracyTestCaseBase
):
    pass
