"""NPU Qwen 3 Ruler lm-eval Evaluation Test (2-NPU)

Tests Qwen/Qwen3-30B-A3B with lm-eval ruler benchmark on NPU
"""


import unittest

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    QWEN3_30B_A3B_MODEL_PATH,
)
from sglang.test.ascend.e2e.test_npu_kvtc_utils import (
    TestAscendPerformanceKvtcTestCaseBase,
    KVTC_CALIBRATION_PARAMS,
)
from sglang.test.ci.ci_register import register_npu_ci
import openai

import os
import unittest
import numpy as np
import yaml
import requests

from pathlib import Path

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.ascend.e2e.test_npu_kvtc_utils import (
    TestAscendPerformanceKvtcTestCaseLME,
)
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

QWEN3_30B_A3B_MODEL_PATH = "/root/.models/Qwen3-30B-A3B"
DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH = 3600
TP_SIZE = 2


KVTC_PARAMS_PATH = "/root/.cache/KVTC/QWEN3_30B_A3B/kvtc_conf.pt"

ENVS = {
    "ASCEND_LAUNCH_BLOCKING": "0",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:False",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "INF_NAN_MODE_FORCE_DISABLE": "1",
    "HCCL_ALGO": "level0:NA;level1:ring",
    "DP_ROUND_ROBIN": "1",
    "SGLANG_USE_MAX_DP_ATT": "1",
    "SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE": "1",
    "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES": "200",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
}

OTHER_ARGS = [
    "--trust-remote-code",
    "--nnodes",
    "1",
    "--node-rank",
    "0",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--max-running-requests",
    168,
    "--chunked-prefill-size",
    -1,
    "--tp-size",
    2,
    "--enable-dp-attention",
    "--dp-size",
    1,
    "--mem-fraction-static",
    0.85,
    "--cuda-graph-bs",
    1,
    2,
    4,
    8,
    16,
    20,
    24,
    28,
    32,
    36,
    40,
    44,
    48,
    52,
    56,
    60,
    64,
    68,
    72,
    76,
    80,
    84,
    "--dtype",
    "bfloat16",
    "--reasoning-parser",
    "qwen3",
    "--tool-call-parser",
    "qwen",
]


QWEN3_30B_A3B_ENVS = {
    "ASCEND_LAUNCH_BLOCKING": "0",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:False",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "INF_NAN_MODE_FORCE_DISABLE": "1",
    "HCCL_ALGO": "level0:NA;level1:ring",
    "DP_ROUND_ROBIN": "1",
    "SGLANG_USE_MAX_DP_ATT": "1",
    "SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE": "1",
    "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES": "200",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",

}

QWEN3_30B_A3B_OTHER_ARGS = [
    "--trust-remote-code",
    "--nnodes",
    "1",
    "--node-rank",
    "0",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--max-running-requests",
    168,
    "--chunked-prefill-size",
    -1,
    "--max-prefill-tokens",
    8300,
    "--tp-size",
    2,
    "--enable-dp-attention",
    "--dp-size",
    1,
    "--mem-fraction-static",
    0.85,
    "--cuda-graph-bs",
    1,
    2,
    4,
    8,
    16,
    20,
    24,
    28,
    32,
    36,
    40,
    44,
    48,
    52,
    56,
    60,
    64,
    68,
    72,
    76,
    80,
    84,
    "--dtype",
    "bfloat16",
    "--reasoning-parser",
    "qwen3",
    "--tool-call-parser",
    "qwen",
    "--enable-hierarchical-cache",
    "--hicache-kvtc-params",
    KVTC_PARAMS_PATH,
    "--hicache-kvtc-k-cr",
    "8",
    "--hicache-kvtc-v-cr",
    "8",
    "--hicache-size",
    "80",
]


class TestNPUQwen3_30BA3B_1P_ruler(TestAscendPerformanceKvtcTestCaseLME, CustomTestCase):
    """Qwen 3 Ruler lm-eval Test for NPU"""

    model_config_name = "/workspace/sglang/test/lm_eval_configs/NPU-kvtc-Qwen3-30B-A3B-ruler.yaml"

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_30B_A3B_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            model=QWEN3_30B_A3B_MODEL_PATH,
            base_url=DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            env=QWEN3_30B_A3B_ENVS,
            other_args=QWEN3_30B_A3B_OTHER_ARGS,
        )



    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_lm_eval(self):
        eval_config = yaml.safe_load(
            Path(self.model_config_name).read_text(encoding="utf-8")
        )

        # filling kvcache run
        self.launch_lm_eval(eval_config)

        # requests.post(url=self.base_url + "/flush_cache", timeout=30)
        # resp = requests.get(url=self.base_url+ "/radix_tree", timeout=30)
        # data = ast.literal_eval(resp.text)
        # self.assertFalse(self._validate_trim_eviction(self, data))


        results = self.launch_lm_eval(eval_config)
        rtol = eval_config.get("rtol", self.default_rtol)
        model_name = eval_config.get("model_name", self.model)

        success = True
        summary = f"### lm-eval accuracy ({model_name})\n"
        summary += "| task | metric | expected | measured | status |\n"
        summary += "| ---- | ------ | -------- | -------- | ------ |\n"
        for task in eval_config["tasks"]:
            for metric in task["metrics"]:
                expected = metric["value"]
                measured = results["results"][task["name"]][metric["name"]]
                passed = bool(np.isclose(expected, measured, rtol=rtol))
                status = "✅" if passed else "❌"
                summary += f"| {task['name']} | {metric['name']} | {expected:.4f} | {measured:.4f} | {status} |\n"
                print(
                    f"{task['name']} | {metric['name']}: "
                    f"expected={expected:.3f} | measured={measured:.3f} | rtol={rtol}"
                )
                success = success and passed

        self.assertTrue(success, "lm-eval validation failed")


            # model_name: "/root/.models/Qwen3-30B-A3B"
            # tasks:
            # - name: "ruler"
            #   metrics:
            #   - name: "4096,none"
            #     value: 0.95
            # limit: 4096
            # num_concurrent: 16
            # max_retries: 3
            # tokenized_requests: False
            # metadata:
            #   pretrained: /root/.models/Qwen3-30B-A3B
            #   max_seq_lengths:
            #     - 4096


class TestKVTCQwen30BCalibrateLong(TestAscendPerformanceKvtcTestCaseBase):
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    dataset_typ e = AISBENCHMARK_DATASET_DEFAULT
    model = QWEN3_30B_A3B_MODEL_PATH
    other_args = OTHER_ARGS
    envs = ENVS
    kvtc_force_calibration = False
    kvtc_calibration_params = {
        **KVTC_CALIBRATION_PARAMS,
        "N": 200000,
        "q": 8000,
    }

    def test_kvtc_qwen3_30b_dump_reuse(self):
        client = openai.Client(base_url=f"{self.base_url}/v1", api_key="None")

        messages = [
            {"role": "system", "content": "You are a helpful asistant."},
            {"role": "user", "content": "Compute (3+5)"},
        ]

        response = client.chat.completions.create(
            model=self.model,
            max_tokens=2048,
            messages=messages,
            temperature=0.8,
            top_p=0.8,
            stream=False,
        )

        reason = response.choices[0].finish_reason
        self.assertEqual(reason, "stop")
        self.assertTrue(
            self._has_current_kvtc_dump(),
            f"KVTC dump was not created: {self.kvtc_dump_path}",
        )

if __name__ == "__main__":
    unittest.main()
