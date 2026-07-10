import requests
import unittest
import openai
import pandas as pd
from urllib.parse import urlparse
from pathlib import Path

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    QWEN3_30B_A3B_MODEL_PATH,
    QWEN3_A3B_EAGLE_MODEL_PATH,
    TestAscendPerformanceTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="",
    nightly=True,
    disabled="performance testcase",
)

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
    "--disable-radix-cache",
    "--chunked-prefill-size",
    -1,
    "--max-prefill-tokens",
    8300,
    "--tp-size",
    2,
    "--enable-dp-attention",
    "--dp-size",
    2,
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

KVTC_DATASET_PATH = Path("/root/.cache/KVTC/datasets")

class TestKVTCQwen30B(TestAscendPerformanceTestCaseBase):
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    dataset_type = AISBENCHMARK_DATASET_DEFAULT
    model = QWEN3_30B_A3B_MODEL_PATH
    other_args = OTHER_ARGS
    envs = ENVS
    dataset_name = "random"
    max_concurrency = 162
    num_prompts = 624
    input_len = 1000
    output_len = 100
    random_range_ratio = 1
    seed = 1
    mean_e2e_latency = 10000
    output_token_throughput = 2047.81
    max_attempts = 4

    def _download_dataset(self, name: str, remote_address: str):
        download_path = KVTC_DATASET_PATH
        download_path.mkdir(parents=True, exist_ok=True)

        proxies = {
                "http": None,
                "https": None,
                }

        ret = requests.get(remote_address, verify=False, proxies=proxies)
        # print download stats 

        file_path = download_path / name
        with open(file_path, "wb") as f:
            f.write(ret.content)

        return file_path

#    def test_kvtc_qwen3_30b_generate_openmath_dumps(self):
#        remote_address = "https://huggingface.co/datasets/HuggingFaceFW/fineweb/blob/main/data/CC-MAIN-2025-26/000_00000.parquet"
#
#        openmath_path = _download_dataset("openmath", remote_address)
#        openmath_dataset = pd.read_parquet(file_path)


    def test_kvtc_qwen3_30b_generate_openmath_dumps(self):
        remote_address = "https://huggingface.co/datasets/nvidia/OpenMathReasoning/resolve/main/data/additional_problems-00000-of-00001.parquet"

        openmath_path = self._download_dataset("openmath", remote_address)

        openmath_dataset = pd.read_parquet(file_path).iloc

#        for idx in prompt_indices:
#            print(openmath_dataset[idx])

        for entry in openmath_dataset:
            print(f"{entry=}")

        import pdb
        pdb.set_trace()

        parsed_url = urlparse(self.base_url)
        host = parsed_url.hostname
        port = parsed_url.port

        print(parsed_url)

        client = openai.Client(base_url=f"http://{host}:{port}/v1", api_key="None")

        response = client.chat.completions.create(
                model="Qwen3-30B-A3B",
                messages = [
                    {"role": "user", "content": "Who is the maintainer of Open CAS Linux"},
                    ]
                temperature=0,
                max_tokens=64,
                )

        print(f"The server responed with {response}")

        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
