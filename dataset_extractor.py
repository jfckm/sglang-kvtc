from pathlib import Path
import os
import pyarrow as pa
import pyarrow.parquet as pq

KVTC_CACHE_PATH = Path("/home/chen_baihua/.cache/")
KVTC_DATASET_PATH = KVTC_CACHE_PATH / "datasets"

OPENMATH_PARTS = 10
KVTC_DATASET_CONFIG = {
    "openmath": {
        "urls": [
            f"https://huggingface.co/datasets/open-r1/OpenR1-Math-220k/resolve/main/data/train-00{i:03d}-of-00{OPENMATH_PARTS:03d}.parquet"
            for i in range(OPENMATH_PARTS)
        ],
        "prompt_column": "problem",
    },
    "fineweb": {
        "urls": [
            "https://huggingface.co/datasets/HuggingFaceFW/fineweb/resolve/main/data/CC-MAIN-2025-26/000_00000.parquet"
        ],
        "prompt_column": "text",
    },
}
KVTC_CALIBRATION_DATASET_INDICES = {
        "openmath": [58206, 30846, 86257, 62172, 89955, 58954, 48176, 71170, 75425, 30081, 17363, 78117, 17710, 42529, 24770, 28411, 63028, 91536, 59447, 47476, 68188, 15946, 11659, 17629, 12965, 64863, 63027, 49969, 43738, 47661, 9966, 43157, 86012, 22209, 34289, 50645, 75863, 43686, 11990, 27702, 39182, 64550, 6052, 50470, 27117, 52861, 55618, 73588, 19120, 59375, 50016, 67330, 57835, 77309, 44169, 80662, 55623, 89991, 59748, 77363, 59130, 86340, 69605, 14566, 73648, 90882, 88252, 8319, 85768, 16577, 74395, 92810, 90737, 9604, 16891, 26183, 74697, 13514, 17839, 39823, 30303, 26688, 78003, 22155, 21407, 49405, 31116, 36531, 89985, 27028, 4083, 89029, 40921, 41374, 93295, 50350, 5875, 13901, 42198, 58722, 4204, 71445, 6256, 23984, 63180, 78849, 6496, 73753, 18227, 25617, 66758, 88668, 8035, 75476, 26824, 30422, 67533, 6349, 10784, 88275, 42286, 56863, 2338, 22046, 34174, 89582, 69772, 63975, 2646, 22673, 44331, 1661, 75562, 77784, 14867, 29348, 49768, 5925],
        "fineweb": [394, 3052, 6299, 9609, 12725, 13821, 698, 4345, 6729, 11361, 12764, 14045, 1569, 4946, 7297, 12089, 12794, 14360, 1912, 5472, 7537, 12528, 12850, 16279, 2563, 5775, 9101, 12652, 13035, 16553, 52, 76, 80, 86, 99, 101, 106, 111, 115, 121, 123, 129, 149, 175, 186, 192, 201, 209, 214, 223, 241, 245, 249, 257, 266, 274, 293, 299, 321, 324, 345, 351, 374, 383, 395, 406, 413, 435, 451, 452, 461, 467, 470, 483, 500, 504, 509, 524, 555, 567, 578, 579, 588, 630, 638, 640, 644, 648, 649, 674, 706, 724, 741, 761, 769, 778, 784, 792, 795, 798, 801, 810, 812, 816, 857, 859, 871, 885, 900, 905, 921, 924, 925, 928, 953, 992, 1024, 1026, 1075, 1100, 1109, 1111, 1113, 1156, 1162, 1174, 1186, 1195, 1198, 1214, 1216, 1224, 1236, 1319, 1404, 1259, 1262, 1282, 1308, 1310, 1316, 1322, 1326, 1340, 1354, 1360, 1372, 1378, 1383, 1384]
}

def load_kvtc_dataset(dataset_name: str):
    dataset_path = KVTC_DATASET_PATH / dataset_name
    output_path = Path(f"{dataset_name}_selected_problems.parquet")

    indices = sorted(KVTC_CALIBRATION_DATASET_INDICES[dataset_name])

    selected_rows = []
    global_idx = 0
    writer = None
    prev_batches = 0

    for parquet_file in sorted(dataset_path.glob("*.parquet")):
        pf = pq.ParquetFile(parquet_file)
        if writer is None:
            writer = pq.ParquetWriter(output_path, pf.schema_arrow)

        for i, batch in enumerate(pf.iter_batches(batch_size=256*1024*1024)):

            while global_idx < len(indices):
                curr_idx = indices[global_idx]
                curr_idx -= prev_batches

                row = batch.slice(curr_idx, 1)
                if row.num_rows == 0:
                    break

                selected_rows.append(row)

                global_idx += 1


            prev_batches += len(batch)

    assert global_idx == len(indices), (
            f"{dataset_name}: found {len(selected_rows)} rows "
            f"but expected {len(indices)}"
            )
    assert writer is not None

    print(f"{dataset_name}: extracted {len(selected_rows)} rows")

    for i, row in enumerate(selected_rows):
        writer.write_batch(row)

    writer.close()

    return selected_rows

def main():
    fineweb_rows = load_kvtc_dataset("fineweb")
    print(f"Fineweb schema {fineweb_rows[0].schema.names}")
    for row in fineweb_rows:
        #print(row["text"][0].as_py())
        assert len(row["text"][0].as_py()) > 0

    del fineweb_rows

    openmath_rows = load_kvtc_dataset("openmath")
    print(f"Openmath schema {openmath_rows[0].schema.names}")
    for row in openmath_rows:
        #print(row["problem"][0].as_py())
        assert len(row["problem"][0].as_py()) > 0

    del openmath_rows

if __name__ == "__main__":
    main()
