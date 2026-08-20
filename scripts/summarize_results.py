import json
from pathlib import Path

# 定义输出目录路径
OUTPUT_DIR = Path("./outputs")


def summarize_results():
    results = []

    # 遍历所有实验文件夹
    if not OUTPUT_DIR.exists():
        print(f"目录 {OUTPUT_DIR} 不存在。")
        return

    for run_dir in OUTPUT_DIR.iterdir():
        metrics_file = run_dir / "test_metrics.json"
        if metrics_file.exists():
            with open(metrics_file, "r") as f:
                data = json.load(f)
                results.append({
                    "name": run_dir.name,
                    "accuracy": data.get("macro_accuracy", 0.0),
                    "qwk": data.get("qwk", 0.0)
                })

    # 按 QWK 降序排列
    results.sort(key=lambda x: x["qwk"], reverse=True)

    print(f"{'Experiment Name':<25} | {'QWK':<8} | {'Accuracy':<8}")
    print("-" * 50)
    for r in results:
        print(f"{r['name']:<25} | {r['qwk']:.4f}   | {r['accuracy']:.4f}")


if __name__ == "__main__":
    summarize_results()