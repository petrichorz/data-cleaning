"""汇总 Bash 本批次的转换结果，打印问题清单并保存 JSON。"""

import json
import os
import sys
from collections import Counter
from pathlib import Path


def main():
    root, dataset_list, run_dir = map(Path, sys.argv[1:])
    datasets = [Path(os.fsdecode(p)) for p in dataset_list.read_bytes().split(b"\0") if p]
    report = {"root": str(root.resolve()), "datasets": [p.name for p in datasets],
              "total": len(datasets), "clean": 0, "with_errors": 0, "incomplete": 0,
              "problems": []}
    for dataset in datasets:
        qc = dataset / "qc"
        summary_path = qc / "conversion_summary.json"
        reasons = []
        summary = None
        exit_code = None
        incomplete = False
        try:
            exit_code = int((run_dir / (dataset.name + ".status")).read_text())
            if exit_code != 0:
                raise ValueError(f"转换进程失败（退出状态 {exit_code}），摘要可能未更新")
        except (OSError, ValueError) as exc:
            incomplete = True
            reasons.append(str(exc))
        try:
            summary = json.loads(summary_path.read_text())
            if not isinstance(summary, dict):
                raise ValueError("摘要不是 JSON 对象")
            status = summary.get("status")
            errors = summary.get("errors")
            zeroed = summary.get("zeroed_rows")
            if not all(type(n) is int and n >= 0 for n in (errors, zeroed)):
                raise ValueError("摘要缺少有效的 errors/zeroed_rows 计数")
            if status not in ("complete", "complete_with_errors"):
                incomplete = True
                reasons.append(f"转换状态为 {status}")
            masked = summary.get("masked_rows", 0)  # 兼容之前仅有 zeroed_rows 的报告。
            if type(masked) is not int or masked < 0:
                raise ValueError("摘要的 masked_rows 计数无效")
            if status == "complete_with_errors" or errors or zeroed or masked:
                reasons.append(f"转换错误 {errors} 条，部分维度屏蔽 {masked} 行，整行置零 {zeroed} 行")
        except (OSError, ValueError) as exc:
            incomplete = True
            reasons.append(f"摘要缺失或无法读取：{exc}")

        if not reasons:
            report["clean"] += 1
            continue
        report["incomplete" if incomplete else "with_errors"] += 1
        # 只读取问题数据集的错误明细：按类型/字段计数，并保留前三条示例。
        counts = Counter()
        examples = []
        try:
            with (qc / "conversion_errors.jsonl").open() as log:
                for line in log:
                    error = json.loads(line)
                    counts[(error.get("error_type", "unknown"), error.get("field", ""))] += 1
                    if len(examples) < 3:
                        examples.append(error)
        except (OSError, ValueError, AttributeError, TypeError) as exc:
            reasons.append(f"错误明细缺失或无法读取：{exc}")
        report["problems"].append({
            "dataset": dataset.name, "exit_code": exit_code, "reasons": reasons,
            "summary_path": str(summary_path.resolve()), "summary": summary,
            "log_path": str((qc / "conversion.log").resolve()),
            "error_counts": [{"error_type": kind, "field": field, "count": count}
                             for (kind, field), count in sorted(counts.items())],
            "error_examples": examples,
        })

    # 只汇总本批次选中的目录；其他批次的旧 qc 不参与。
    output = root / "qc/conversion_batch_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(output)
    print(f"\n本批次 {report['total']} 个子数据集：正常 {report['clean']}，"
          f"完成但有错误 {report['with_errors']}，未完成/报告异常 {report['incomplete']}。")
    for problem in report["problems"]:
        print(f"- {problem['dataset']}：{'；'.join(problem['reasons'])}")
        for error in problem["error_counts"]:
            print(f"    {error['error_type']} {error['field']}：{error['count']} 条")
    print(f"汇总报告：{output}")
    return 1 if report["incomplete"] else 0


if __name__ == "__main__":
    sys.exit(main())
