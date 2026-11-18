import os
import glob
import json
from collections import defaultdict
from datetime import datetime
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
BERTSCORE_THRESHOLD = 0.7


def load_latest_raw() -> tuple:
    # 注意这里：用 os.path.join 拼接绝对路径去找
    search_pattern = os.path.join(REPORTS_DIR, "eval_raw_*.json")
    files = sorted(glob.glob(search_pattern))

    if not files:
        raise FileNotFoundError(f"在 {REPORTS_DIR} 下没有找到 eval_raw_*.json")

    latest = files[-1]
    print(f"找到的最新的文件是: {latest}")  # 先打印出来看看对不对

    with open(latest, "r", encoding="utf-8") as f:
        records = json.load(f)

    return records, latest


def summarize(records: list) -> dict:
    total = len(records)

    # 1. 筛选：API成功 且 有指标数据，才算有效评测
    success = [r for r in records if r["api_success"] and r["metrics"]]
    # API 失败单独拿出来
    api_fail = [r for r in records if not r["api_success"]]

    print(f"\n--- 统计开始 ---")
    print(f"总用例: {total}, 成功且有指标: {len(success)}, API失败: {len(api_fail)}")

    # 2. 提取所有成功的 BERTScore
    bert_scores = [r["metrics"]["bertscore_f1"] for r in success if r["metrics"]["bertscore_f1"] is not None]

    # 3. 按类别分组统计
    by_category = defaultdict(list)
    for r in success:
        by_category[r["category"]].append(r)

    category_stats = {}
    for cat, items in by_category.items():
        scores = [i["metrics"]["bertscore_f1"] for i in items if i["metrics"]["bertscore_f1"] is not None]
        avg_score = round(sum(scores) / len(scores), 4) if scores else None
        category_stats[cat] = {
            "count": len(items),
            "avg_bertscore": avg_score
        }
        print(f"类别 [{cat}] 统计完毕: {len(items)} 条, 平均分 {avg_score}")

    # 4. 筛选 badcase（BERTScore < 阈值）
    badcases = []
    for r in success:
        score = r["metrics"]["bertscore_f1"] if r["metrics"] else None
        if score is not None and score < BERTSCORE_THRESHOLD:
            badcases.append(r)
    print(f"找到 badcase: {len(badcases)} 条")

    # 5. 组装返回结果
    return {
        "total": total,
        "success_count": len(success),
        "api_fail_count": len(api_fail),
        "bert_avg": round(sum(bert_scores) / len(bert_scores), 4) if bert_scores else None,
        "bert_p50": sorted(bert_scores)[len(bert_scores) // 2] if bert_scores else None,
        "bert_p95": round(float(np.percentile(bert_scores, 95)), 4) if len(bert_scores) >= 2 else None,
        "category_stats": category_stats,
        "badcases": badcases,
        "api_fail": api_fail,
    }


def render_markdown(summary: dict, raw_path: str) -> str:
    lines = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 拿到原始 JSON 的相对路径用于展示
    raw_filename = os.path.basename(raw_path)

    lines.append(f"# 评测报告")
    lines.append(f"")
    lines.append(f"- 生成时间：{ts}")
    lines.append(f"- 原始数据：`reports/{raw_filename}`")
    lines.append(f"- BERTScore 阈值：**{BERTSCORE_THRESHOLD}**（低于此值视为 badcase）")
    lines.append(f"")

    # 1. 概览
    lines.append("## 1. 概览")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("|---|---|")
    lines.append(f"| 总用例数 | {summary['total']} |")
    lines.append(f"| API 成功 | {summary['success_count']} |")
    lines.append(f"| API 失败 | {summary['api_fail_count']} |")
    lines.append(f"| 有效评测数 | {summary['success_count']} |")
    lines.append("")

    # 2. 指标汇总
    lines.append("## 2. 指标汇总（仅统计 API 成功且指标有效的用例）")
    lines.append("")
    lines.append("| 指标 | 平均值 | P50 | P95 |")
    lines.append("|---|---|---|---|")
    lines.append(f"| BERTScore-F1 | {summary['bert_avg']} | {summary['bert_p50']} | {summary['bert_p95']} |")
    lines.append("")
    lines.append(f"> P95 表示 95% 的用例分数不低于此值。样本量小时 P95 参考意义有限。")
    lines.append("")

    # 3. 类别分布
    lines.append("## 3. 类别分布")
    lines.append("")
    lines.append("| 类别 | 用例数 | 平均 BERTScore |")
    lines.append("|---|---|---|")
    for cat, stats in summary["category_stats"].items():
        lines.append(f"| {cat} | {stats['count']} | {stats['avg_bertscore']} |")
    lines.append("")
    lines.append(f"> 只看总体平均会掩盖类别差异。如果某个类别明显偏低，说明模型在该类场景下较弱。")
    lines.append("")

    # 4. badcase 列表
    lines.append(f"## 4. Badcase（BERTScore < {BERTSCORE_THRESHOLD}）")
    lines.append("")
    if summary["badcases"]:
        lines.append("| case_id | 类别 | Prompt（截断） | BERTScore | 输出（截断） |")
        lines.append("|---|---|---|---|---|")
        for r in summary["badcases"]:
            # 截断处理，防止表格爆炸
            prompt_short = r["prompt"][:30] + ("..." if len(r["prompt"]) > 30 else "")
            output_short = (r["output"] or "")[:40].replace("\n", " ") + (
                "..." if r["output"] and len(r["output"]) > 40 else "")
            score = r["metrics"]["bertscore_f1"] if r["metrics"] else "N/A"
            lines.append(f"| {r['case_id']} | {r['category']} | {prompt_short} | {score} | {output_short} |")
    else:
        lines.append("无。所有成功用例的 BERTScore 均不低于阈值。")
    lines.append("")

    # 5. API 失败列表
    lines.append("## 5. API 失败列表")
    lines.append("")
    if summary["api_fail"]:
        lines.append("| case_id | 错误信息 |")
        lines.append("|---|---|")
        for r in summary["api_fail"]:
            lines.append(f"| {r['case_id']} | {r['api_error']} |")
    else:
        lines.append("无。所有用例 API 调用均成功。")
    lines.append("")

    # 6. 局限说明
    lines.append("## 6. 报告局限")
    lines.append("")
    lines.append(f"- 阈值 {BERTSCORE_THRESHOLD} 为演示值，生产环境需业务方确认")
    lines.append("- BLEU/ROUGE 对中文短句未分词敏感，导致部分分数异常为 0，建议生产环境引入分词器")
    lines.append("- BERTScore 衡量语义相似度，不保证事实正确性")
    lines.append("- 样本量小时 P50/P95 参考意义有限")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    records, raw_path = load_latest_raw()
    summary = summarize(records)

    # 渲染 Markdown
    md = render_markdown(summary, raw_path)

    # 写入文件
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(REPORTS_DIR, f"eval_report_{ts}.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"\n✅ 报告已成功生成: {out_path}")