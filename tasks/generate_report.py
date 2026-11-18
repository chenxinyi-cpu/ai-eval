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
    judged = sorted(glob.glob(os.path.join(REPORTS_DIR, "eval_judged_*.json")))
    raw = sorted(glob.glob(os.path.join(REPORTS_DIR, "eval_raw_*.json")))

    if judged:
        latest = judged[-1]
    elif raw:
        latest = raw[-1]
    else:
        raise FileNotFoundError(f"在 {REPORTS_DIR} 下没有找到 eval_raw_*.json")

    print(f"找到的最新的文件是: {latest}")  # 先打印出来看看对不对

    with open(latest, "r", encoding="utf-8") as f:
        records = json.load(f)

    return records, latest

def has_judge(records:list)->bool:
    return any(
        isinstance(r.get("judge"), dict) and r["judge"].get("success")
        for r in records
    )


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

    judge_stats = None
    if has_judge(records):
        judged_ok = [r for r in records if r.get("judge",{}).get("success")]
        judged_fail = [r for r in records if isinstance(r.get("judge"), dict) and not r["judge"].get("success")]

        dim_avg = {}
        #维度分
        for d in ["accuracy", "completeness", "relevance"]:
            vals = [r["judge"]["result"][d]["score"] for r in judged_ok
                    if isinstance(r["judge"]["result"].get(d), dict)
                    and isinstance(r["judge"]["result"][d].get("score"), (int, float))]
            dim_avg[d] = round(sum(vals) / len(vals), 2) if vals else None
        overall_vals = [r["judge"]["result"]["overall"] for r in judged_ok
                        if isinstance(r["judge"]["result"].get("overall"), (int, float))]
        dim_avg["overall"] = round(sum(overall_vals) / len(overall_vals), 2) if overall_vals else None

        judge_stats = {
            "success_count": len(judged_ok),
            "fail_count": len(judged_fail),
            "avg": dim_avg,
            "fail_reasons": [str(r["judge"].get("error", ""))[:80] for r in judged_fail],
        }
        print(f"Judge 统计: 成功 {len(judged_ok)}, 失败 {len(judged_fail)}, 平均分 {dim_avg}")

    # 4. badcase 双视角三分类
    JUDGE_OVERALL_THRESHOLD = 3.0
    confirmed_bad, metric_only_bad, judge_only_bad = [], [], []
    for r in success:
        bert = r["metrics"].get("bertscore_f1") if r.get("metrics") else None
        judge_res = r.get("judge", {}).get("result") if r.get("judge", {}).get("success") else None
        overall = judge_res.get("overall") if judge_res else None

        bert_low = bert is not None and bert < BERTSCORE_THRESHOLD
        judge_low = overall is not None and overall < JUDGE_OVERALL_THRESHOLD

        if bert_low and judge_low:
            confirmed_bad.append(r)
        elif bert_low and judge_res is not None:
            metric_only_bad.append(r)
        elif judge_low and bert is not None:
            judge_only_bad.append(r)
    print(
        f"Badcase 三分类: 确认 {len(confirmed_bad)}, 仅指标低 {len(metric_only_bad)}, 仅 judge 低 {len(judge_only_bad)}")


    # 5. 组装返回结果
    return {
        "total": total,
        "success_count": len(success),
        "api_fail_count": len(api_fail),
        "bert_avg": round(sum(bert_scores) / len(bert_scores), 4) if bert_scores else None,
        "bert_p50": sorted(bert_scores)[len(bert_scores) // 2] if bert_scores else None,
        "bert_p95": round(float(np.percentile(bert_scores, 95)), 4) if len(bert_scores) >= 2 else None,
        "category_stats": category_stats,
        "judge_stats": judge_stats,
        "confirmed_bad": confirmed_bad,
        "metric_only_bad": metric_only_bad,
        "judge_only_bad": judge_only_bad,
        "api_fail": api_fail,
    }

def _render_badcase_table(records: list, empty_text: str = "无") -> list:
        if not records:
            return [empty_text]
        lines = ["| case_id | 类别 | BERTScore | Judge overall | 输出摘要 |", "|---|---|---|---|---|"]
        for r in records:
            bert = r["metrics"].get("bertscore_f1") if r.get("metrics") else "N/A"
            judge_res = r.get("judge", {}).get("result") or {}
            overall = judge_res.get("overall", "N/A")
            out = (r.get("output") or "")[:40].replace("\n", " ")
            lines.append(f"| {r['case_id']} | {r['category']} | {bert} | {overall} | {out} |")
        return lines


def render_markdown(summary: dict,records: list, raw_path: str) -> str:
    lines = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 拿到原始 JSON 的相对路径用于展示
    raw_filename = os.path.basename(raw_path)

    lines.append(f"# 评测报告")
    lines.append(f"")
    lines.append(f"- 生成时间：{ts}")
    lines.append(f"- 原始数据：`reports/{raw_filename}`")
    lines.append(f"- BERTScore 阈值：**{BERTSCORE_THRESHOLD}**（低于此值视为 badcase）")
    if summary["judge_stats"]:
        lines.append(f"- Judge overall 阈值：**3.0**（1-5 分制，低于视为 judge 认为不合格）")
        lines.append(f"- 分歧判定：judge 归一化分与 BERTScore 差值 ≥ **0.3**")
    lines.append("")

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

    # 4. Judge 汇总（新增，仅在有 judge 数据时渲染）
    if summary["judge_stats"]:
        js = summary["judge_stats"]
        lines.append("## 4. Judge 汇总")
        lines.append("")
        lines.append("| 项 | 数值 |")
        lines.append("|---|---|")
        lines.append(f"| Judge 成功 | {js['success_count']} |")
        lines.append(f"| Judge 失败 | {js['fail_count']} |")
        for d, v in js["avg"].items():
            lines.append(f"| 平均 {d} | {v} |")
        lines.append("")
        if js["fail_reasons"]:
            lines.append("**Judge 失败原因：**")
            lines.append("")
            for reason in js["fail_reasons"]:
                lines.append(f"- {reason}")
            lines.append("")

    # 5. Badcase 双视角（三张表分开）
    lines.append("## 5. Badcase 双视角")
    lines.append("")

    lines.append(f"### 5.1 确认 badcase（BERTScore < {BERTSCORE_THRESHOLD} 且 judge overall < 3.0）")
    lines.append("")
    lines.extend(_render_badcase_table(summary["confirmed_bad"]))
    lines.append("")

    lines.append("### 5.2 仅指标低（疑似 reference 问题：BERTScore 低但 judge 认为质量可接受）")
    lines.append("")
    lines.extend(_render_badcase_table(summary["metric_only_bad"]))
    lines.append("")

    lines.append("### 5.3 仅 judge 低（指标盲区：BERTScore 高但 judge 认为质量不合格）")
    lines.append("")
    lines.extend(_render_badcase_table(summary["judge_only_bad"]))
    lines.append("")


    # 6. 显著分歧案例（复用数据里已算好的 _analysis.diff）
    lines.append("## 6. 显著分歧案例（|judge归一化 - BERTScore| ≥ 0.25）")
    lines.append("")
    disagree = [r for r in records
                if r.get("_analysis") and abs(r["_analysis"].get("diff", 0)) >= 0.25]
    if disagree:
        lines.append("| case_id | 类别 | BERTScore | Judge归一化 | 差值 | Judge理由 |")
        lines.append("|---|---|---|---|---|---|")
        for r in disagree:
            a = r["_analysis"]
            judge_result = (r.get("judge") or {}).get("result") or {}
            reason = (judge_result.get("summary") or "")[:50]
            lines.append(f"| {r['case_id']} | {r['category']} | {a['bert']} | {a['judge_norm']} | {a['diff']:+.3f} | {reason} |")
    else:
        lines.append("无显著分歧。演示数据中模型输出与 reference 过于接近，分歧案例需要更丰富的评测集。")
    lines.append("")

    # 7. API 失败列表
    lines.append("## 7. API 失败列表")
    lines.append("")
    if summary["api_fail"]:
        lines.append("| case_id | 错误信息 |")
        lines.append("|---|---|")
        for r in summary["api_fail"]:
            lines.append(f"| {r['case_id']} | {r['api_error']} |")
    else:
        lines.append("无。所有用例 API 调用均成功。")
    lines.append("")

    # 8. 局限说明
    lines.append("## 8. 报告局限")
    lines.append("")
    lines.append(f"- BERTScore 阈值 {BERTSCORE_THRESHOLD} 和 judge 阈值 3.0 均为演示值，生产需业务方确认")
    lines.append("- BERTScore 衡量语义相似度，不保证事实正确性；judge 补上事实判断，但裁判模型本身也可能犯错")
    lines.append("- Judge 调用成本高于指标计算，生产环境通常只对候选 badcase 或中间地带跑 judge")
    lines.append("- Rubric 质量直接决定 judge 质量，rubric 迭代需要人工参与")
    lines.append("- 样本量小时 P50/P95 参考意义有限")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    records, raw_path = load_latest_raw()
    # print("是否有可用的judge:",has_judge(records))
    summary = summarize(records)
    # import json as _j
    # print(_j.dumps(summary["judge_stats"], ensure_ascii=False, indent=2))
    # print("确认 badcase:", [r["case_id"] for r in summary["confirmed_bad"]])
    # print("仅指标低:", [r["case_id"] for r in summary["metric_only_bad"]])
    # print("仅 judge 低:", [r["case_id"] for r in summary["judge_only_bad"]])


    # 渲染 Markdown
    md = render_markdown(summary, records, raw_path)

    # 写入文件
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(REPORTS_DIR, f"eval_report_{ts}.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"\n✅ 报告已成功生成: {out_path}")