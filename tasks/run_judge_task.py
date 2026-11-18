import json
import glob
import os
from datetime import datetime
from common.judge_client import JudgeClient

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_latest_raw()->tuple:
    files = sorted(glob.glob(os.path.join(BASE_DIR,"reports/eval_raw_*.json")))
    if not files:
        raise FileNotFoundError("没有 eval_raw_*.json，先跑 run_eval_task.py")
    latest = files[-1]
    with open(latest, "r", encoding="utf-8") as f:
        records = json.load(f)
    return records, latest


def main():
    records, raw_path = load_latest_raw()
    judge = JudgeClient()

    results=[]
    for r in records:
        if not r["api_success"] or not r["output"]:
            r["judge"] = {"success": False, "result": None, "error": "原API调用失败，跳过judge"}
            results.append(r)
            continue
        print(f"judge {r['case_id']}......")
        judged = judge.judge(
            question=r["prompt"],
            output=r["output"],
            reference=r.get("reference", ""),
        )
        r["judge"] = judged
        results.append(r)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = os.path.join(BASE_DIR,f"reports/eval_judged_{ts}.json")

    print(f"\n===BERTScore vs Judge 对照 ===")
    valid = [r for r in results
             if r["judge"]["success"]
             and r.get("metrics")
             and r["metrics"].get("bertscore_f1") is not None]
    disagreements = []
    for r in valid:
        bert = r["metrics"]["bertscore_f1"]
        overall = r["judge"]["result"].get("overall", 0)
        judge_norm = (overall - 1) / 4 if overall else 0
        diff = judge_norm - bert
        r["_analysis"] = {"bert": bert, "judge_norm": round(judge_norm, 4), "diff": round(diff, 4)}
        if abs(diff) >= 0.25:
            disagreements.append(r)

    print(f"有效对照: {len(valid)} 条, 显著分歧(差值≥0.3): {len(disagreements)} 条")
    for r in disagreements:
        a = r["_analysis"]
        print(f"  {r['case_id']}: BERT={a['bert']}, Judge归一化={a['judge_norm']}, "
              f"差值={a['diff']:+.3f} | {r['judge']['result'].get('summary','')[:40]}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n已保存: {out_path}")


if __name__ == "__main__":
    main()
















