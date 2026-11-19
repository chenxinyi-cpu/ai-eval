import glob
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')


def load_latest_two(prefix:str)->tuple:
    files = sorted(glob.glob(os.path.join(REPORTS_DIR,f"{prefix}_*.json")))
    if not files:
        return None,None

    curr_path = files[-1]
    with open(curr_path,"r",encoding="utf-8") as f:
        curr = json.load(f)
    if len(files) == 1:
        return None,(curr,curr_path)

    prev_path = files[-2]
    with open(prev_path,"r",encoding="utf-8") as f:
        prev = json.load(f)
    return (prev,prev_path),(curr,curr_path)


def avg_bertscore(records:list)->float | None:
    vals = [r["metrics"]["bertscore_f1"] for r in records
            if r.get("metrics") and r["metrics"].get("bertscore_f1") is not None]
    return round(sum(vals) / len(vals),2) if vals else None


def avg_judge_overall(records:list)->float | None:
    vals = [r["judge"]["result"]["overall"] for r in records
            if r.get("judge", {}).get("success")
            and isinstance(r["judge"].get("result"), dict)
            and r["judge"]["result"].get("overall") is not None]
    return round(sum(vals) / len(vals),2) if vals else None


BERTSCORE_DROP_THRESHOLD = 0.05   # BERTScore 均值下跌超过 5% 告警
JUDGE_DROP_THRESHOLD = 0.3        # judge overall 均值下跌超过 0.3 分告警


def main():
    prev_pair,curr_pair = load_latest_two("eval_judged")
    if prev_pair is None and curr_pair is None:
        prev_pair,curr_pair = load_latest_two("eval_raw")

    if curr_pair is None:
        print("没有评测数据，跳过回归检查")
        sys.exit(0)

    curr_records,curr_path = curr_pair
    curr_bert = avg_bertscore(curr_records)
    curr_judge = avg_judge_overall(curr_records)

    if prev_pair is None:
        print(f"首次评测，无基线。当前数据: {curr_path}")
        print(f"当前 BERTScore 均值: {curr_bert}, Judge overall 均值: {curr_judge}")
        print("基线已记录，下次运行开始对比")
        sys.exit(0)

    prev_records, prev_path = prev_pair
    prev_bert = avg_bertscore(prev_records)
    prev_judge = avg_judge_overall(prev_records)
    print(f"基线: {prev_path}")
    print(f"本次: {curr_path}")

    failed = False

    if curr_bert is not None and prev_bert is not None:
        drop = curr_bert - prev_bert
        print(f"BERTScore 均值: 基线 {prev_bert:.4f} -> 本次 {curr_bert:.4f} (变化 {drop:.4f})")
        if drop > BERTSCORE_DROP_THRESHOLD:
            print(f"❌ BERTScore 下跌 {drop:.4f}，超过阈值 {BERTSCORE_DROP_THRESHOLD}")
            failed = True

    if curr_judge is not None and prev_judge is not None:
        drop = curr_judge - prev_judge
        print(f"Judge overall 均值: 基线 {prev_judge:.2f} -> 本次 {curr_judge:.2f} (变化 {drop:.2f})")
        if drop > JUDGE_DROP_THRESHOLD:
            print(f"❌ Judge overall 下跌 {drop:.2f}，超过阈值 {JUDGE_DROP_THRESHOLD}")
            failed = True

    if failed:
        sys.exit(1)
    print("✅ 回归检查通过，无显著下跌")




if __name__ == "__main__":
    main()


