import json
import os
from datetime import datetime

import evaluate
import yaml

from common.llm_client import LLMClient


# ===== 修复 BERTScore int too big to convert =====
from transformers import AutoTokenizer

_original_from_pretrained = AutoTokenizer.from_pretrained

def _patched_from_pretrained(*args, **kwargs):
    tokenizer = _original_from_pretrained(*args, **kwargs)
    # 强制把异常的 model_max_length 钳制到 512
    tokenizer.model_max_length = 512
    return tokenizer

AutoTokenizer.from_pretrained = _patched_from_pretrained
# ================================================

# 加载三个评测指标
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")

# ==========================================
# 核心：动态获取项目根目录（绝对路径）
# __file__ 是当前文件路径，找两次 parent 就退到了 ai-eval 根目录
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_SET_PATH = os.path.join(BASE_DIR, "eval_sets", "demo_set.yaml")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
MODEL_PATH = os.path.join(BASE_DIR, "bert-base-chinese")
# 本地有模型文件夹就用本地（你本地开发），CI 没有就回退到 HuggingFace 模型名
BERT_MODEL = MODEL_PATH if os.path.isdir(MODEL_PATH) else "bert-base-chinese"


def load_cases(path: str) -> list:
    """读取yaml评测集"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_metrics(prediction: str, reference: str) -> dict:
    """单条指标计算，容错：任何指标失败返回None，不中断全流程。"""
    result = {"bleu": None, "rouge1": None, "rougeL": None, "bertscore_f1": None}

    # 1. 计算 BLEU（加入 jieba 分词，避免短句全 0）
    try:
        # BLEU 的 references 参数需要嵌套列表（支持多个参考答案）
        result["bleu"] = round(bleu.compute(predictions=[prediction], references=[[reference]])["bleu"], 4)
    except Exception as e:
        print(f"❌ BLEU 计算失败: {e}")

    # 2. 计算 ROUGE
    try:
        r = rouge.compute(predictions=[prediction], references=[reference])
        result["rouge1"] = round(r["rouge1"], 4)
        result["rougeL"] = round(r["rougeL"], 4)
    except Exception as e:
        print(f"❌ ROUGE 计算失败: {e}")

    # 3. 计算 BERTScore（使用动态获取的本地绝对路径）
    try:
        result["bertscore_f1"] = round(
            bertscore.compute(
                predictions=[prediction],
                references=[reference],
                model_type=BERT_MODEL,
                num_layers=12
            )["f1"][0], 4
        )
    except Exception as e:
        print(f"❌ BERTScore 计算失败: {e}")

    return result


def main():
    # 1. 初始化客户端和读取数据（使用绝对路径）
    client = LLMClient()
    cases = load_cases(EVAL_SET_PATH)
    results = []

    # 2. 遍历执行评测
    for case in cases:
        print(f"正在处理: {case['case_id']} - {case['prompt']}")
        resp = client.chat(case["prompt"])

        # 组装单条记录
        record = {
            "case_id": case["case_id"],
            "category": case["category"],
            "prompt": case["prompt"],
            "reference": case["reference"],
            "output": resp["content"],
            "api_success": resp["success"],
            "api_error": resp["error"],
            "metrics": None,
        }

        # 如果API调用成功且有返回内容，才去算分
        if resp["success"] and resp["content"]:
            record["metrics"] = compute_metrics(resp["content"], case["reference"])

        results.append(record)

    # 3. 保存结果到JSON文件（使用绝对路径）
    os.makedirs(REPORTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = os.path.join(REPORTS_DIR, f"eval_raw_{timestamp}.json")

    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 4. 打印简单汇总
    print(f"\n评测完成！共 {len(results)} 条。")
    success_cases = [r for r in results if r["api_success"] and r["metrics"]]
    print(f"成功执行并打分: {len(success_cases)} 条")
    print(f"结果已保存至: {raw_path}")


if __name__ == "__main__":
    main()