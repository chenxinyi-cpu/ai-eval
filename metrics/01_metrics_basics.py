import evaluate
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import AutoTokenizer
import bert_score.utils

# --- 核心补丁：拦截 tokenizer 加载，强制限制 max_length ---
_original_from_pretrained = AutoTokenizer.from_pretrained

def _patched_from_pretrained(*args, **kwargs):
    tokenizer = _original_from_pretrained(*args, **kwargs)
    # 如果 model_max_length 大于 1,000,000（说明是异常占位值），强行改成 512
    if tokenizer.model_max_length > 1_000_000:
        tokenizer.model_max_length = 512
    return tokenizer

AutoTokenizer.from_pretrained = _patched_from_pretrained
# -----------------------------------------------------------

# 参考文本（标准答案）
reference = "今天天气很好，适合出门散步。"

# 三组对比数据
cases = [
    ("完全相同", "今天天气很好，适合出门散步。"),
    ("同义改写", "今日气候不错，适宜外出走走。"),
    ("语义相反", "今天下雨了，最好待在家里。"),
]

bleu = evaluate.load("bleu")
print("bleu加载完成")
rouge = evaluate.load("rouge")
print("rouge加载完成")
bertscore = evaluate.load("bertscore")
print("bertscore加载完成")

def score(pred:str)->dict:
    b = bleu.compute(predictions=[pred],references=[reference])
    r = rouge.compute(predictions=[pred],references=[reference])
    s = bertscore.compute(
        predictions=[pred],
        references=[reference],
        model_type=r"E:\test\ai-eval\bert-base-chinese",
        num_layers=12,
        lang="zh"
    )
    return {
        "bleu": round(b["bleu"], 4),
        "rouge1": round(r["rouge1"], 4),
        "rougeL": round(r["rougeL"], 4),
        "bertscore_f1": round(s["f1"][0], 4),
    }

if __name__ == "__main__":
    print(f"{'场景':<10} {'BLEU':<10} {'ROUGE-1':<10} {'ROUGE-L':<10} {'BERTScore-F1':<14}")
    print("-" * 62)
    for name, pred in cases:
        r = score(pred)
        print(f"{name:<10} {r['bleu']:<10} {r['rouge1']:<10} "
              f"{r['rougeL']:<10} {r['bertscore_f1']:<14}")
