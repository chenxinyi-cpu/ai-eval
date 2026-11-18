import json
import os
import re
import time

import requests

JUDGE_API_URL = "https://api.deepseek.com/chat/completions"

DEFAULT_JUDGE_PROMPT = """
你是一个严格的技术问答评审专家。请对下面这条模型输出进行评分。

**用户问题：**
{question}

**参考回答（仅供参考，不是唯一标准答案）：**
{reference}

**模型输出：**
{output}

请从以下三个维度打分，每个维度 1-5 分：
1. 准确性 (accuracy)：事实是否正确
2. 完整性 (completeness)：是否覆盖关键信息
3. 相关性 (relevance)：是否切题

参考回答不是唯一标准。如果模型输出与参考不同但事实正确、回答了问题，应给高分。

请严格按以下 json 格式输出，不要输出其他内容：
{{"accuracy": {{"score": 数字, "reason": "理由"}},
  "completeness": {{"score": 数字, "reason": "理由"}},
  "relevance": {{"score": 数字, "reason": "理由"}},
  "overall": 数字,
  "summary": "一句话总结"}}
"""

from dotenv import load_dotenv

load_dotenv()


class JudgeClient:
    def __init__(self, api_key: str = None, model: str = "deepseek-chat",
                 prompt_path: str = "prompts/judge_prompt.txt"):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("缺少 DEEPSEEK_API_KEY")
        self.model = model
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })
        if os.path.exists(prompt_path):
            with open(prompt_path, "r", encoding="utf-8") as f:
                self.template = f.read()
        else:
            self.template = DEFAULT_JUDGE_PROMPT

    def judge(self, question: str, output: str, reference: str = "", max_retries: int = 2) -> dict:
        prompt = self.template.format(
            question=question, output=output, reference=reference or "无"
        )
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 600,
            "response_format": {"type": "json_object"},
        }

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                resp = self.session.post(JUDGE_API_URL, json=payload, timeout=60)
                if resp.status_code != 200:
                    last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                else:
                    content = resp.json()["choices"][0]["message"]["content"]
                    if not content or not content.strip():
                        last_error = "裁判返回空内容"
                    else:
                        parsed = self._parse_json(content)
                        if parsed:
                            return {"success": True, "result": parsed, "error": None}
                        last_error = f"JSON解析失败: {content[:200]}"
            except requests.exceptions.Timeout:
                last_error = "裁判请求超时"
            except requests.exceptions.ConnectionError:
                last_error = "裁判连接失败"
            except Exception as e:
                last_error = str(e)

            if attempt < max_retries:
                time.sleep(1 * (attempt + 1))

        return {"success": False, "result": None, "error": last_error}

    @staticmethod
    def _parse_json(text: str) -> dict:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    return None
            return None