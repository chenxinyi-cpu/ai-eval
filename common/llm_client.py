import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    """统一LLM客户端：封装API调用、超时、重试、异常处理。"""

    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.model = model or os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

        if not self.api_key:
            raise ValueError("缺少 DEEPSEEK_API_KEY，请检查 .env 文件")

        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })

    def chat(self, prompt: str, timeout: int = 30, max_retries: int = 2) -> dict:
        """发一条prompt，返回{'success': bool, 'content': str, 'error': str|None}。"""
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,  # 评测时降低随机性
        }

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                resp = self.session.post(url, json=payload, timeout=timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"]
                    return {"success": True, "content": content, "error": None}
                last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
            except requests.exceptions.Timeout:
                last_error = "请求超时"
            except requests.exceptions.ConnectionError:
                last_error = "连接失败"
            except Exception as e:
                last_error = str(e)

            if attempt < max_retries:
                time.sleep(1 * (attempt + 1))

        return {"success": False, "content": None, "error": last_error}


if __name__ == "__main__":

    client = LLMClient()

    print("准备发送测试请求...")
    result = client.chat("请用一句话介绍你自己。")

    print("\n--- 调用结果 ---")
    print("成功状态:", result["success"])
    print("模型回复:", result["content"])
    if not result["success"]:
        print("错误信息:", result["error"])