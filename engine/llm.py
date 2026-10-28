"""DeepSeek client — 极简，只用标准库，便于部署。"""
import json
import os
import time
import urllib.request
import urllib.error
from pathlib import Path

ENV_PATH = Path.home() / ".hermes" / ".env"


def _load_env():
    if not ENV_PATH.exists():
        return
    for line in ENV_PATH.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)


_load_env()

BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")

# 本机需要走代理才能访问外网 API；urllib 默认 opener 在本环境下会 reset。
# 优先读 env，读不到时用本机常驻的代理端口（WSL → Windows 宿主代理）。
_PROXY = os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY") or ""
if not _PROXY:
    import socket

    for cand in ("http://127.0.0.1:7892", "http://127.0.0.1:7890", "http://127.0.0.1:10809"):
        host, port = cand.split("//")[1].split(":")
        try:
            with socket.create_connection((host, int(port)), timeout=0.6):
                _PROXY = cand
                break
        except OSError:
            continue

_OPENER = (
    urllib.request.build_opener(
        urllib.request.ProxyHandler({"http": _PROXY, "https": _PROXY})
    )
    if _PROXY
    else urllib.request.build_opener()
)



def chat(messages, temperature=0.7, max_tokens=3000, json_mode=False, retries=2):
    if not API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY 未配置")
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    last_err = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(
            f"{BASE_URL}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}",
            },
            method="POST",
        )
        try:
            with _OPENER.open(req, timeout=120) as resp:
                data = json.loads(resp.read().decode())
            msg = data["choices"][0]["message"]
            content = msg.get("content") or ""
            if not content.strip():
                last_err = "模型返回空内容（可能 max_tokens 被推理过程耗尽）"
                if attempt < retries:
                    payload["max_tokens"] = min(payload["max_tokens"] * 2, 8000)
                    continue
                break
            return content
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="ignore")[:300]
            last_err = f"HTTP {e.code}: {body}"
            if e.code in (429, 500, 502, 503) and attempt < retries:
                time.sleep(2 * (attempt + 1))
                continue
            break
        except Exception as e:
            last_err = str(e)
            if attempt < retries:
                time.sleep(2)
                continue
    raise RuntimeError(f"LLM 调用失败: {last_err}")


def chat_json(messages, temperature=0.6, max_tokens=3000, attempts=2):
    """带 JSON 兜底解析：模型偶尔会包 ```json，或者被 max_tokens 截断。

    截断是真会发生的——推理模型会先花掉一部分 budget 在 reasoning 上。所以失败时
    不重试原参数，而是把预算翻倍再要一次。
    """
    last_err = None
    for i in range(attempts):
        raw = chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens * (i + 1),
            json_mode=True,
        )
        text = (raw or "").strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            if text.endswith("```"):
                text = text[:-3]
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            last_err = e
            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    pass
            continue
    raise RuntimeError(f"模型输出不是合法 JSON：{last_err}")

