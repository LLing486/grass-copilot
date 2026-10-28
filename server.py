"""本地/服务器入口：stdlib HTTP + SSE，零第三方依赖。

为什么不用 FastAPI：这个 demo 要能在他自己的机器、面试现场、云服务器上三种环境
以同一条命令跑起来，少一个依赖就少一次现场翻车。
"""
import json
import mimetypes
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine import pipeline, retriever  # noqa: E402

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
PORT = int(os.environ.get("PORT", "8017"))

CASES = [
    {
        "id": "aw",
        "name": "秋冬上新 · 干皮救急",
        "desc": "正常的节点推广 brief",
        "brief": "澄野氨基酸沐浴油，秋冬推一波。想打北方干皮、洗完腿上起白屑的人。主推三油配方和无香精版，别碰可卸防晒那条因为没报告。要有用过的真实感，不要一看就是广告。",
    },
    {
        "id": "rush",
        "name": "换季冲销量 · 要爆款",
        "desc": "快节奏、要求「够爆」——最容易写出一批同质化套话的那种需求",
        "brief": "双十一要冲一波销量，三天内出稿。要够爆、要网感，标题一定要抓眼球，多用点种草语气，年轻人爱看的那种。产品主推三油配方，其他你看着写。",
    },
    {
        "id": "risk",
        "name": "客户要求 · 违规版",
        "desc": "客户点名要「修复屏障、七天见效、全网最温和」，看闸门怎么处理",
        "brief": "客户说上次的稿子卖点不够狠，这次要突出「能修复皮肤屏障、七天见效」，标题里用「全网最温和」这个说法，说这样转化高。产品是三油配方，无香精版也要提。尽量配合客户要求。",
    },
]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    # ---------------------------------------------------------------- helpers
    def _json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path):
        if not path.exists():
            self._json({"error": "not found"}, 404)
            return
        body = path.read_bytes()
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if path.suffix in (".html", ".js", ".css"):
            ctype += "; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ---------------------------------------------------------------- routes
    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            return self._file(WEB / "index.html")
        if path == "/api/meta":
            data = retriever.load_assets()
            return self._json(
                {
                    "brand": data["brand"],
                    "cases": CASES,
                    "assets": [
                        {"id": a["id"], "type": a["type"], "title": a["title"], "evidence": a.get("evidence")}
                        for a in data["assets"]
                    ],
                    "rules": [
                        {"id": r["id"], "name": r["name"], "level": r["level"], "basis": r["basis"], "n": len(r["patterns"])}
                        for r in data["compliance_rules"]
                    ],
                }
            )
        return self._file(WEB / path.lstrip("/"))

    def do_POST(self):
        if self.path.split("?")[0] != "/api/run":
            return self._json({"error": "not found"}, 404)
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length) or b"{}")
        brief = (payload.get("brief") or "").strip()
        if len(brief) < 10:
            return self._json({"error": "brief 太短"}, 400)

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        self.end_headers()

        def send(event, data):
            chunk = f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
            try:
                self.wfile.write(chunk.encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

        def on_step(item):
            send("step", item)

        try:
            started = time.time()
            result = pipeline.run(brief, on_step=on_step)
            result["elapsed"] = round(time.time() - started, 1)
            for n in result["notes"]:
                send(
                    "note",
                    {
                        "structure": n.get("structure"),
                        "title": n["title"],
                        "chars": n["metrics"]["chars"],
                    },
                )
            send("result", result)
            send("done", {"ok": True})
        except Exception as e:  # 现场演示不能白屏
            send("error", {"message": f"{type(e).__name__}: {e}"})
        self.close_connection = True


def main():
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"种草流水线 running → http://localhost:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
