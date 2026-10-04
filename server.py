"""610 考研工作台 · 后端服务

用 Python 标准库 http.server 实现，不依赖任何 Web 框架。
路由：
  GET  /                  → web/index.html
  GET  /static/<file>     → 静态资源
  GET  /api/topics        → 主题列表 [{name, count}, ...]
  GET  /api/health        → 健康检查 / 建库状态
  POST /api/run           → 跑一轮 Agent，以 SSE 流式返回事件

密钥设计：API Key 由前端放在请求体里传来，服务端不保存、不写日志。
云端部署同样不带任何密钥——每人用自己的 Key，额度各自结算。
"""
import os
import sys
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import kb_core as kb

WEB_DIR = os.path.join(HERE, "web")
PORT = int(os.environ.get("PORT", "8000"))
HOST = os.environ.get("HOST", "0.0.0.0")

_init_lock = threading.Lock()

_init_lock = threading.Lock()


def ensure_ready(embed_key=None):
    """确保知识库已就绪。有向量缓存时无需任何 Key。"""
    if kb.READY:
        return True
    with _init_lock:
        if kb.READY:
            return True
        return kb.init_knowledge(embed_key, verbose=True)


# ===== 启动时先尝试用缓存加载（不需要 Key）=====
print("=" * 58)
print("610 考研工作台 启动中……")
if ensure_ready(None):
    print("知识库就绪：%d 块 / %d 个主题" % (len(kb.CHUNKS), len(kb.TOPIC_COUNT)))
else:
    print("未能从缓存加载知识库（缺少 vectors.json 或已过期）")
    print("→ 首次请求会带上 Key 时自动重建，请耐心等待")
print("=" * 58)


MIME = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".ico": "image/x-icon",
}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        # 只记录路径与方法，不记录请求体（避免密钥泄漏到日志）
        sys.stderr.write("[%s] %s\n" % (self.command, self.path))

    # ---------- 工具方法 ----------
    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False))

    def _read_body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            return {}
        raw = self.rfile.read(n)
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return {}

    def _serve_file(self, path):
        if not os.path.isfile(path):
            self._send(404, "not found", "text/plain; charset=utf-8")
            return
        ext = os.path.splitext(path)[1].lower()
        with open(path, "rb") as f:
            data = f.read()
        self._send(200, data, MIME.get(ext, "application/octet-stream"))

    # ---------- 路由 ----------
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0]

        if path in ("/", "/index.html"):
            self._serve_file(os.path.join(WEB_DIR, "index.html"))
            return

        if path.startswith("/static/"):
            rel = path[len("/static/"):]
            safe = os.path.normpath(rel).replace("\\", "/")
            if safe.startswith(".."):
                self._send(403, "forbidden", "text/plain")
                return
            self._serve_file(os.path.join(WEB_DIR, safe))
            return

        if path == "/api/health":
            self._json({
                "ready": kb.READY,
                "chunks": len(kb.CHUNKS),
                "topics": len(kb.TOPIC_COUNT),
            })
            return

        if path == "/api/topics":
            self._json({"topics": [{"name": t, "count": n} for t, n in kb.topics()]})
            return

        self._send(404, "not found", "text/plain; charset=utf-8")

    def do_POST(self):
        path = self.path.split("?")[0]

        # ===== Key 体检接口：只做格式与连通性检查，不消耗对话额度 =====
        if path == "/api/checkkey":
            body = self._read_body()
            keys = body.get("keys") or {}
            self._json(kb.check_keys(keys))
            return

        if path != "/api/run":
            self._send(404, "not found", "text/plain; charset=utf-8")
            return

        body = self._read_body()
        keys = body.get("keys") or {}
        messages = body.get("messages") or []
        system_kind = body.get("system") or "default"

        if not keys.get("deepseek") or not keys.get("siliconflow"):
            self._json({"error": "缺少 API Key"}, 400)
            return

        # 知识库还没就绪（缓存缺失）→ 用这次的 Key 现场建库
        if not kb.READY:
            ok = ensure_ready(keys.get("siliconflow"))
            if not ok:
                self._json({"error": "知识库加载失败"}, 500)
                return

        if system_kind == "quiz":
            system = kb.QUIZ_PROMPT
        elif system_kind == "expand":
            system = kb.EXPAND_PROMPT
        else:
            system = kb.SYSTEM_PROMPT

        # —— SSE 流式返回 ——
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Connection", "close")
        self.end_headers()

        history = list(messages)
        # 论点扩写模式：开启篇幅自动校验与重写
        is_expand = (system_kind == "expand")
        try:
            for ev in kb.run_agent(history, keys, system=system,
                                  auto_fix=is_expand, max_fix=2):
                payload = json.dumps(ev, ensure_ascii=False)
                self.wfile.write(("data: %s\n\n" % payload).encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            payload = json.dumps({"type": "error", "message": "服务端异常：%s" % e},
                                 ensure_ascii=False)
            try:
                self.wfile.write(("data: %s\n\n" % payload).encode("utf-8"))
                self.wfile.flush()
            except Exception:
                pass

        try:
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except Exception:
            pass


def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print("服务已启动： http://127.0.0.1:%d" % PORT)
    print("局域网内可用手机访问： http://<本机IP>:%d" % PORT)
    print("按 Ctrl+C 停止")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
        server.server_close()


if __name__ == "__main__":
    main()