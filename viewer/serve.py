#!/usr/bin/env python3
"""
serve.py — 给数据查看器提供本地文件浏览能力.

零依赖(仅标准库)。启动后把 DataFlow-MemTensor 仓库目录作为站点根,
额外提供 /api/list 列出所有 .jsonl 文件,让 viewer/index.html 能在
左侧列出目录树、点击直接加载数据(无需拖拽)。

用法:
    python viewer/serve.py          # 默认端口 8000,自动打开浏览器
    python viewer/serve.py 8080     # 指定端口
"""
import json
import os
import sys
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from functools import partial

# 仓库根 = 本文件的上一级目录
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# 扫描这些目录下的 jsonl(相对仓库根);None 表示扫全仓库
SCAN_DIRS = ["data", "cache_cot", "cache_evidence", "cache_interleaved", "viewer", "."]
SKIP_DIRS = {".git", "__pycache__", "node_modules"}
MAX_MB = 200  # 超过则不列(避免误开超大文件)


def list_jsonl():
    seen = set()
    out = []
    for base in SCAN_DIRS:
        root = os.path.join(REPO_ROOT, base)
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fn in filenames:
                if not fn.endswith((".jsonl", ".json")):
                    continue
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, REPO_ROOT)
                if rel in seen:
                    continue
                seen.add(rel)
                try:
                    sz = os.path.getsize(full)
                except OSError:
                    continue
                if sz > MAX_MB * 1024 * 1024:
                    continue
                out.append({
                    "path": rel.replace(os.sep, "/"),
                    "dir": os.path.dirname(rel).replace(os.sep, "/") or ".",
                    "name": fn,
                    "size": sz,
                })
    out.sort(key=lambda x: (x["dir"], x["name"]))
    return out


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/api/list"):
            payload = json.dumps({"root": REPO_ROOT, "files": list_jsonl()},
                                 ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        return super().do_GET()

    def log_message(self, fmt, *args):
        pass  # 静默


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    handler = partial(Handler, directory=REPO_ROOT)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    url = f"http://127.0.0.1:{port}/viewer/index.html"
    n = len(list_jsonl())
    print(f"[serve] 站点根: {REPO_ROOT}")
    print(f"[serve] 已发现 {n} 个 jsonl 文件")
    print(f"[serve] 打开: {url}")
    print("[serve] Ctrl+C 停止")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] 已停止")


if __name__ == "__main__":
    main()
