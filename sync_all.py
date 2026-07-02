#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""无界面（headless）每日数据同步脚本。

复用 etf_app.py 里 UI 之前的纯数据/同步函数，逐个标的调用 sync_target_data()
把最新行情写入 Supabase。用于 GitHub Actions 定时任务，也可本地手动运行。

工作原理：
  etf_app.py 在导入时就会调用 streamlit（st.secrets / cache 装饰器 / st.warning 等），
  且文件下半部分是直接执行的 Streamlit UI 脚本体。这里做两件事：
    1) 注入一个 streamlit「垫片」到 sys.modules —— secrets 从环境变量读取，
       其余 UI 调用全部变成 no-op / 打印。
    2) 只加载 etf_app.py 中「# ─── Streamlit UI」标记之前的部分（纯函数），
       避免执行任何 UI 代码。

本地运行：
    DATABASE_URL_POOLER="postgresql://..." python sync_all.py

环境变量：
    DATABASE_URL_POOLER  Supabase 连接池 URI（推荐，IPv4 可达）
    DATABASE_URL         直连 URI（可选，作为兜底）
    TICKFLOW_API_KEY     可选，不填则用 TickFlow 免费额度
"""
import os
import sys
import time
import types


# ── 1. Streamlit 垫片 ────────────────────────────────────────────────────────
class _Secrets:
    """把 st.secrets.get('database_url_pooler') 映射到环境变量 DATABASE_URL_POOLER。"""

    def get(self, key, default=None):
        return os.environ.get(str(key).upper(), default)

    def __getitem__(self, key):
        return os.environ[str(key).upper()]


def _cache(*dargs, **dkwargs):
    """兼容 @st.cache_data 与 @st.cache_data(ttl=..., show_spinner=...) 两种写法。"""
    if len(dargs) == 1 and callable(dargs[0]) and not dkwargs:
        return dargs[0]

    def deco(fn):
        return fn

    return deco


_REPORTERS = {"warning", "error", "info", "caption", "success", "write", "toast", "exception"}


def _reporter(prefix):
    def emit(*args, **kwargs):
        msg = " ".join(str(a) for a in args)
        print(f"  [st.{prefix}] {msg}", flush=True)

    return emit


class _StreamlitShim(types.ModuleType):
    secrets = _Secrets()
    cache_data = staticmethod(_cache)
    cache_resource = staticmethod(_cache)

    def __getattr__(self, name):
        # st.warning/error/... 打印到日志；其余（title/columns/button/sidebar…）静默 no-op
        if name in _REPORTERS:
            return _reporter(name)
        return lambda *a, **k: None


sys.modules["streamlit"] = _StreamlitShim("streamlit")


# ── 2. 加载 etf_app.py 中 UI 之前的纯函数部分 ────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
APP_PATH = os.path.join(HERE, "etf_app.py")
UI_MARKER = "# ─── Streamlit UI"  # 与 etf_app.py 中的分节注释一致

with open(APP_PATH, encoding="utf-8") as f:
    _src = f.read()

_cut = _src.find(UI_MARKER)
if _cut == -1:
    raise RuntimeError(
        f"未在 {APP_PATH} 中找到 UI 分节标记，为避免误跑 Streamlit UI 而中止。"
    )

core = types.ModuleType("etf_core")
core.__file__ = APP_PATH
exec(compile(_src[:_cut], APP_PATH, "exec"), core.__dict__)


# ── 3. 逐个标的同步 ──────────────────────────────────────────────────────────
MAX_ATTEMPTS = 3   # 上游数据源（TickFlow/Yahoo）偶发断连，单标的重试几次
RETRY_SLEEP = 5    # 每次重试间隔（秒）


def _sync_one(index_code):
    """同步单个标的，失败自动重试。返回落库行数；重试用尽则抛最后一次异常。"""
    last_err = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            _, _, written = core.sync_target_data(index_code)
            return int(written or 0), attempt
        except Exception as e:  # noqa: BLE001  含上游网络抖动，可重试
            last_err = e
            if attempt < MAX_ATTEMPTS:
                print(f"  ↻ {index_code} 第 {attempt}/{MAX_ATTEMPTS} 次失败：{e}，{RETRY_SLEEP}s 后重试",
                      flush=True)
                time.sleep(RETRY_SLEEP)
    raise last_err


def main():
    if not getattr(core, "DATABASE_URL", None):
        print("ERROR: 未配置数据库连接串，请设置环境变量 DATABASE_URL_POOLER（或 DATABASE_URL）。",
              file=sys.stderr)
        return 1

    targets = core.load_targets_from_db()
    if not targets:
        print("ERROR: etf_targets 中没有任何标的。", file=sys.stderr)
        return 1

    print(f"共 {len(targets)} 个标的，开始同步……", flush=True)
    failures = []
    total_written = 0
    for name, cfg in targets.items():
        index_code = cfg.get("index_code")
        try:
            written, attempt = _sync_one(index_code)
            total_written += written
            suffix = f"（第 {attempt} 次尝试）" if attempt > 1 else ""
            print(f"OK   {name} ({index_code})：落库 {written} 条{suffix}", flush=True)
        except Exception as e:  # noqa: BLE001  单个标的失败不影响其余标的
            failures.append((name, index_code))
            print(f"FAIL {name} ({index_code})：{e}", flush=True)

    ok = len(targets) - len(failures)
    print(f"\n完成：{ok}/{len(targets)} 成功，累计落库 {total_written} 条。", flush=True)
    if failures:
        print("失败标的：" + ", ".join(f"{n}({c})" for n, c in failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
