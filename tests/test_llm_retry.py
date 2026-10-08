# -*- coding: utf-8 -*-
"""LLM 调用重试机制测试

覆盖 eva.py 中新增的重试层：_open_llm / _with_retry / _backoff_delay / _set_socket_timeout
以及 llm_chat、llm_chat_stream 两个入口。

运行方式：
    python tests/test_llm_retry.py
    python -m unittest tests.test_llm_retry
"""
import io
import json
import os
import re
import runpy
import sys
import tempfile
import unittest
import urllib.error
import warnings
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

EVA_FILE = Path(__file__).resolve().parents[1] / "eva.py"

# 可选依赖（try/except 包裹，缺失也能运行），允许出现在 import 中
OPTIONAL_DEPENDENCIES = {"prompt_toolkit", "rich"}

# 重试配置区期望的默认值（直接配置的常量）
EXPECTED_DEFAULTS = {
    "EVA_MAX_RETRY": 5,
    "EVA_RETRY_BASE": 1.0,
    "EVA_RETRY_FACTOR": 4.0,
    "EVA_RETRY_MAX_DELAY": 32.0,
    "EVA_CONNECT_TIMEOUT": 10.0,
    "EVA_READ_TIMEOUT": 300.0,
}


class FakeModelsResponse:
    """detect_model_len() 拉取 /models 用"""
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return b'{"data":[{"id":"deepseek-flash","max_model_len":100000}]}'


class CtxResponse:
    """非流式响应"""
    status = 200

    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return self._payload

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class StreamResponse:
    """流式响应：可迭代吐出 SSE 行，并支持在指定位置抛错"""
    def __init__(self, lines, fail_after=None, err=None):
        self.lines = lines
        self.fail_after = fail_after
        self.err = err
        self._i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.fail_after is not None and self._i >= self.fail_after:
            raise self.err
        if self._i >= len(self.lines):
            raise StopIteration
        self._i += 1
        return self.lines[self._i - 1]

    def close(self):
        pass


def sse(*deltas):
    lines = [b"data: " + json.dumps({"choices": [{"delta": d}]}).encode() + b"\n"
             for d in deltas]
    lines.append(b"data: [DONE]\n")
    return lines


CHAT_JSON = json.dumps({
    "choices": [{"message": {"role": "assistant", "content": "hello"}}],
    "usage": {"total_tokens": 5},
}).encode()


def http_error(code, retry_after=None):
    headers = {"Retry-After": retry_after} if retry_after is not None else {}
    return urllib.error.HTTPError("http://x", code, "err", headers,
                                  io.BytesIO(b'{"error":"boom"}'))


def raise_from(exc):
    """返回一个调用即抛出 exc 的函数（用于构造失败响应）"""
    def _f():
        raise exc
    return _f


def load_eva(extra_env=None, urlopen=None):
    """在临时目录、假 /models 响应下加载 eva.py，返回其模块命名空间。
    返回的 dict 即为函数们的 __globals__，可直接改写常量来控制重试行为。
    urlopen 可传入自定义的假实现（默认返回可用的 /models 响应）。"""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        project_dir = temp_path / "project"
        project_dir.mkdir()
        old_cwd = os.getcwd()
        env = {
            "EVA_API_KEY": "test",
            "EVA_BASE_URL": "https://example.invalid/v1",
            "EVA_MODEL_NAME": "deepseek-flash",
            "EVA_HOME": str(temp_path / "eva-home"),
        }
        env.update(extra_env or {})
        fake = urlopen or (lambda req, timeout=None: FakeModelsResponse())
        try:
            os.chdir(project_dir)
            with patch.dict(os.environ, env, clear=False), \
                    patch("urllib.request.urlopen", new=fake):
                return runpy.run_path(str(EVA_FILE))
        finally:
            os.chdir(old_cwd)


class LlmRetryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = load_eva()
        # 注意：runpy.run_path 返回的是 globals 的副本，改它不会影响已定义的函数；
        # 函数真正使用的全局命名空间要从 __globals__ 取（常量读写都用它）。
        cls.g = cls.ns["llm_chat"].__globals__
        # 在 setUp 改常量之前，记录出厂默认值
        cls.defaults = {k: cls.g[k] for k in EXPECTED_DEFAULTS}

    def setUp(self):
        # 缩短退避，避免测试真的等待（这些是模块常量，函数在调用时读取，可直接改）
        self.g["EVA_MAX_RETRY"] = 4
        self.g["EVA_RETRY_BASE"] = 0.001
        self.g["EVA_RETRY_MAX_DELAY"] = 0.01

    def _patch_chat(self, behaviors):
        """依次把 behaviors 作为 /chat/completions 的响应工厂。
        返回 (patcher, state)，state["calls"] 记录实际发起的 chat 请求次数。"""
        state = {"i": 0, "calls": 0}

        def fake(req, timeout=None):
            url = getattr(req, "full_url", str(req))
            if url.endswith("/models"):
                return FakeModelsResponse()
            behavior = behaviors[min(state["i"], len(behaviors) - 1)]
            state["i"] += 1
            state["calls"] += 1
            return behavior()

        return patch("urllib.request.urlopen", fake), state

    # ---------- 配置与依赖约束 ----------

    def test_defaults_are_expected_constants(self):
        self.assertEqual(self.defaults, EXPECTED_DEFAULTS)

    def test_retry_params_not_read_from_env(self):
        """重试参数应为直接配置的常量，不受环境变量影响"""
        ns = load_eva(extra_env={
            "EVA_MAX_RETRY": "999",
            "EVA_RETRY_BASE": "99",
            "EVA_RETRY_FACTOR": "9",
            "EVA_RETRY_MAX_DELAY": "1",
        })
        g = ns["llm_chat"].__globals__
        self.assertEqual(g["EVA_MAX_RETRY"], EXPECTED_DEFAULTS["EVA_MAX_RETRY"])
        self.assertEqual(g["EVA_RETRY_BASE"], EXPECTED_DEFAULTS["EVA_RETRY_BASE"])
        self.assertEqual(g["EVA_RETRY_FACTOR"], EXPECTED_DEFAULTS["EVA_RETRY_FACTOR"])
        self.assertEqual(g["EVA_RETRY_MAX_DELAY"], EXPECTED_DEFAULTS["EVA_RETRY_MAX_DELAY"])

    def test_only_stdlib_imports(self):
        """eva 只允许使用内置库（可选依赖除外）"""
        src = EVA_FILE.read_text(encoding="utf-8")
        mods = set()
        for line in src.splitlines():
            m = re.match(r"\s*(?:import|from)\s+([A-Za-z0-9_.]+)", line)
            if m:
                mods.add(m.group(1).split(".")[0])
        third_party = {m for m in mods
                       if m not in sys.stdlib_module_names
                       and m not in OPTIONAL_DEPENDENCIES}
        self.assertEqual(third_party, set(), f"发现非标准库导入：{third_party}")

    def test_startup_probe_does_not_retry(self):
        """启动期 /models 端点探测必须快速失败，不做任何重试（只请求 1 次）"""
        calls = {"n": 0}

        def failing_models(req, timeout=None):
            calls["n"] += 1
            raise urllib.error.HTTPError("http://x", 500, "err", {},
                                         io.BytesIO(b'{"error":"boom"}'))

        with self.assertRaises(SystemExit), redirect_stdout(io.StringIO()), \
                warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)  # 忽略假 HTTPError 未关闭的告警
            load_eva(urlopen=failing_models)
        self.assertEqual(calls["n"], 1)

    # ---------- 非流式 llm_chat ----------

    def test_chat_retries_then_succeeds(self):
        patcher, state = self._patch_chat([
            raise_from(urllib.error.URLError("reset")),
            raise_from(urllib.error.URLError("reset")),
            lambda: CtxResponse(CHAT_JSON),
        ])
        with patcher:
            msg, usage = self.ns["llm_chat"]([{"role": "user", "content": "hi"}])
        self.assertEqual(msg["content"], "hello")
        self.assertEqual(state["calls"], 3)

    def test_chat_exhausts_retries_and_raises(self):
        patcher, state = self._patch_chat([raise_from(urllib.error.URLError("down"))])
        with patcher:
            with self.assertRaises(self.ns["LLMRetryError"]):
                self.ns["llm_chat"]([{"role": "user", "content": "hi"}])
        self.assertEqual(state["calls"], self.g["EVA_MAX_RETRY"])

    def test_chat_401_is_not_retried(self):
        patcher, state = self._patch_chat([raise_from(http_error(401))])
        with patcher:
            with self.assertRaises(self.ns["LLMError"]):
                self.ns["llm_chat"]([{"role": "user", "content": "hi"}])
        self.assertEqual(state["calls"], 1)

    def test_chat_5xx_is_retried(self):
        patcher, state = self._patch_chat([
            raise_from(http_error(503)),
            lambda: CtxResponse(CHAT_JSON),
        ])
        with patcher:
            msg, _ = self.ns["llm_chat"]([{"role": "user", "content": "hi"}])
        self.assertEqual(msg["content"], "hello")
        self.assertEqual(state["calls"], 2)

    def test_chat_400_is_retried(self):
        """端点偶发抽风返回 400 时应重试"""
        patcher, state = self._patch_chat([
            raise_from(http_error(400)),
            lambda: CtxResponse(CHAT_JSON),
        ])
        with patcher:
            msg, _ = self.ns["llm_chat"]([{"role": "user", "content": "hi"}])
        self.assertEqual(msg["content"], "hello")
        self.assertEqual(state["calls"], 2)

    def test_chat_persistent_400_gives_up(self):
        """持续 400（如请求体字段不被端点支持）最终放弃，不会无限重试"""
        patcher, state = self._patch_chat([raise_from(http_error(400))])
        with patcher:
            with self.assertRaises(self.ns["LLMRetryError"]):
                self.ns["llm_chat"]([{"role": "user", "content": "hi"}])
        self.assertEqual(state["calls"], self.g["EVA_MAX_RETRY"])

    def test_chat_429_honors_retry_after(self):
        patcher, state = self._patch_chat([
            raise_from(http_error(429, retry_after="1")),
            lambda: CtxResponse(CHAT_JSON),
        ])
        with patcher:
            msg, _ = self.ns["llm_chat"]([{"role": "user", "content": "hi"}])
        self.assertEqual(msg["content"], "hello")
        self.assertEqual(state["calls"], 2)

    def test_chat_invalid_json_is_retried(self):
        patcher, state = self._patch_chat([
            lambda: CtxResponse(b"not-json"),
            lambda: CtxResponse(CHAT_JSON),
        ])
        with patcher:
            msg, _ = self.ns["llm_chat"]([{"role": "user", "content": "hi"}])
        self.assertEqual(msg["content"], "hello")
        self.assertEqual(state["calls"], 2)

    # ---------- 流式 llm_chat_stream ----------

    def test_stream_retries_without_duplicating_output(self):
        patcher, state = self._patch_chat([
            raise_from(urllib.error.URLError("reset")),
            raise_from(urllib.error.URLError("reset")),
            lambda: StreamResponse(sse({"content": "Hel"}, {"content": "lo"})),
        ])
        buf = io.StringIO()
        with patcher, redirect_stdout(buf):
            msg, _ = self.ns["llm_chat_stream"]([{"role": "user", "content": "hi"}])
        self.assertEqual(msg["content"], "Hello")
        self.assertEqual(buf.getvalue(), "Hello")  # 未发生重复输出
        self.assertEqual(state["calls"], 3)

    def test_stream_no_retry_after_partial_content(self):
        """已向用户输出正文后再中断：不重试，抛不可重试错误"""
        patcher, state = self._patch_chat([
            lambda: StreamResponse(sse({"content": "par"}, {"content": "tial"}),
                                   fail_after=1, err=OSError("broken pipe")),
        ])
        with patcher, redirect_stdout(io.StringIO()):
            with self.assertRaises(self.ns["LLMError"]):
                self.ns["llm_chat_stream"]([{"role": "user", "content": "hi"}])
        self.assertEqual(state["calls"], 1)

    def test_stream_no_retry_after_reasoning_output(self):
        """think 内容已实时输出后再中断：不重试"""
        patcher, state = self._patch_chat([
            lambda: StreamResponse(sse({"reasoning_content": "th"}),
                                   fail_after=1, err=OSError("reset")),
        ])
        with patcher, redirect_stdout(io.StringIO()):
            with self.assertRaises(self.ns["LLMError"]):
                self.ns["llm_chat_stream"]([{"role": "user", "content": "hi"}])
        self.assertEqual(state["calls"], 1)

    def test_stream_retries_when_nothing_emitted(self):
        patcher, state = self._patch_chat([
            lambda: StreamResponse(sse(), fail_after=0, err=OSError("reset")),
            lambda: StreamResponse(sse({"content": "OK"})),
        ])
        with patcher, redirect_stdout(io.StringIO()):
            msg, _ = self.ns["llm_chat_stream"]([{"role": "user", "content": "hi"}])
        self.assertEqual(msg["content"], "OK")
        self.assertEqual(state["calls"], 2)

    def test_stream_think_repeat_error_propagates(self):
        """ThinkRepeatError 必须原样透传，不能被重试逻辑吞掉"""
        patcher, state = self._patch_chat([
            lambda: StreamResponse(sse({"content": "x"}), fail_after=0,
                                   err=self.ns["ThinkRepeatError"]()),
        ])
        with patcher, redirect_stdout(io.StringIO()):
            with self.assertRaises(self.ns["ThinkRepeatError"]):
                self.ns["llm_chat_stream"]([{"role": "user", "content": "hi"}])
        self.assertEqual(state["calls"], 1)

    def test_stream_applies_read_timeout_after_connect(self):
        """建连后应把底层 socket 超时从 connect 超时放开为 read 超时"""
        recorded = []
        sock = type("Sock", (), {"settimeout": lambda self, t: recorded.append(t)})()
        resp = StreamResponse(sse({"content": "X"}))
        resp.fp = type("Fp", (), {"raw": type("Raw", (), {"_sock": sock})()})()
        patcher, _ = self._patch_chat([lambda: resp])
        with patcher, redirect_stdout(io.StringIO()):
            self.ns["llm_chat_stream"]([{"role": "user", "content": "hi"}])
        self.assertEqual(recorded, [self.g["EVA_READ_TIMEOUT"]])


    def test_backoff_sequence(self):
        """退避序列应为 1、4、16、32（jitter 归零后精确校验）"""
        g = self.g
        saved = {k: g[k] for k in ("EVA_RETRY_BASE", "EVA_RETRY_FACTOR",
                                   "EVA_RETRY_MAX_DELAY")}
        g.update(EVA_RETRY_BASE=1.0, EVA_RETRY_FACTOR=4.0, EVA_RETRY_MAX_DELAY=32.0)
        try:
            with patch("random.uniform", return_value=0.0):
                seq = [self.ns["_backoff_delay"](n) for n in (1, 2, 3, 4, 5)]
        finally:
            g.update(saved)
        self.assertEqual(seq, [1.0, 4.0, 16.0, 32.0, 32.0])

    # ---------- agent 循环状态回报（goal 热循环修复） ----------

    def test_agent_loop_returns_false_on_llm_failure(self):
        """LLM 调用失败时 agent_single_loop 必须返回 False，goal 循环才能停止空转"""
        def boom(*_a, **_k):
            raise self.ns["LLMError"]("HTTP 400: bad request")

        with patch.dict(self.g, {"llm_chat_stream": boom}), \
                redirect_stdout(io.StringIO()):
            self.assertFalse(self.ns["agent_single_loop"]())

    def test_agent_loop_returns_true_on_normal_finish(self):
        """模型给出文字回复（正常结束）时应返回 True"""
        def ok(*_a, **_k):
            return {"role": "assistant", "content": "done"}, {"total_tokens": 1}

        with patch.dict(self.g, {
            "llm_chat_stream": ok,
            "messages": [{"role": "user", "content": "hi"}],
            "COMPACT_PANIC": False,
        }), redirect_stdout(io.StringIO()):
            self.assertTrue(self.ns["agent_single_loop"]())


if __name__ == "__main__":
    unittest.main()
