"""Execution machinery shared by the stages.

- :mod:`r3trace.runtime.llm` — LiteLLM wrapper + structured-output schema enforcement.
- :mod:`r3trace.runtime.codeact` — the generic multi-turn code-execution loop.
- :mod:`r3trace.runtime.python_executor` — vendored sandboxed in-process Python
  interpreter used by the reasoning loop (smolagents, Apache-2.0 — see THIRD-PARTY-NOTICES).
"""
