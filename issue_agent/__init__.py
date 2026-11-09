"""只读 GitHub Issue 排障 Agent。

分层：config/budget/cache → security → llm/tools/rag/memory → agent(graph) → api。
"""
import warnings

# 第三方库的弃用提示会干扰 CLI/测试输出，且与本项目行为无关，定向静音。
warnings.filterwarnings("ignore", message=".*allowed_objects.*")
warnings.filterwarnings("ignore", message=".*httpx2.*")

__version__ = "1.0.0"
