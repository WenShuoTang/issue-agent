# 图片素材

README 使用的图片都在 `docs/assets/`，已针对 GitHub 渲染做过尺寸与体积优化。

| 文件 | 用途 | 尺寸 |
|---|---|---|
| `architecture.svg` | 架构图（文字精确、可编辑） | 1600×900 |
| `social-preview.png` | README 顶部横幅 / GitHub Social Preview | 1280×640 |
| `ui-preview.png` | 前端界面示意 | 1600×800 |
| `logo-256.png` | Logo | 256 |

## 设计规范

- 背景 `#0B1220`，卡片 `#101B2B`，描边 `#26374D`
- 主色薄荷青 `#4FD1B5`，辅助雾蓝 `#7FD8FF`，警示琥珀 `#F0B37E`
- 字体：`Segoe UI / system-ui`，标题 700，正文 400
- 扁平、细描边、无 3D、无照片、无霓虹、无大面积渐变

## 修改与重绘

- **架构图**：直接编辑 `docs/assets/architecture.svg`（纯文本，改完即用）；
- **Social Preview**：改完后上传到仓库 **Settings → Social preview**；
- 需要重新生成位图时，用下面的提示词，并保持上面的配色与尺寸。

```text
为开源项目「Issue Triage Agent」绘制极简深色架构图，16:9，背景 #0B1220，四周留白 ≥8%。
6 层自上而下：Web UI（Provider · API Key BYOK · Repo/Issue · Streaming Trace · Cited Report）
→ API（FastAPI · Auth · Tenant · Rate Limit · SSE）→ Orchestration（Ingest→Plan→Act→Assess
→Compose→Verify→Finalize，Act⇄Assess 与 Verify⇄Compose 用虚线循环，旁路 Degrade）
→ Capabilities（Tool Registry / Hybrid RAG / Memory / Observability）
→ Models（DeepSeek / OpenAI / Claude / Qwen / Ollama）→ Data（GitHub API · SQLite）。
左侧 Guardrails 竖栏：Prompt-Injection Defense / Secret Redaction / Citation Whitelist / Read-only。
圆角卡片、1px 描边、薄荷青强调，禁止渐变块、3D、人物、水印；只用英文短标签。
```

```text
深色 SaaS 界面概念图，1600×1000，三栏布局：左栏 Repository / Issue / Model Provider / GitHub Token /
API Key (BYOK, not stored) / Analyze；中栏 Workflow 时间线 7 个节点；
右栏 Triage Report，两条带 LOW/MEDIUM 标签与 [1][2] 引用角标的结论，
底部指标胶囊 Rounds / Tools / Tokens / Latency。扁平、圆角、薄荷青强调，文字清晰，无人物水印。
```

```text
GitHub 社交封面 1280×640，深色 #0B1220，左侧文字 Issue Triage Agent /
Agentic RAG for GitHub Issues / Read-only tools · Cited answers · Multi-provider LLM，
右侧放大镜 + 分支图标，三个胶囊 Plan / Act / Verify，右下角 MIT License。
极简、扁平、无人物、无复杂纹理，元素 ≤12。
```
