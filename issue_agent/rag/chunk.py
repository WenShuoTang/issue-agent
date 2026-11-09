"""Markdown 感知的切块。

为什么不用固定 512 token 窗口：Issue 文本里「错误栈」和「复现步骤」被切断是最常见的
检索质量事故。这里按标题/段落边界切，尽量不把代码块切开，并保留 100 字符重叠，
让跨块的句子仍有机会命中。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_HEADING = re.compile(r"^(#{1,6})\s+(.+)$")
_FENCE = re.compile(r"^\s*```")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    url: str
    kind: str
    order: int
    start: int = 0
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"chunk_id": self.chunk_id, "doc_id": self.doc_id, "title": self.title,
                "url": self.url, "kind": self.kind, "order": self.order,
                "chars": len(self.text)}


def _split_blocks(text: str) -> list[tuple[str, str]]:
    """按标题分段，返回 [(heading, body)]；代码块内部不会被当作标题。"""
    blocks: list[tuple[str, list[str]]] = [("", [])]
    in_fence = False
    for line in text.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
        heading = _HEADING.match(line) if not in_fence else None
        if heading and len(blocks[-1][1]) > 0:
            blocks.append((heading.group(2).strip(), []))
            continue
        blocks[-1][1].append(line)
    return [(title, "\n".join(lines).strip()) for title, lines in blocks if "\n".join(lines).strip()]


def chunk_document(doc_id: str, title: str, text: str, url: str, kind: str, *,
                   max_chars: int = 1000, overlap: int = 100) -> list[Chunk]:
    chunks: list[Chunk] = []
    if not text:
        return chunks
    position = 0
    for section_title, body in _split_blocks(text):
        # 超长代码块/日志先按行切，避免单块爆掉预算。
        pieces: list[str] = []
        if len(body) <= max_chars:
            pieces = [body]
        else:
            lines = body.splitlines()
            buffer: list[str] = []
            size = 0
            for line in lines:
                if size + len(line) + 1 > max_chars and buffer:
                    pieces.append("\n".join(buffer))
                    # 重叠：保留最后若干行，保证跨块上下文不断裂。
                    tail = "\n".join(buffer)[-overlap:]
                    buffer = [tail] if tail else []
                    size = len(tail)
                buffer.append(line)
                size += len(line) + 1
            if buffer:
                pieces.append("\n".join(buffer))

        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            label = section_title or title
            chunks.append(Chunk(
                chunk_id=f"{doc_id}#c{len(chunks) + 1}",
                doc_id=doc_id,
                title=label[:180],
                text=piece[: max_chars + overlap],
                url=url,
                kind=kind,
                order=len(chunks),
                start=position,
                meta={"section": section_title or ""},
            ))
            position += len(piece)
    return chunks
