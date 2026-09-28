"""文本切分：把长文档切成适合 Embedding 的小块（chunk）。

为什么不引 LangChain 的 RecursiveCharacterTextSplitter？
——自己实现一遍，能看清每一步的取舍。

核心思路：递归切分（Recursive Splitting）
按优先级依次尝试分隔符 ["\\n\\n", "\\n", "。", "；", "，", ""]：
能按段落切就按段落切，段落太长就退到句子，句子还太长才硬切。
多级兜底，保证切片尽量不断在半句话中间。
"""
from app import config


def _split_by_separator(text: str, separator: str) -> list[str]:
    """按单个分隔符切，并把分隔符拼回去（保留句号等标点，语义更完整）。"""
    if separator == "":
        # 兜底：没有任何分隔符可用时按固定长度硬切
        return [text[i:i + config.CHUNK_SIZE] for i in range(0, len(text), config.CHUNK_SIZE)]
    parts = text.split(separator)
    # split 会丢掉分隔符本身，这里把分隔符拼回每一段末尾（最后一段除外）
    return [p + separator for p in parts[:-1]] + [parts[-1]]


def _merge_pieces(pieces: list[str]) -> list[str]:
    """把过小的片段合并、过长的片段留给上层继续切，并做重叠（overlap）。"""
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if len(current) + len(piece) <= config.CHUNK_SIZE:
            current += piece  # 还能装下，继续拼
        else:
            if current:
                chunks.append(current)
            current = piece
    if current:
        chunks.append(current)

    # 重叠处理：每个 chunk 带上前一个 chunk 的尾巴，防止关键信息刚好被切断
    if config.CHUNK_OVERLAP > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-config.CHUNK_OVERLAP:]
            overlapped.append(tail + chunks[i])
        return overlapped
    return chunks


def split_text(text: str, separators: list[str] | None = None) -> list[str]:
    """递归切分主函数：返回不超过 CHUNK_SIZE 的文本块列表。"""
    if separators is None:
        separators = ["\n\n", "\n", "。", "；", "，", ""]

    if len(text) <= config.CHUNK_SIZE:
        return [text] if text.strip() else []

    separator = separators[0]
    if separator == "" or separator in text:
        pieces = _split_by_separator(text, separator)
        fine: list[str] = []
        for piece in pieces:
            if len(piece) <= config.CHUNK_SIZE:
                fine.append(piece)
            else:
                # 这一片还是太长，换更细的分隔符递归切
                fine.extend(split_text(piece, separators[1:]))
        # 把过短的片段合并、并按 CHUNK_OVERLAP 做重叠，
        # 否则切出来的块又碎又互不重叠，检索时容易缺上下文
        return [c for c in _merge_pieces(fine) if c.strip()]

    # 当前分隔符不在文本里，直接试下一级
    return split_text(text, separators[1:])


def split_document(doc: dict) -> list[dict]:
    """把 loader 解析出的文档切成带元数据的 chunk 列表。

    每个 chunk 携带的元数据（metadata）是"引用来源"功能的基础：
    - source: 来自哪个文件
    - page:   来自第几页（近似）
    """
    chunks: list[dict] = []
    for page_no, page_text in enumerate(doc["pages"], start=1):
        for piece in split_text(page_text):
            chunks.append({
                "content": piece,
                "metadata": {"source": doc["source"], "page": page_no},
            })
    return chunks
