"""问答链（Chain）：RAG 的最后一步 —— 检索 + 拼提示词 + 调大模型生成答案。

完整流水线：
    用户问题 → 向量检索 top-k 相关切片 → 拼成带编号的上下文
    → 连同历史消息一起发给大模型 → 流式返回答案 → 附上引用来源
"""
from collections.abc import Generator
from openai import OpenAI
from app import config
from app.rag import history
from app.rag.store import get_store

# 系统提示词：给模型"立规矩"，是控制幻觉（不瞎编）的第一道防线
SYSTEM_PROMPT = """你是企业内部知识库助手。规则：
1. 只能根据下面【参考资料】回答，参考资料里没有的信息，明确说"知识库中未找到相关内容"，禁止编造。
2. 回答中引用资料时，用 [资料1]、[资料2] 这样的标注指明出处。
3. 回答用简体中文，条理清晰，必要时分点。
"""


class RagChain:
    """RAG 问答链。stream_answer() 是生成器，逐段吐出答案供 SSE 流式推送。"""

    def __init__(self) -> None:
        if not config.LLM_API_KEY:
            raise RuntimeError("请设置环境变量 DEEPSEEK_API_KEY（或 LLM_API_KEY）")
        self._client = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)
        self._store = get_store()

    def _build_context(self, hits: list[dict]) -> str:
        """把检索到的切片拼成编号上下文，编号就是回答里的引用标记。"""
        blocks = []
        for i, h in enumerate(hits, start=1):
            blocks.append(f"【资料{i}】（来源：{h['source']} 第{h['page']}页）\n{h['content']}")
        return "\n\n".join(blocks)

    def stream_answer(self, question: str, session_id: str, top_k: int) -> Generator[dict, None, None]:
        """流式问答。yield 两种事件：
        - {"type": "sources", "data": [引用列表]}   最先返回，前端先展示来源
        - {"type": "content", "data": "答案片段"}    模型逐字输出
        """
        # 1. 检索：拿问题去向量库找最相关的切片
        hits = self._store.search(question, top_k=top_k)

        # 2. 检索为空：直接兜底，不浪费 token 调模型
        if not hits:
            yield {"type": "sources", "data": []}
            yield {"type": "content", "data": "知识库中未找到相关内容，请先上传相关文档。"}
            history.append_message(session_id, "user", question)
            history.append_message(session_id, "assistant", "知识库中未找到相关内容")
            return

        # 3. 先把引用来源推给前端（用户等待时能先看看答案出自哪）
        yield {
            "type": "sources",
            "data": [
                {"source": h["source"], "page": h["page"], "score": h["score"]} for h in hits
            ],
        }

        # 4. 组装 messages：系统提示 + 参考资料 + 历史消息 + 当前问题
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + "\n\n【参考资料】\n" + self._build_context(hits)},
            *history.get_history(session_id),  # 多轮记忆：把最近几轮对话带上
            {"role": "user", "content": question},
        ]

        # 5. 流式调用大模型：stream=True 时返回迭代器，来一个 token 吐一个
        stream = self._client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=messages,
            temperature=0.3,  # 知识库问答要稳不要浪，温度调低
            stream=True,
        )
        full_answer = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                full_answer += delta
                yield {"type": "content", "data": delta}

        # 6. 落库：把这一轮问答存进会话历史，下一轮就能"记得"
        history.append_message(session_id, "user", question)
        history.append_message(session_id, "assistant", full_answer)


# 模块级单例（懒加载：第一次问答时才检查 API Key）
_chain: RagChain | None = None


def get_chain() -> RagChain:
    global _chain
    if _chain is None:
        _chain = RagChain()
    return _chain
