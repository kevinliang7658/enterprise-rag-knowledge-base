"""Embedding：把文本变成向量（一串浮点数），让计算机能算"语义相似度"。

给每段文字算一个"语义坐标"，
含义相近的文字坐标距离近 —— 检索时就变成"找最近的 K 个邻居"。

- siliconflow（默认）：调在线 Embedding API（现指向阿里云百炼 text-embedding-v4），不用下载模型，适合开发

对外只暴露一个 embed() 方法，上层无感知。
"""
from openai import OpenAI

from app import config


class Embedder:
    """双后端 Embedding 封装。用法：Embedder().embed(["文本1", "文本2"]) -> [[0.1, ...], ...]"""

    def __init__(self) -> None:
        self.provider = config.EMBEDDING_PROVIDER
        if self.provider == "siliconflow":
            if not config.SILICONFLOW_API_KEY:
                raise RuntimeError("使用千问 Embedding 需要设置环境变量 QWEN_API_KEY")
            # 该端点是 OpenAI 兼容协议，直接用 openai SDK 换 base_url 即可
            self._client = OpenAI(
                base_url=config.SILICONFLOW_BASE_URL,
                api_key=config.SILICONFLOW_API_KEY,
            )
        elif self.provider == "local":
            # 延迟 import：不装 sentence-transformers 也能用 siliconflow 模式
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(config.LOCAL_EMBEDDING_MODEL)
        else:
            raise ValueError(f"未知的 EMBEDDING_PROVIDER：{self.provider}")

    def embed(self, texts: list[str]) -> list[list[float]]:
        """把一批文本转成向量。批量调用比逐条调快得多（类似 JDBC batch）。"""
        if not texts:
            return []
        if self.provider == "siliconflow":
            all_vecs = []
            for i in range(0, len(texts), 10):   # 千问单次最多10条，分批发
                batch = texts[i:i + 10]
                resp = self._client.embeddings.create(model=config.EMBEDDING_MODEL, input=batch)
                data = sorted(resp.data, key=lambda d: d.index)
                all_vecs.extend(d.embedding for d in data)
            return all_vecs
        # local 模式：numpy 数组转普通 list，方便 JSON 序列化
        return [vec.tolist() for vec in self._model.encode(texts)]


# 模块级单例：全进程共用，避免重复建连/加载模型（相当于 Spring 的单例 Bean）
_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder
