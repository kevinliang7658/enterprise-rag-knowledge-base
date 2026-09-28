"""向量库封装：默认 Chroma（免安装、数据存本地文件），接口与 Milvus 对齐。

Chroma 相当于 H2/SQLite（嵌入式、开箱即用），
Milvus 相当于 MySQL（独立服务、生产级）。本类的四个方法是对外接口，
将来切 Milvus 只需按同样签名实现一个 MilvusStore（详见 README 的切换说明）。
"""
import uuid

import chromadb

from app import config
from app.rag.embedder import get_embedder


class ChromaStore:
    """Chroma 向量库：负责 chunk 的写入、相似度检索、按来源删除、列表统计。"""

    COLLECTION_NAME = "knowledge_base"  # 集合名，相当于数据库里的一张表

    def __init__(self) -> None:
        # PersistentClient：数据落盘到 CHROMA_DIR，重启不丢
        self._client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},  # 用余弦相似度，文本检索的常规选择
        )

    def add_chunks(self, chunks: list[dict]) -> int:
        """把切分好的 chunk 批量写入向量库，返回写入条数。"""
        if not chunks:
            return 0
        texts = [c["content"] for c in chunks]
        embeddings = get_embedder().embed(texts)  # 批量算向量
        self._collection.add(
            ids=[str(uuid.uuid4()) for _ in chunks],  # 每条一个唯一 ID，相当于主键
            documents=texts,
            embeddings=embeddings,
            metadatas=[c["metadata"] for c in chunks],
        )
        return len(chunks)

    def search(self, question: str, top_k: int | None = None) -> list[dict]:
        """相似度检索：把问题也向量化，找最近的 K 个 chunk。

        返回 [{"content": ..., "source": ..., "page": ..., "score": ...}]，
        score 是相似度（余弦），越大越相关。
        """
        top_k = top_k or config.TOP_K
        query_vec = get_embedder().embed([question])[0]
        result = self._collection.query(
            query_embeddings=[query_vec],
            n_results=top_k,
            # 空切片在 split_document 阶段已被过滤，这里无需再加 where_document 过滤
        )
        hits: list[dict] = []
        # chroma 返回的是按列组织的二维数组，第一维对应我们传入的 1 个查询
        for content, meta, distance in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            hits.append({
                "content": content,
                "source": meta.get("source", "未知"),
                "page": meta.get("page", 0),
                "score": round(1 - distance, 4),  # 余弦距离转相似度
            })
        return hits

    def list_documents(self) -> list[dict]:
        """列出已入库的文档（按 source 去重统计 chunk 数），相当于 GROUP BY 查询。"""
        data = self._collection.get(include=["metadatas"])
        counter: dict[str, int] = {}
        for meta in data["metadatas"]:
            source = meta.get("source", "未知")
            counter[source] = counter.get(source, 0) + 1
        return [{"source": s, "chunks": n} for s, n in sorted(counter.items())]

    def delete_by_source(self, source: str) -> int:
        """删除某个文件的全部 chunk（重新上传前先删旧数据，避免重复）。"""
        data = self._collection.get(where={"source": source})
        ids = data["ids"]
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)


# 模块级单例
_store: ChromaStore | None = None


def get_store() -> ChromaStore:
    global _store
    if _store is None:
        _store = ChromaStore()
    return _store
