"""API 路由：文档上传/列表/删除 + 流式问答 + 会话历史。

流式响应用 SSE（Server-Sent Events）：HTTP 长连接，
服务端不断往响应里写 "data: {...}\\n\\n"，前端逐条解析。
"""
import json
import shutil
from collections.abc import Generator

from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app import config
from app.api.schemas import ChatRequest, UploadResponse
from app.rag import history
from app.rag.chain import get_chain
from app.rag.loader import SUPPORTED_EXTENSIONS, load_document
from app.rag.splitter import split_document
from app.rag.store import get_store

router = APIRouter()


@router.post("/documents/upload", response_model=UploadResponse)
def upload_document(file: UploadFile) -> UploadResponse:
    """上传文档：保存 → 解析 → 切分 → Embedding → 入向量库。

    完整流水线走一遍，这是本系统的核心数据流。
    """
    filename = file.filename or "unnamed"
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型 {ext}，仅支持 pdf/docx/md/txt")

    # 1. 落盘保存原始文件
    save_path = config.UPLOAD_DIR / filename
    with save_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        # 2. 解析成纯文本（按页）
        doc = load_document(save_path)
        # 3. 切成 chunk 并携带来源元数据
        chunks = split_document(doc)
        if not chunks:
            raise HTTPException(status_code=400, detail="文档解析后没有有效文本内容")
        store = get_store()
        # 4. 同名文档先删旧数据再写入，保证"重新上传=覆盖更新"
        store.delete_by_source(filename)
        # 5. 批量 Embedding 并入库
        count = store.add_chunks(chunks)
    except HTTPException:
        raise
    except Exception as e:  # 解析/Embedding 失败要清理半成品，避免脏数据
        save_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"文档处理失败：{e}") from e

    return UploadResponse(filename=filename, chunks=count, message="上传并入库成功")


@router.get("/documents")
def list_documents() -> list[dict]:
    """查看知识库里有哪些文档、各切了多少块。"""
    return get_store().list_documents()


@router.delete("/documents/{source}")
def delete_document(source: str) -> dict:
    """按文件名删除文档的全部切片。"""
    deleted = get_store().delete_by_source(source)
    return {"source": source, "deleted_chunks": deleted}


def _sse_stream(question: str, session_id: str, top_k: int) -> Generator[str, None, None]:
    """把 RagChain 的事件流包装成 SSE 格式文本流。

    SSE 协议很简单：每条消息是 "data: <json>\\n\\n"，前端用 EventSource 或 fetch 流式读取。
    """
    # 先把 session_id 发给前端（新会话时前端需要保存它供后续请求使用）
    yield f"data: {json.dumps({'type': 'session', 'data': session_id}, ensure_ascii=False)}\n\n"
    try:
        for event in get_chain().stream_answer(question, session_id, top_k=top_k):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'data': str(e)}, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"  # 结束标记，前端据此关闭流


@router.post("/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    """流式问答接口。curl 测试：

    curl -N -X POST http://127.0.0.1:8000/api/chat \\
         -H "Content-Type: application/json" \\
         -d '{"question": "年假有几天？"}'
    """
    session_id = req.session_id or history.create_session(title=req.question[:20])
    return StreamingResponse(
        _sse_stream(req.question, session_id, req.top_k),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},  # 禁缓冲，保证实时
    )


@router.get("/sessions")
def list_sessions() -> list[dict]:
    """会话列表（前端侧边栏）。"""
    return history.list_sessions()


@router.get("/sessions/{session_id}/history")
def session_history(session_id: str) -> list[dict]:
    """查看某会话的完整历史消息。"""
    return history.get_history(session_id, limit=100)
