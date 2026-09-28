"""请求/响应模型：相当于 Java 里的 DTO。

用 pydantic 定义后，FastAPI 自动做参数校验，
类型不对直接返回 422。
"""
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """问答接口入参。session_id 可选：不传则自动开新会话。"""
    question: str = Field(..., min_length=1, description="用户问题，不能为空")
    session_id: str = Field(default="", description="会话ID，空则新建会话")
    top_k: int = Field(default=4, ge=1, le=20, description="指定检索切片数，1~20")

class UploadResponse(BaseModel):
    """上传接口出参。"""
    filename: str
    chunks: int          # 切成了多少块
    message: str
