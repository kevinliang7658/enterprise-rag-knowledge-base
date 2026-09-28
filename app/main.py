"""FastAPI 应用入口。启动命令（项目根目录下）：

    uvicorn app.main:app --host 0.0.0.0 --port 8000

然后浏览器访问 http://127.0.0.1:8000 即可看到测试页面。
"""
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import config
from app.api.routes import router

# 启动前先把 data/uploads、data/chroma 等目录建好
config.ensure_dirs()

app = FastAPI(
    title="企业知识库RAG问答系统",
    description="文档上传 → 解析切分 → 向量入库 → 带引用来源的流式问答",
    version="1.0.0",
)

# 挂载 API 路由
app.include_router(router, prefix="/api")

# 模拟查询订单

# 挂载静态页面目录，访问 / 直接返回测试页面
app.mount("/static", StaticFiles(directory=config.BASE_DIR / "static"), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """首页：返回最简 HTML 测试页面。"""
    return FileResponse(config.BASE_DIR / "static" / "index.html")


@app.get("/api/health")
def health() -> dict:
    """健康检查。"""
    return {"status": "ok", "llm_model": config.LLM_MODEL, "embedding_provider": config.EMBEDDING_PROVIDER}
