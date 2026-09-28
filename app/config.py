"""全局配置：所有可变项都从环境变量读取，绝不把密钥写进代码。

"""
import os
from pathlib import Path

# ---------- 路径 ----------
BASE_DIR = Path(__file__).resolve().parent.parent   # 项目根目录（app/ 的上一级）
DATA_DIR = Path(os.getenv("RAG_DATA_DIR", BASE_DIR / "data"))          # 运行数据目录
UPLOAD_DIR = DATA_DIR / "uploads"                   # 上传的原始文档
CHROMA_DIR = DATA_DIR / "chroma"                    # Chroma 向量库持久化目录
SQLITE_PATH = DATA_DIR / "chat_history.db"          # 会话历史 sqlite 文件

# ---------- 大模型（对话）：默认 DeepSeek，OpenAI 兼容协议 ----------
# 想换通义千问/智谱/Kimi，只需改环境变量 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL，不用改代码
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

# ---------- Embedding：DashScope text-embedding-v4（DeepSeek 没有 embedding 接口） ----------
# key 从环境变量 DASHSCOPE_API_KEY 读取，绝不写进代码
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "siliconflow")  # siliconflow / local
SILICONFLOW_BASE_URL = os.getenv("SILICONFLOW_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY") or os.getenv("DASHSCOPE_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")

# ---------- 切分参数 ----------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "400"))      # 每个切片最大字符数
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "60"))  # 相邻切片重叠字符数（防切断语义）

# ---------- 检索参数 ----------
TOP_K = int(os.getenv("TOP_K", "4"))                   # 每次检索返回的切片数


def ensure_dirs() -> None:
    """启动时确保数据目录存在。"""
    for d in (DATA_DIR, UPLOAD_DIR, CHROMA_DIR):
        d.mkdir(parents=True, exist_ok=True)
