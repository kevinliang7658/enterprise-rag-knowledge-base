"""文档解析：把 pdf / word / md / txt 统一解析成纯文本。

设计思路：一个 DocumentLoader 接口 + 三个实现类，
再用一个工厂方法按扩展名分发。Python 里用字典当工厂就够了。

返回的统一结构：
    {"text": "全文纯文本", "source": "文件名", "pages": [第1页文本, 第2页文本, ...]}
为什么保留 pages？—— 引用来源要能告诉用户"答案出自第几页"，这是企业知识库的硬需求。
"""
from pathlib import Path

from pypdf import PdfReader
from docx import Document  # python-docx 包，import 名叫 docx，注意别写错

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".md", ".txt"}


def load_pdf(path: Path) -> dict:
    """解析 PDF：逐页提取文本，保留页码供引用。"""
    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    pages = [p for p in pages if p]  # 去掉扫描件等提取不出文字的空页
    return {"text": "\n\n".join(pages), "source": path.name, "pages": pages}


def load_docx(path: Path) -> dict:
    """解析 Word：按段落提取。Word 没有固定页码概念，按段落近似分页（每 ~15 段算一'页'）。"""
    doc = Document(str(path))
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    # 把段落每 15 个合成一个逻辑页，让引用粒度不至于太碎
    pages = ["\n".join(paragraphs[i:i + 15]) for i in range(0, len(paragraphs), 15)]
    return {"text": "\n\n".join(paragraphs), "source": path.name, "pages": pages}


def load_text(path: Path) -> dict:
    """解析 md / txt：按空行分成逻辑页。"""
    text = path.read_text(encoding="utf-8")
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    pages = ["\n".join(blocks[i:i + 8]) for i in range(0, len(blocks), 8)]
    return {"text": text, "source": path.name, "pages": pages}


# 工厂：扩展名 -> 解析函数。Java 里的 Map<String, DocumentLoader> 注册表
_LOADERS = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".md": load_text,
    ".txt": load_text,
}


def load_document(path: Path) -> dict:
    """统一入口：按扩展名选择解析器。"""
    ext = path.suffix.lower()
    if ext not in _LOADERS:
        raise ValueError(f"不支持的文件类型：{ext}，仅支持 {sorted(SUPPORTED_EXTENSIONS)}")
    return _LOADERS[ext](path)
