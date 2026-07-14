"""文档加载器。

Markdown/TXT 使用标准库读取；DOCX/XLSX/PDF 为可选能力，缺少依赖时抛出
``OptionalDependencyError``，由入库流程记录为跳过原因，而不是让整批任务失败。
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Final

from .errors import LoaderError, OptionalDependencyError

SUPPORTED_SUFFIXES: Final[frozenset[str]] = frozenset(
    {".md", ".markdown", ".txt", ".docx", ".xlsx", ".pdf"}
)
_TEXT_SUFFIXES: Final = frozenset({".md", ".markdown", ".txt"})
_OPTIONAL_PACKAGES: Final = {".docx": "python-docx", ".xlsx": "openpyxl", ".pdf": "pypdf"}


def file_sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    """计算文件内容 SHA-256，用于增量入库的幂等判断。"""

    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while block := handle.read(block_size):
                digest.update(block)
    except OSError as exc:
        raise LoaderError(f"无法读取文件 {path.name}：{exc}") from exc
    return digest.hexdigest()


def iter_source_files(root: Path) -> list[Path]:
    """递归列出受支持的源文件，忽略隐藏目录和临时文件。"""

    if not root.exists() or not root.is_dir():
        raise LoaderError(f"源目录不存在或不是目录：{root}")
    files: list[Path] = []
    root = root.resolve()
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        # 只忽略源目录内部的隐藏目录；源目录本身位于 .work 之类的路径时仍应入库。
        relative_parts = path.relative_to(root).parts
        if any(part.startswith(".") or part == "__pycache__" for part in relative_parts):
            continue
        files.append(path)
    return sorted(files, key=lambda item: item.as_posix().lower())


def relative_source(path: Path, root: Path) -> str:
    """转为统一的正斜杠相对路径，作为跨平台稳定的 source。"""

    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise LoaderError(f"文件 {path} 不在源目录 {root} 下") from exc


def load_file(path: Path) -> tuple[str, dict[str, str]]:
    """按扩展名读取文档。

    返回值是 ``(纯文本, 元信息)``。元信息只放字符串，便于直接写入 SQLite。
    """

    suffix = path.suffix.lower()
    if suffix in _TEXT_SUFFIXES:
        return _load_text(path)
    if suffix == ".docx":
        return _load_docx(path)
    if suffix == ".xlsx":
        return _load_xlsx(path)
    if suffix == ".pdf":
        return _load_pdf(path)
    raise LoaderError(f"不支持的文件类型：{suffix or '无扩展名'}")


def _decode_bytes(raw: bytes) -> tuple[str, str]:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise LoaderError("无法识别文件编码，请转换为 UTF-8 后重试")


def _load_text(path: Path) -> tuple[str, dict[str, str]]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise LoaderError(f"读取 {path.name} 失败：{exc}") from exc
    text, encoding = _decode_bytes(raw)
    if not text.strip():
        raise LoaderError(f"{path.name} 内容为空")
    return text, {"encoding": encoding, "format": path.suffix.lower().lstrip(".")}


def _load_docx(path: Path) -> tuple[str, dict[str, str]]:
    try:
        import docx  # type: ignore[import-not-found]
    except ImportError as exc:
        raise OptionalDependencyError(path.suffix.lower(), "python-docx") from exc

    try:
        document = docx.Document(str(path))
    except Exception as exc:  # python-docx 的异常类型随版本变化，这里统一归类
        raise LoaderError(f"解析 DOCX 失败（{path.name}）：{exc}") from exc

    lines: list[str] = []
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        style_name = (paragraph.style.name if paragraph.style else "").lower()
        level_match = re.search(r"(\d+)\s*$", style_name)
        if ("heading" in style_name or "标题" in style_name) and level_match:
            level = min(6, max(1, int(level_match.group(1))))
            lines.append(f"{'#' * level} {text}")
        else:
            lines.append(text)

    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
            if any(cells):
                lines.append(" | ".join(cells))

    text = "\n\n".join(lines).strip()
    if not text:
        raise LoaderError(f"DOCX 中没有可提取的文本：{path.name}")
    return text, {"format": "docx", "paragraphs": str(len(document.paragraphs))}


def _load_xlsx(path: Path) -> tuple[str, dict[str, str]]:
    try:
        import openpyxl  # type: ignore[import-not-found]
    except ImportError as exc:
        raise OptionalDependencyError(path.suffix.lower(), "openpyxl") from exc

    try:
        workbook = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    except Exception as exc:
        raise LoaderError(f"解析 XLSX 失败（{path.name}）：{exc}") from exc

    sections: list[str] = []
    sheet_count = len(workbook.sheetnames)
    try:
        for sheet in workbook.worksheets:
            rows: list[str] = []
            for row in sheet.iter_rows(values_only=True):
                cells = ["" if value is None else str(value).strip() for value in row]
                if any(cells):
                    rows.append(" | ".join(cells))
            if rows:
                sections.append(f"## {sheet.title}\n\n" + "\n".join(rows))
    finally:
        workbook.close()

    text = "\n\n".join(sections).strip()
    if not text:
        raise LoaderError(f"XLSX 中没有可提取的内容：{path.name}")
    return text, {"format": "xlsx", "sheets": str(sheet_count)}


def _load_pdf(path: Path) -> tuple[str, dict[str, str]]:
    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]
    except ImportError as exc:
        raise OptionalDependencyError(path.suffix.lower(), "pypdf") from exc

    try:
        reader = PdfReader(str(path))
        pages: list[str] = []
        for number, page in enumerate(reader.pages, start=1):
            page_text = (page.extract_text() or "").strip()
            if page_text:
                pages.append(f"## 第 {number} 页\n\n{page_text}")
    except Exception as exc:
        raise LoaderError(f"解析 PDF 失败（{path.name}）：{exc}") from exc

    text = "\n\n".join(pages).strip()
    if not text:
        raise LoaderError(f"PDF 中没有可提取的文本，可能是扫描件：{path.name}")
    metadata: dict[str, str] = {"format": "pdf", "pages": str(len(reader.pages))}
    if reader.metadata and reader.metadata.title:
        metadata["pdf_title"] = str(reader.metadata.title)
    return text, metadata


def infer_title(text: str, fallback: str) -> str:
    """优先使用第一个 Markdown 一级标题。"""

    for line in text.splitlines():
        match = re.match(r"^#\s+(.+?)\s*$", line.strip())
        if match:
            return match.group(1).strip()
    return fallback
