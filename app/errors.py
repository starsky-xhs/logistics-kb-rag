"""共享异常类型。

异常按处理阶段分类，便于 CLI 输出友好提示，也便于 API 层统一映射 HTTP 状态码。
所有异常都带一个可读的中文提示，不把第三方库的原始错误直接暴露给最终用户。
"""

from __future__ import annotations


class KbError(Exception):
    """项目内所有可预期异常的基类。"""


class ConfigError(KbError):
    """配置缺失、取值越界或环境变量格式错误。"""


class LoaderError(KbError):
    """文档读取失败，例如编码错误、文件损坏。"""


class OptionalDependencyError(LoaderError):
    """读取某类文档需要可选依赖，但当前环境没有安装。"""

    def __init__(self, suffix: str, package: str) -> None:
        self.suffix = suffix
        self.package = package
        super().__init__(
            f"读取 {suffix} 文件需要可选依赖 {package}，"
            f"请执行 `pip install {package}` 后重试；核心检索功能不依赖它。"
        )


class ValidationError(KbError):
    """调用方输入不合法，例如空问题、非法过滤条件。"""


class IngestError(KbError):
    """入库流程中的非预期错误。"""


class RetrievalError(KbError):
    """向量检索、融合或重排失败。"""


class GenerationError(KbError):
    """生成模型调用失败、响应格式异常或超出预算。"""
