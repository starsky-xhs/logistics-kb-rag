"""测试共享工具。

不 mock 检索、分块、存储或生成逻辑；只负责创建临时目录、配置与样例文本。
"""

from __future__ import annotations

import shutil
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from app.config import Settings

WORKSPACE_ROOT = Path(__file__).resolve().parent / ".work"


@contextmanager
def temporary_workspace(label: str = "case") -> Iterator[Path]:
    """在仓库内创建隔离工作区。

    不使用 ``tempfile.TemporaryDirectory``：部分 Windows 沙箱会拒绝在被
    ``mkdtemp`` 创建的目录里继续建子目录。``tests/.work`` 已加入 .gitignore，
    ``ignore_errors=True`` 保证清理失败不会掩盖真正的测试断言。
    """

    root = WORKSPACE_ROOT / f"{label}-{uuid.uuid4().hex}"
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "db_path": tmp_path / "kb.sqlite3",
        "top_k": 3,
        "candidate_k": 10,
        "chunk_size": 420,
        "chunk_overlap": 80,
        "llm_provider": "mock",
        "embed_provider": "hash",
        "rerank_provider": "heuristic",
        "log_level": "WARNING",
    }
    values.update(overrides)
    return Settings(**values)


def write_doc(root: Path, name: str, content: str) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


REFUND_DOC = """# 售后与退货政策

## 退款到账时效

仓库验收通过后，退款在 1 个工作日内提交支付渠道。原路退回一般 1 至 3 个工作日到账；
对公转账为 3 至 5 个工作日。银行清算延迟不计入处理时效。

## 退货运费

七天无理由退货由客户承担退回运费，原订单已产生的发货运费不退。
质量问题、错发和漏发由迅达物流承担来回运费。
"""

FREIGHT_DOC = """# 运费与时效说明

## 体积重

标准快递体积重按长 × 宽 × 高（厘米）除以 6000 计算；
大件专线按除以 3000 计算。实重与体积重取较大值计费。

## 跨省时效

跨省主要城市一般为 3 至 5 个工作日，偏远地区为 5 至 7 个工作日。
"""

DAMAGE_DOC = """# 货物破损理赔流程

## 所需材料

1. 运单号、订单号及发货清单；
2. 破损货物全景照片、包装六面照片、破损部位特写，合计不少于 6 张；
3. 货物价值证明，包括采购发票或销售合同；
4. 签收单或签收记录；
5. 加盖公章的索赔函。

## 未保价赔付

未保价货物按实际损失赔付，单票最高不超过运费的 3 倍，且不超过人民币 1000 元。
"""
