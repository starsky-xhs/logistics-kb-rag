PY ?= python

.PHONY: help demo test eval ingest serve install clean

help:
	@echo "demo    离线端到端演示（零依赖）"
	@echo "test    unittest 全量测试"
	@echo "eval    运行 gold set 并写入 docs/EVAL_REPORT.md"
	@echo "ingest  入库 data/sample_docs"
	@echo "serve   启动 FastAPI（需 requirements.txt）"
	@echo "install 安装完整可选依赖"

install:
	$(PY) -m pip install -r requirements.txt

demo:
	$(PY) -m app.demo

test:
	$(PY) -m unittest discover -s tests -v

eval:
	$(PY) -m app.cli eval

ingest:
	$(PY) -m app.cli ingest data/sample_docs

serve:
	$(PY) -m uvicorn app.api.app:create_app --factory --host 127.0.0.1 --port 8000

clean:
	$(PY) -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
