.PHONY: install ingest train serve test lint

install:
	pip install -r requirements.txt

ingest:
	python -m data.ingest

train:
	python -m training.train

serve:
	uvicorn api.main:app --reload --port 8000

test:
	pytest tests/ -v

lint:
	ruff check . && black --check .

format:
	black . && ruff check --fix .
