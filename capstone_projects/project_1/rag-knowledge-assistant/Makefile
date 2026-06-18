# Thin wrappers so the common workflow is discoverable and consistent.
.PHONY: install data ingest serve eval test docker

export PYTHONPATH := src

install:
	pip install -r requirements.txt

data:               ## generate the mock knowledge base + eval set
	python scripts/generate_mock_data.py

ingest:             ## chunk, embed, and index data/raw into the vector store
	python scripts/ingest.py --data-dir data/raw

serve:              ## run the API locally with reload
	uvicorn rag.api.main:app --reload --port 8000

eval:               ## run the evaluation harness (add --no-judge to skip LLM judge)
	python eval/run_eval.py

eval-retrieval:     ## retrieval + latency metrics only, no API key needed
	python eval/run_eval.py --no-judge

test:
	pytest -q

docker:
	docker compose up --build
