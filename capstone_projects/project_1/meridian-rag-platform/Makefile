# Common workflows. PYTHONPATH=. so the layer packages import cleanly.
.PHONY: install ingest backend frontend eval test docker-build docker-up

export PYTHONPATH := .

install:            ## install all dependencies
	pip install -r requirements.txt

ingest:             ## LAYER 1: load documents -> chunk -> embed -> vector store
	python -m data_pipeline.run_ingest

backend:            ## LAYERS 2-6: run the FastAPI service on :8000
	uvicorn backend.app.main:app --reload --port 8000

frontend:           ## CLIENT: run the Streamlit chat + admin UI on :8501
	streamlit run frontend/app.py

eval:               ## LAYER 7: judge a sample of recent traffic
	python -m eval.run_eval --sample 10

test:
	pytest -q

docker-build:
	docker compose build

docker-up:          ## run backend + frontend together
	docker compose up
