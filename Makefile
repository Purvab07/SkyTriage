.PHONY: up down api test lint

up:
	docker compose up -d db redis

down:
	docker compose down

api:
	uvicorn app.main:app --reload

test:
	pytest

lint:
	ruff check . && mypy app
