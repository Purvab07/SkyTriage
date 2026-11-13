# SkyTriage

Safety and reliability assistant for small aviation operators.
Advisory only: it never replaces the official AD, approved maintenance data,
or a certified mechanic's judgement.

## Quick start

```bash
cp .env.example .env
docker compose up -d db redis
pip install -e ".[dev]"
uvicorn app.main:app --reload
```

Then open http://localhost:8000/docs
