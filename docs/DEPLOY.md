# Deploying the demo for free

The public demo serves **recorded results**. It needs no database and no LLM key, so it stays free and
cannot be abused to spend anything.

## What gets deployed
`Dockerfile` builds a container that runs `uvicorn api.main:app` and serves the dashboard from
`results/*.jsonl` and `results/*.traces.json`. Live runs are off unless `ENABLE_LIVE_RUN=1`.

## Hugging Face Spaces (Docker SDK), free CPU tier
1. Create a Space at huggingface.co/new-space, choose the **Docker** SDK, blank template.
2. In the Space's `README.md`, use this front matter:
   ```
   ---
   title: Agent Reliability Lab
   sdk: docker
   app_port: 7860
   ---
   ```
3. Push this repo's files (`Dockerfile`, `requirements.txt`, `core/`, `domains/`, `experiments/`,
   `api/`, `static/`, `results/`) to the Space's git remote.
4. The Space builds and serves the dashboard at its URL.

Check Hugging Face's current free-tier terms before relying on this; they can change.

## Before you publish
- Run the analysis so the snapshot is complete:
  `python -m experiments.analyze --name main`
  (this also exports traces to `results/main.traces.json`).
- Confirm `.env` is not in the image (`.dockerignore` excludes it) and not committed.
- Rotate your LLM key if it was ever pasted anywhere public.

## Local run
```
docker compose up -d                 # Postgres on :5433
python -m uvicorn api.main:app       # dashboard at http://localhost:8000
```
