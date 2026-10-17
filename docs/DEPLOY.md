# Deploying the demo for free

The public demo serves **recorded results**. It needs no database and no LLM key, so it stays free and
cannot be abused to spend anything.

## What gets deployed
The dashboard serves **recorded results**, so the public demo needs no database and no LLM key. It stays free and
cannot be abused to spend anything. There are two ways to host it:

1. **Static site (free everywhere, recommended).** `python -m experiments.export_static --out <dir> --space-readme`
   writes `index.html` plus `data.json` (about 200 KB). The same page falls back to `data.json` when there is no
   `/api`. Works on a **static Hugging Face Space**, GitHub Pages, Netlify, etc.
2. **Container with the FastAPI backend** (`Dockerfile`). Needs a host that runs containers; on Hugging Face the
   Docker SDK may not be free on every account, so check before relying on it. Live runs are off unless
   `ENABLE_LIVE_RUN=1`.

## Static Hugging Face Space
1. Create a Space at huggingface.co/new-space: SDK **Static**, template **Blank**, Public.
2. Build the site: `python -m experiments.export_static --out ../agent-reliability-lab-static --space-readme`
3. Clone the Space, copy the three files in (`index.html`, `data.json`, `README.md`), commit and push.
   Authenticate with your HF username and a **Write** access token.
4. The site is live at `https://<hf-username>-agent-reliability-lab.static.hf.space` within a minute or two.
   Note the `.static.` in the hostname: static Spaces are not served from plain `.hf.space`.

Re-run the export whenever `results/` changes, then copy and push again.

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
