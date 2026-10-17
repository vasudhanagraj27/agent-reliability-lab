# Serves the dashboard + API from recorded results (no database, no LLM key needed).
# Works on Hugging Face Spaces (Docker SDK, port 7860) and any container host.
FROM python:3.13-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY core ./core
COPY domains ./domains
COPY experiments ./experiments
COPY api ./api
COPY static ./static
COPY results ./results

ENV PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
