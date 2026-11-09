FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ISSUE_AGENT_DATA_DIR=/app/data

WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY issue_agent ./issue_agent
COPY README.md ./

RUN mkdir -p /app/data
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/healthz')"

# 生产请设置 ISSUE_AGENT_ENV=prod + API_KEYS + CORS_ORIGINS，并在前面挂 TLS/反向代理
CMD ["python", "-m", "uvicorn", "issue_agent.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
