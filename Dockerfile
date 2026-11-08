FROM python:3.12-slim@sha256:b18b261e1b4974b4f321451ad5251ded85abb22c2b8b262da527dda1d031ab30
WORKDIR /app
COPY requirements-demo.txt requirements-demo.lock.txt ./
RUN pip install --no-cache-dir -r requirements-demo.txt -c requirements-demo.lock.txt \
    && useradd --create-home --uid 10001 demo
COPY --chown=demo:demo streamlit_app.py ./
COPY --chown=demo:demo .streamlit ./.streamlit
COPY --chown=demo:demo src/visionbench/orchestration ./src/visionbench/orchestration
COPY --chown=demo:demo src/visionbench/retrieval ./src/visionbench/retrieval
COPY --chown=demo:demo src/visionbench/storage ./src/visionbench/storage
COPY --chown=demo:demo src/visionbench/ui ./src/visionbench/ui
COPY --chown=demo:demo data/corpus ./data/corpus
COPY --chown=demo:demo data/demo ./data/demo
COPY --chown=demo:demo reports/visionbench.sqlite ./reports/visionbench.sqlite
COPY --chown=demo:demo reports/resources_100.sqlite reports/resources_100.manifest.json ./reports/
COPY --chown=demo:demo reports/resources_100/raw.json ./reports/resources_100/raw.json
COPY --chown=demo:demo reports/j2_latency_pilot.json ./reports/j2_latency_pilot.json
COPY --chown=demo:demo reports/j2_coco_eval/metrics.json ./reports/j2_coco_eval/metrics.json
COPY --chown=demo:demo reports/model_comparison/summary.json ./reports/model_comparison/summary.json
COPY --chown=demo:demo reports/model_comparison/protocol.json ./reports/model_comparison/protocol.json
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
USER demo
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health',timeout=3)"
CMD ["python", "-B", "-m", "streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
