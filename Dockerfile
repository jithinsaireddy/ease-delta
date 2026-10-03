# EASE-Delta service, CPU only. The model directory is mounted or copied in at /models.
#
#   docker build -t ease-delta .
#   docker run --rm -p 8791:8791 -v $PWD/release:/models:ro -v ease-data:/data ease-delta
#
# Multi-workspace mode with an admin key:
#   docker run --rm -p 8791:8791 -e EASE_MULTI=1 -e EASE_ADMIN_KEY=... -v $PWD/release:/models:ro -v ease-data:/data ease-delta
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
    EASE_DEVICE=cpu EASE_DATA=/data EASE_MODEL=/models/edge EASE_AGGREGATOR=/models/aggregator EASE_PORT=8791

RUN useradd --create-home --uid 1000 ease
WORKDIR /app

# torch from the CPU index keeps the image a few hundred MB instead of several GB
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.6"
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

USER ease
VOLUME ["/data"]
EXPOSE 8791
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8791/health', timeout=4).status == 200 else 1)"
CMD ["sh", "-c", "exec ease serve --model \"$EASE_MODEL\" --aggregator \"$EASE_AGGREGATOR\" --data \"$EASE_DATA\" --host 0.0.0.0 --port \"$EASE_PORT\" ${EASE_MULTI:+--multi}"]
