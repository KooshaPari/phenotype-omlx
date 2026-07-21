FROM ubuntu:24.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
# Baked for environments that COPY task context at build time.
COPY solution/niah_openai_smoke.py /app/niah_openai_smoke.py
