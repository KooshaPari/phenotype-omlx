FROM ubuntu:24.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
# Build context is environment/ only. Oracle script lands via Harbor solution upload
# (solve.sh → niah_openai_smoke.py next to solve.sh).
