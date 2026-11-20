# The client side of the bench: AIPerf and the Python dependencies. The repository itself is not copied: Compose
# mounts it at /app, so results, traces and the tokenizer stay in your folder (README, section 4.4).

# Stage 1: some dependencies of AIPerf have no ready-made package on every processor (arm64) and are compiled here;
# the compiler stays in this stage.
FROM python:3.12-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
# AIPerf 0.13 is required (README, section 4.3).
# The other libraries are pinned to the versions this image was tested with (they satisfy the README's ranges).
RUN python -m venv /opt/bench \
    && /opt/bench/bin/pip install --no-cache-dir aiperf==0.13.0 openai==3.55.1 httpx==0.28.1 aiohttp==3.14.5 tokenizers==0.23.3

# Stage 2: what runs.
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 DO_NOT_TRACK=1 HOME=/home/bench VIRTUAL_ENV=/opt/bench PATH=/opt/bench/bin:$PATH
RUN apt-get update && apt-get install -y --no-install-recommends curl procps \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /opt/bench /opt/bench
# A home folder that any user can write to: Compose can run the container as your own user (BENCH_UID, BENCH_GID).
RUN mkdir -p /home/bench && chmod 1777 /home/bench
WORKDIR /app
