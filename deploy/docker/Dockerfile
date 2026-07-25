# Multi-stage build of the serving replica.
#
# Models are not baked into the image: mount a GGUF at runtime and point
# CAMELID_ENTERPRISE_MODEL at it. Inside a container the bind address must be
# 0.0.0.0 (the loopback default is for bare-metal use).

FROM rust:1-bookworm AS builder
WORKDIR /build
# git is needed to fetch the pinned engine dependency.
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
RUN cargo build --release --bin camelid-enterprise

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /build/target/release/camelid-enterprise /usr/local/bin/camelid-enterprise

ENV CAMELID_ENTERPRISE_ADDR=0.0.0.0:8181
EXPOSE 8181

# Listening is not readiness — the model loads after the port is bound. Probe
# /v1/health rather than /v1/models: a model whose runtime could not be built is
# still listed by /v1/models, so grepping for an id reports healthy on a replica
# that fails every generation request.
HEALTHCHECK --interval=10s --timeout=3s --start-period=120s \
    CMD curl -sf http://127.0.0.1:8181/v1/health | grep -q '"generation_ready":true' || exit 1

ENTRYPOINT ["camelid-enterprise", "serve"]
