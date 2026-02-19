FROM rust:1-slim-bookworm AS dev

WORKDIR /app

RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install cargo-watch for hot reload
RUN cargo install cargo-watch --locked

# Development command - no frontend embedding, watch for changes
CMD ["cargo", "watch", "-x", "run --no-default-features"]

FROM oven/bun:1-alpine AS frontend-builder

WORKDIR /app/frontend

COPY frontend/package.json frontend/bun.lock* ./
COPY frontend/.forge ./.forge

RUN bun install --frozen-lockfile || bun install

COPY frontend ./

RUN bun run build

FROM rust:1-alpine AS builder

WORKDIR /app

RUN apk add --no-cache musl-dev openssl-dev openssl-libs-static pkgconf

COPY Cargo.toml Cargo.lock* ./

# Cache dependencies (without frontend embedding)
RUN mkdir -p src && \
    echo "fn main() {}" > src/main.rs && \
    cargo build --release --no-default-features && \
    rm -rf src

COPY src ./src
COPY migrations ./migrations

# Copy frontend build for embedding
COPY --from=frontend-builder /app/frontend/build ./frontend/build

# Build release with embedded frontend
RUN touch src/main.rs && cargo build --release

FROM alpine:3.21 AS production

WORKDIR /app

RUN apk add --no-cache ca-certificates libgcc

COPY --from=builder /app/target/release/tera /app/tera
COPY --from=builder /app/migrations /app/migrations

EXPOSE 8080

ENV RUST_LOG=info
ENV HOST=0.0.0.0
ENV PORT=8080

CMD ["/app/tera"]
