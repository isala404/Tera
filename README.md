# tera

A full-stack application built with [FORGE](https://tryforge.dev) - PostgreSQL is your only infrastructure.

## What is FORGE?

FORGE handles the hard parts of full-stack engineering so you can focus on business logic:

- **Auth & Sessions** - JWT validation, role-based access, multi-tenancy
- **Smart Caching** - Query-level caching with rate limiting out of the box
- **Transactional Safety** - Atomic writes with automatic rollback on failure
- **End-to-End Type Safety** - Backend types flow directly to your frontend
- **Background Jobs** - Retries, progress tracking, worker capabilities
- **Cron Scheduling** - Timezone-aware, catch-up runs, leader-only execution
- **Durable Workflows** - Multi-step processes that survive restarts
- **Real-time Updates** - SSE subscriptions with automatic invalidation
- **Observability** - Built-in metrics, logs, and traces

No Redis. No Kafka. No message queues. Just PostgreSQL.

## Quick Start

```bash
forge dev
```

This single command:
- Starts embedded PostgreSQL (data in `./pg_data/`)
- Compiles and runs the backend with hot reload
- Starts the frontend dev server
- Opens your browser

Backend: `http://localhost:8080` | Frontend: `http://localhost:5173`

Useful options:

```bash
# Use external PostgreSQL from DATABASE_URL
forge dev --no-pg

# Kill process(es) occupying 8080/5173/5432 and take over
forge dev --takeover-ports

# Customize ports
forge dev --backend-port 8081 --frontend-port 4173 --db-port 5433
```

Backend hot reload watches only backend-relevant files (`src/`, `migrations/`, `build.rs`, `Cargo.toml`, `Cargo.lock`, `.env`, `forge.toml`), so unrelated root files do not trigger restarts. By default, if a requested port is busy, Forge shows the owning process and exits.

## Using Docker Compose

For containerized development:

```bash
docker compose up --build
```

## Using External Database

If you prefer your own PostgreSQL:

```bash
# Set your database URL
echo "DATABASE_URL=postgres://user:pass@localhost/tera" >> .env

# Start without embedded postgres
forge dev --no-pg
```

## Build

Single binary (backend + embedded frontend):
```bash
cd frontend && bun install && bun run build && cd ..
cargo build --release
```

## Test

```bash
# Run tests with embedded PostgreSQL (recommended)
cargo test --features embedded-db

# Or with external database
TEST_DATABASE_URL=postgres://localhost/test cargo test
```

See `src/functions/` for test examples.

## Deployment

For deployment options (Docker, VM, embedded PostgreSQL, etc.), see the [Deployment Guide](https://tryforge.dev/docs/concepts/deployment).

## Project Structure

```
tera/
├── src/
│   ├── main.rs              # Entry point
│   ├── schema/              # Data models
│   └── functions/           # Queries, mutations, jobs, crons, workflows
├── migrations/              # SQL migrations
├── frontend/                # SvelteKit frontend
├── forge.toml               # FORGE configuration
├── docker-compose.yml       # Development containers
└── Dockerfile               # Production image
```
