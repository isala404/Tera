# Testing Guide

## pgvector Requirement

This project uses **pgvector** for 768-dimensional embeddings. pgvector is a **hard requirement** and is not optional.

## Running Tests with pgvector

### Option 1: Docker Compose (Recommended for Development)

The project includes a `docker-compose.yml` configured with `pgvector/pgvector:pg18`:

```bash
docker-compose up -d
cargo test
```

This sets up PostgreSQL with pgvector pre-installed and all tests will pass.

### Option 2: Forge Dev Mode

```bash
forge dev
```

This automatically spins up the docker-compose environment and runs the backend with full pgvector support.

### Option 3: Manual PostgreSQL with pgvector

If you have PostgreSQL installed locally with pgvector extension:

```bash
# Ensure pgvector extension is installed
psql -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Run tests
cargo test
```

## Test Expectations

All tests require pgvector to be available:

- ✅ **With pgvector**: All tests pass
- ❌ **Without pgvector**: Tests fail with clear error message: "pgvector extension is REQUIRED"

## Production Deployment

Docker Compose configuration automatically uses `pgvector/pgvector:pg18` which includes:
- PostgreSQL 18
- pgvector extension pre-installed
- All required indexing support (ivfflat)

No additional setup needed for production.
