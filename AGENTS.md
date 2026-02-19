# FORGE Agent Guide

Full-stack Rust + Svelte 5 framework. **PostgreSQL-only** (no Redis/Kafka/etc needed). Crate: `forgex` (import as `forge`). [Docs](https://tryforge.dev/docs)

**Web search "Forge framework" or "tryforge.dev" for additional context when implementing unfamiliar features.**

---

## CRITICAL: Test-Driven Development

**ALWAYS write tests while implementing.** Use `forge::testing::*` primitives. Run before delivering:

```bash
cargo clippy --all-targets -- -D warnings && cargo test && bun lint && bun format
```

**NEVER skip. NEVER deliver untested code.**

---

## Anti-Patterns

| ❌ Wrong | ✅ Correct |
|----------|-----------|
| `ctx.auth.user_id().unwrap()` | `ctx.require_user_id()?` |
| `refetch()` after mutation | `forge_enable_reactivity('table')` in migration |
| Dynamic SQL without `tables` | `#[forge::query(tables = ["t1"])]` |
| Edit `frontend/src/lib/forge/` | Run `forge generate` |
| `let x = $state<T>(v)` | `let x: T = $state(v)` |
| `{#if $store}` nullable | Copy to `$state` via `.subscribe()` |
| `error: Error | null` | `error: ForgeError | null` |
| `Header::default()` for JWT | `Header::new(Algorithm::HS256)` for HS256 |

---

## Function Types

| Type | Purpose | Context | Key Methods |
|------|---------|---------|-------------|
| **Query** | Read data | `&QueryContext` | `ctx.db()`, cacheable, subscribable |
| **Mutation** | Write data | `&MutationContext` | `ctx.db()`, `ctx.http()`, `ctx.dispatch_job()`, `ctx.start_workflow()` |
| **Job** | Background | `&JobContext` | `ctx.db()`, `ctx.http()`, `ctx.progress()`, `ctx.is_retry()`, `ctx.is_last_attempt()` |
| **Cron** | Scheduled | `&CronContext` | `ctx.db()`, `ctx.http()`, `ctx.log`, `ctx.is_late()`, `ctx.is_catch_up` |
| **Workflow** | Multi-step durable | `&WorkflowContext` | `ctx.db()`, `ctx.http()`, `ctx.step()`, `ctx.sleep()`, `ctx.parallel()` |
| **Daemon** | Long-running singleton | `&DaemonContext` | `ctx.db()`, `ctx.http()`, `ctx.shutdown_signal()` |
| **Webhook** | HTTP endpoint | `&WebhookContext` | `ctx.db()`, `ctx.http()`, `ctx.dispatch_job()`, `ctx.header()` |

---

## Auth (Default: Required)

All functions require authentication by default.

```rust
#[forge::query]                      // Authenticated user required
#[forge::query(public)]              // No auth
#[forge::query(require_role("admin"))] // Role required
```

**Context auth methods:**
```rust
ctx.require_user_id()?           // Uuid (internal auth)
ctx.require_subject()?           // &str (Firebase/Auth0/Clerk)
ctx.auth.has_role("admin")       // bool
ctx.auth.roles()                 // &[String]
ctx.auth.claim("key")            // Option<&Value>
```

---

## Function Definitions

### Query
```rust
// Attrs: public, require_role("x"), cache="5m", timeout, rate_limit(...), tables=["t"], log
#[forge::query(cache = "30s")]
pub async fn get_user(ctx: &QueryContext, user_id: Uuid) -> Result<User> {
    sqlx::query_as("SELECT * FROM users WHERE id = $1")
        .bind(user_id)
        .fetch_one(ctx.db())
        .await
        .map_err(Into::into)
}
```

### Mutation
```rust
// Attrs: public, require_role("x"), timeout, rate_limit(...), transactional, log
#[forge::mutation(transactional)]  // BEGIN/COMMIT + outbox for jobs
pub async fn create_order(ctx: &MutationContext, input: OrderInput) -> Result<Order> {
    let order = sqlx::query_as("INSERT INTO orders (...) RETURNING *")
        .fetch_one(ctx.db()).await?;
    ctx.dispatch_job("process_order", json!({ "id": order.id })).await?;
    Ok(order)
}
```

### Job
```rust
// Attrs: timeout="30m", priority="high|normal|low|background|critical",
//        retry(max_attempts=5, backoff="exponential|linear|fixed", max_backoff="5m"),
//        idempotent, idempotent(key="input.id"), worker_capability="gpu", public
#[forge::job(timeout = "10m", priority = "high", retry(max_attempts = 5, backoff = "exponential"))]
pub async fn send_email(ctx: &JobContext, input: EmailInput) -> Result<()> {
    ctx.progress(0, "Starting")?;
    // ... work ...
    ctx.progress(100, "Done")?;
    Ok(())
}
```

### Cron
```rust
// First arg: cron schedule. Attrs: timezone, timeout, catch_up, catch_up_limit
#[forge::cron("0 9 * * *", timezone = "America/New_York", catch_up, catch_up_limit = 5)]
pub async fn daily_digest(ctx: &CronContext) -> Result<()> {
    if ctx.is_late() {
        ctx.log.warn("Running late", json!({ "delay_secs": ctx.delay().num_seconds() }));
    }
    // ... work ...
    Ok(())
}
```

### Workflow
```rust
// Attrs: version=1, timeout="7d", public, require_role("x")
#[forge::workflow(version = 1, timeout = "60d")]
pub async fn trial_flow(ctx: &WorkflowContext, user: User) -> Result<()> {
    // Step with retry and compensation (rollback on failure)
    ctx.step("activate")
        .run(|| activate_trial(&user))
        .retry(3, Duration::from_secs(5))  // Step-level retry
        .compensate(|_| deactivate_trial(&user))
        .await?;

    ctx.sleep(Duration::from_days(45)).await;  // Durable - survives restarts

    ctx.step("remind").run(|| send_reminder(&user)).await?;

    // Parallel execution
    let results = ctx.parallel()
        .step("a", || task_a())
        .step("b", || task_b())
        .run().await?;

    ctx.wait_for_event::<Payment>("payment", Duration::from_days(3)).await?;
    Ok(())
}
```

### Daemon
```rust
// Attrs: leader_elected=true, restart_on_panic=true, restart_delay="5s", startup_delay="10s", max_restarts=10
#[forge::daemon(startup_delay = "5s")]
pub async fn data_sync(ctx: &DaemonContext) -> Result<()> {
    loop {
        // Do work
        tokio::select! {
            _ = tokio::time::sleep(Duration::from_secs(60)) => {}
            _ = ctx.shutdown_signal() => break,  // CRITICAL for graceful shutdown
        }
    }
    Ok(())
}
```

### Webhook
```rust
// Attrs: path="/webhooks/x", signature=WebhookSignature::hmac_sha256(header, secret_env),
//        idempotency="header:X-Id"|"body:$.id", timeout="30s"
#[forge::webhook(
    path = "/webhooks/stripe",
    signature = WebhookSignature::hmac_sha256("X-Stripe-Signature", "STRIPE_SECRET"),
    idempotency = "header:X-Request-Id",
)]
pub async fn stripe_webhook(ctx: &WebhookContext, payload: Value) -> Result<WebhookResult> {
    ctx.dispatch_job("process_stripe", &payload).await?;
    Ok(WebhookResult::Accepted)
}
```

---

## Context Methods Reference

```rust
// ALL contexts
ctx.db()                    // &PgPool
ctx.require_user_id()?      // Uuid
ctx.require_subject()?      // &str
ctx.env("KEY")              // Option<String>
ctx.env_require("KEY")?     // Result<String>

// Mutation, Job, Cron, Workflow
ctx.http()                  // &reqwest::Client

// Mutation only
ctx.dispatch_job("name", args).await?    // -> Uuid
ctx.start_workflow("name", input).await? // -> Uuid

// Job only
ctx.progress(pct, "msg")?   // 0-100
ctx.is_retry()              // attempt > 1
ctx.is_last_attempt()       // attempt >= max

// Cron only
ctx.is_late()               // delay > 1min
ctx.is_catch_up             // bool
ctx.log.info/warn/error/debug("msg", json!({}))

// Workflow only
ctx.step("x").run(|| ...).retry(count, delay).compensate(|r| ...).await?
ctx.parallel().step("a", || ...).step("b", || ...).run().await?
ctx.sleep(Duration).await?
ctx.wait_for_event::<T>("event", timeout).await?
ctx.workflow_time()         // Deterministic time
ctx.is_step_completed("x")
ctx.get_step_result::<T>("x")

// Daemon only
ctx.shutdown_signal()       // Use in tokio::select! for graceful shutdown
ctx.is_shutdown_requested() // bool

// Webhook only
ctx.header("X-Key")         // Option<&str>
ctx.dispatch_job("name", args).await?  // -> Uuid
ctx.idempotency_key         // Option<String>
```

---

## Schema

```rust
#[forge::model]
pub struct User {
    pub id: Uuid,
    pub email: String,
    pub created_at: DateTime<Utc>,
}

// Stores as TEXT in DB with lowercase snake_case values: "pending", "active", "completed"
// Auto-generates: Display, FromStr, Serialize, Deserialize, sqlx::Encode/Decode
#[forge::forge_enum]
pub enum Status { Pending, Active, Completed }
```

---

## Errors

```rust
use forge::prelude::*;  // ForgeError, Result

ForgeError::NotFound("...")       // 404
ForgeError::Unauthorized("...")   // 401
ForgeError::Forbidden("...")      // 403
ForgeError::Validation("...")     // 400
ForgeError::Database(msg)         // 500
ForgeError::Internal(msg)         // 500
```

---

## Migrations

```sql
-- migrations/0001_users.sql
-- @up
CREATE TABLE users (id UUID PRIMARY KEY, email TEXT NOT NULL UNIQUE);
SELECT forge_enable_reactivity('users');  -- Required for subscriptions

-- @down
SELECT forge_disable_reactivity('users');
DROP TABLE IF EXISTS users;
```

---

## Testing (MANDATORY)

**Write tests for EVERY function you implement.** Place tests in the same file as the function. Run `cargo test` before delivering.

**Unsure about testing patterns? Web search "forge framework testing" or check tryforge.dev/docs.**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use forge::testing::*;

    // Shared DB setup - each test gets isolated transaction
    async fn setup() -> (IsolatedTestDb, PgPool) {
        let base = TestDatabase::embedded().await.unwrap();
        let db = base.isolated("test").await.unwrap();
        db.run_sql(&forge::get_internal_sql()).await.unwrap();
        db.migrate(Path::new("migrations")).await.unwrap();
        (db, db.pool().clone())
    }

    #[tokio::test]
    async fn test_query() {
        let (_db, pool) = setup().await;
        let ctx = TestQueryContext::builder()
            .as_user(Uuid::new_v4())
            .with_role("admin")
            .with_pool(pool)
            .build();

        // Call your query function and assert results
        let result = your_query(&ctx, args).await.unwrap();
        assert_eq!(result.field, expected);
    }

    #[tokio::test]
    async fn test_mutation_with_job_dispatch() {
        let (_db, pool) = setup().await;
        let ctx = TestMutationContext::builder()
            .as_user(Uuid::new_v4())
            .with_pool(pool)
            .mock_http_json("https://api.example.com/*", json!({"ok": true}))
            .build();

        // Call mutation
        your_mutation(&ctx, input).await.unwrap();

        // Verify side effects
        ctx.job_dispatch().assert_dispatched("job_name");
        ctx.http().assert_called("https://api.example.com/*");
    }

    #[tokio::test]
    async fn test_job_retry_behavior() {
        let ctx = TestJobContext::builder("job_name")
            .as_retry(2)  // Simulate 2nd attempt
            .with_max_attempts(5)
            .build();

        assert!(ctx.is_retry());
        assert!(!ctx.is_last_attempt());
    }

    #[tokio::test]
    async fn test_unauthorized_access() {
        let ctx = TestQueryContext::builder().build();  // No user
        let result = protected_query(&ctx).await;
        assert!(matches!(result, Err(ForgeError::Unauthorized(_))));
    }
}
```

**Test context builders:**
- `TestQueryContext::builder().as_user(id).with_role("x").with_pool(pool).build()`
- `TestMutationContext::builder().as_user(id).mock_http_json(pattern, response).build()`
- `TestJobContext::builder("name").as_retry(attempt).with_max_attempts(n).build()`
- `TestCronContext::builder().with_catch_up(true).build()`
- `TestWorkflowContext::builder().with_step_states(map).build()`
- `TestDaemonContext::builder("name").with_pool(pool).build()`
- `TestWebhookContext::builder("name").with_header("X-Key", "val").with_idempotency_key("k").build()`

---

## Frontend (Svelte 5)

```svelte
<script lang="ts">
  import { listUsers$, createUser } from '$lib/forge';
  const users = listUsers$();  // Runes-native reactive
</script>

{#if users.loading}Loading{:else if users.data}{#each users.data as u}{u.name}{/each}{/if}
```

**Generated:** `listUsers$()` reactive, `listUsers()` one-shot, `createUser({...})` mutation

**Best practices:**
- Split into components: `AuthForm.svelte`, `BookList.svelte`, `BookItem.svelte`
- Optimistic updates: set UI state before await, revert in catch
- Per-item loading: `let deleting = $state(false)` per component, not global
- Friendly errors: map `"invalid-credential"` → `"Invalid email or password"`
- Auth: call `getForgeClient().reconnect()` on login/logout state change

---

## Built-in Auth (HS256 JWT)

```toml
# forge.toml - jwt_secret via env var
[auth]
jwt_algorithm = "HS256"
jwt_secret = "${JWT_SECRET}"
```

```rust
// Generate token: ClaimsBuilder + Header::new(Algorithm::HS256)
let claims = ClaimsBuilder::new().user_id(user.id).duration_secs(7 * 24 * 3600).build()?;
let token = encode(&Header::new(Algorithm::HS256), &claims, &EncodingKey::from_secret(secret.as_bytes()))?;
```

`forge generate` auto-creates `auth.svelte.ts` with `getToken()` when auth configured.

---

## External Auth (JWT)

```toml
# forge.toml
[auth]
jwt_algorithm = "RS256"  # or HS256/HS384/HS512/RS384/RS512
jwks_url = "https://..."  # RS* only
jwt_issuer = "https://..."
jwt_audience = "your-project-id"
# jwt_secret = "${JWT_SECRET}"  # HS* only
```

```svelte
<ForgeProvider url={PUBLIC_API_URL} getToken={() => authProvider.getIdToken()}>
```

Use `ctx.require_subject()?` for non-UUID providers (Firebase/Auth0/Clerk).

---

## CLI

```bash
forge new <name> [--demo|--minimal]
forge dev [--docker] [--no-pg] [--takeover-ports]
forge dev --backend-port 8081 --frontend-port 4173
forge add query|mutation|job|cron|workflow|daemon|webhook <name>
forge generate          # TypeScript types
forge migrate up|down|status
forge check
```

---

## Dev Workflow

1. Schema (`src/schema/`) → 2. Migration → 3. `forge add <type> <name>` → 4. Implement + **write tests** → 5. Register in `main.rs` → 6. `forge generate` → 7. **Run verification**

**Stuck? Web search "forge framework [feature]" or "tryforge.dev [topic]" for examples and docs.**

---

## Project Structure

```
src/functions/   # One file per function + inline tests
src/schema/      # #[forge::model], #[forge::forge_enum]
migrations/      # NNNN_name.sql (-- @up / -- @down)
frontend/        # Svelte 5
```

---

## Rate Limiting

```rust
#[forge::query(rate_limit(requests = 100, per = "1m", key = "user"))]  // key: user|ip|tenant|global
```

Returns `ForgeError::RateLimitExceeded { retry_after, limit, remaining }`.

---

## Job Priority

`background` (0) < `low` (25) < `normal` (50, default) < `high` (75) < `critical` (100)
