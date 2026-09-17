# ADR 0002: Hosted free tiers over a single always-free VPS

Date: 2026-09-09
Status: accepted

## Context

The plan put everything on one Oracle Always Free ARM box. Two facts, verified 2026-09-09,
ended it: Oracle halved the A1 allocation to 2 OCPU and 12 GB in June 2026, enforced
18 August, and A1 capacity has been unavailable to free US accounts for months.

The replacement, a Hugging Face Docker Space, lasted four hours: those SDKs went paid in
July 2026. Two free tiers moved under this project in one day, so this optimises not for the
best free tier but for surviving the next move.

## Decision

No VPS. Actions for heavy compute (free and unlimited on public repos), Vercel Hobby as the
receiver, Neon for pgvector, Hugging Face dataset repos for the corpus.

**The receiver holds no state and touches no database:** verify the HMAC, fire
`repository_dispatch`, return 202. A test asserts importing it does not pull asyncpg, so CI
enforces this rather than a comment. Forty lines move host in an afternoon, and cheap
migration beats elegance.

## Rejected alternatives

**Cloudflare Workers.** More durable: 100,000 requests a day, no card. Rejected only for
being JavaScript. The closest call, and where to go if Vercel's terms change.

## Consequences

**Redelivery dedup moved and got more fragile.** Stateless, the receiver dispatches every
time; `review.yml` writes the delivery row first and skips when the id exists. Reorder those
and PR-Lens double-comments on strangers' pull requests, rising noise the only symptom.


Latency grows and belongs in p95. Vercel Hobby is non-commercial, so PR-Lens never charges.

## Cost of reversing

Low, deliberately. A VPS means a systemd unit and a different `DATABASE_URL`, under a day.
The one-way door would have been a stateful daemon.
