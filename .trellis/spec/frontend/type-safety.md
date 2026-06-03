# Type Safety

> No TypeScript. Type discipline via consistent naming and structure.

---

## Overview

This project has no TypeScript, no JSDoc types, no build step. Type safety comes from consistent naming conventions and matching backend JSON schemas.

---

## Type Organization

API response shapes match the backend `dto.hpp` serialization exactly:

| API Endpoint | Response Shape | JS Access Pattern |
|--------------|---------------|-------------------|
| `GET /api/health` | `{status, version, stage, total_steps, engine_type}` | `health.value.version` |
| `GET /api/stats` | `{key: number}` flat map | `stats.triples_count` |
| `POST /api/learn/text` | `{entities, triples, verification_score}` | `result.entities.length` |
| `POST /api/reason` | `[{content, confidence, method}]` | `results[0].content` |
| `POST /api/emotion` | `{valence, arousal, dominance, label}` | `emotionState.valence` |
| WS event | `{type, timestamp, data}` | `event.type`, `event.data` |

---

## Validation

No runtime validation library. Validate by checking required fields:

```js
async function apiPost(path, body = {}) {
  const r = await fetch(API + path, { ... });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}
```

Server-side validation is in the C++ route handlers (checks for required JSON fields).

---

## Common Patterns

```js
// Safe optional chaining for nested data
const entityCount = result?.entities?.length ?? 0;

// Default values via reactive init
const health = reactive({ status: '-', version: '-', total_steps: 0 });

// Enum mapping (match backend strategy_type_to_string)
const strategyNames = {
  rote_memorization: 'Rote Memorization',
  spaced_repetition: 'Spaced Repetition',
  // ...
};
```

---

## Forbidden Patterns

- **No `as` type casts** — not available without TypeScript
- **No runtime type checking libraries** — keep it simple
- **Don't assume response shape** — always handle `undefined`/`null` with `?.`
