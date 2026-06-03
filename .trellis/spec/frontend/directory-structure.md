# Directory Structure

> Three-file SPA. No src/ tree, no components/ directory.

---

## Directory Layout

```
ai-learning-cpp/web/
├── index.html    # Template + Vue 3 + ECharts CDN imports
├── app.js        # All application logic (setup, API, WS, chart init)
└── style.css     # Global styles (CSS custom properties for theming)
```

That is the entire frontend. No `node_modules/`, no `package.json`, no `vite.config.js`.

---

## Module Organization

Since everything lives in `app.js` (~600 lines), logical sections are separated by comment blocks:

```js
/* ── API helper ─────────────────────────────────────── */
async function apiGet(path) { ... }
async function apiPost(path, body = {}) { ... }

/* ── Vue app ────────────────────────────────────────── */
const app = createApp({
  setup() {
    // reactive state
    // chart init functions
    // WebSocket connection
    // API action functions (doLearn, doReason, etc.)
    // lifecycle hooks (onMounted, onUnmounted)
    // return template bindings
  }
});
```

---

## Naming Conventions

- **Reactive state**: `camelCase` — `learnText`, `qaHistory`, `chatBusy`
- **Action functions**: `do<Action>` — `doLearn()`, `doReason()`, `doInsight()`
- **Fetch functions**: `fetch<Thing>` — `fetchHealth()`, `fetchStats()`
- **Chart refs**: `<name>ChartRef` for template ref, `<name>Chart` for ECharts instance
- **CSS**: BEM-lite with `.card`, `.card-title`, `.stat-pill`, `.stat-row`
