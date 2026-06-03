# Hook Guidelines

> No custom Vue hooks. Use `onMounted`/`onUnmounted` for lifecycle, native `fetch` for data.

---

## Overview

No composable functions (`use*` pattern). All stateful logic lives directly in `setup()`. Data fetching uses native `fetch()` wrapped in simple async helpers.

---

## Data Fetching

Two global helper functions, no library:

```js
async function apiGet(path) {
  const r = await fetch(API + path);
  return r.json();
}

async function apiPost(path, body = {}) {
  const r = await fetch(API + path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return r.json();
}
```

Usage in setup:

```js
async function fetchHealth() {
  health.value = await apiGet('/api/health');
  connected.value = true;
}

async function doLearn() {
  learnBusy.value = true;
  const result = await apiPost('/api/learn/text', { text: learnText.value });
  learnBusy.value = false;
  logResult('Learned: ' + result.entities.length + ' entities');
}
```

---

## WebSocket Connection

Managed manually in `setup()` with reconnect logic:

```js
function connectWS() {
  const ws = new WebSocket(WS_URL);
  ws.onopen = () => { wsConnected.value = true; };
  ws.onmessage = (e) => { handleWSEvent(JSON.parse(e.data)); };
  ws.onclose = () => {
    wsConnected.value = false;
    setTimeout(connectWS, 3000);  // auto-reconnect
  };
}
```

---

## Lifecycle

```js
onMounted(() => {
  initCharts();
  connectWS();
  fetchHealth();
  timer = setInterval(fetchStats, 2000);
});

onUnmounted(() => {
  clearInterval(timer);
});
```

---

## Common Mistakes

- **Don't add Axios or SWR** — native `fetch` is sufficient for this scale
- **Don't forget to set `busy` flags** — use `ref(false)` to disable buttons during requests
- **Don't create WebSocket per component** — single connection in setup(), shared via closures
