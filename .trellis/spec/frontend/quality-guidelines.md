# Quality Guidelines

> Code standards for the Vue 3 web console.

---

## Forbidden Patterns

| Pattern | Why |
|---------|-----|
| `var` | Use `const` or `let` |
| jQuery / DOM manipulation | Use Vue template bindings |
| Inline `<script>` in HTML | Keep logic in `app.js` |
| Sync XHR | Use `async`/`await` with `fetch` |
| Hardcoded API host | Use relative path: `const API = ''` |

---

## Required Patterns

```js
// 1. Destructure Vue globals once at top
const { createApp, ref, reactive, onMounted, onUnmounted, computed } = Vue;

// 2. Busy flags on all async actions
async function doLearn() {
  learnBusy.value = true;
  try {
    const result = await apiPost('/api/learn/text', { text: learnText.value });
    logResult('Learned: ' + result.entities.length + ' entities');
  } finally {
    learnBusy.value = false;
  }
}

// 3. Disable buttons while loading
// <button @click="doLearn" :disabled="learnBusy">Learn</button>

// 4. Result logging
function logResult(msg, type = 'info') {
  resultLog.value.push({ msg, type, time: new Date().toLocaleTimeString() });
  if (resultLog.value.length > 50) resultLog.value.shift();
}
```

---

## Testing Requirements

- No automated frontend tests (manual testing via browser)
- Verify all API endpoints work through the console after backend changes
- Check WebSocket reconnect: stop/start server, verify auto-reconnect

---

## Code Review Checklist

- [ ] No `var` declarations?
- [ ] All async actions have busy flags and `:disabled` bindings?
- [ ] API paths match backend routes exactly?
- [ ] WebSocket reconnect handles `onclose`?
- [ ] No hardcoded URLs (use `location.host`)?
