# State Management

> Vue 3 Composition API only. No Pinia, no Vuex, no store.

---

## State Categories

### Local UI state (ref)

```js
const learnText = ref('');       // form input
const learnBusy = ref(false);    // loading flag
const qaHistory = ref([]);       // operation log
const resultLog = ref([]);       // result history
```

### Shared reactive objects (reactive)

```js
const health = reactive({
  status: '-', version: '-', stage: '-', total_steps: 0
});
const stats = reactive({});
const emotionState = reactive({
  valence: 0, arousal: 0, dominance: 0, label: '-'
});
```

### Server state (polled)

```js
// Fetched every 2 seconds via setInterval
async function fetchStats() {
  Object.assign(stats, await apiGet('/api/stats'));
}
```

### WebSocket state (pushed)

```js
function handleWSEvent(event) {
  switch (event.type) {
    case 'learning_step':
      progressData.value.push(event.data);
      updateProgressChart();
      break;
    case 'emotion':
      Object.assign(emotionState, event.data);
      break;
  }
}
```

---

## When to Use What

| Data source | Pattern | Example |
|-------------|---------|---------|
| Form inputs | `ref('')` | `learnText`, `chatInput` |
| Loading flags | `ref(false)` | `learnBusy`, `qaBusy` |
| Lists | `ref([])` | `qaHistory`, `societyAgents` |
| Merged objects | `reactive({})` | `health`, `stats` |
| Chart data | `ref([])` | `progressData`, `graphNodes` |

---

## Common Mistakes

- **Don't add Pinia/Vuex** — the app is one `setup()`, state is shared via closures
- **Don't use `reactive` for arrays** — use `ref([])`, reassign with `.value = [...]`
- **Don't forget `Object.assign` for reactive** — direct `stats = {...}` breaks reactivity
- **Don't poll for data that has WebSocket push** — listen to WS events instead
