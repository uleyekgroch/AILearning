# Component Guidelines

> Single `createApp()` with `setup()`. No `.vue` files, no SFC.

---

## Overview

The entire UI is one Vue 3 app instance with Composition API (`setup()`). Template is in `index.html` using Vue template syntax directly in the DOM. No component files.

---

## Component Structure

```html
<!-- index.html: template uses Vue directives on plain HTML -->
<div class="card">
  <div class="card-title">Q&A</div>
  <input v-model="qaQuestion" placeholder="Ask a question..." />
  <button @click="doThink" :disabled="qaBusy">Think</button>
  <div v-for="(item, i) in qaHistory" :key="i" class="log-item">
    {{ item }}
  </div>
</div>
```

All data and methods come from the `setup()` return in `app.js`.

---

## Template Patterns

```html
<!-- Conditional rendering -->
<span v-if="wsConnected" style="color:var(--accent-green)">WS</span>

<!-- List rendering with :key -->
<div v-for="(item, i) in resultLog" :key="i" :class="'log-' + item.type">
  {{ item.msg }}
</div>

<!-- Two-way binding -->
<textarea v-model="learnText" rows="3" placeholder="Enter text..."></textarea>

<!-- Event handlers -->
<button @click="doLearn" :disabled="learnBusy">Learn</button>
```

---

## Styling Patterns

- CSS custom properties for theming (in `style.css` `:root`)
- Layout via CSS Grid / Flexbox
- Cards with `.card` class, pill badges with `.stat-pill`
- No CSS-in-JS, no scoped styles

```css
:root {
  --bg-primary: #0f0f1a;
  --accent-green: #4caf50;
  --accent-purple: #7c4dff;
}
```

---

## Common Mistakes

- **Don't split into .vue files** — this project has no build step
- **Don't import Vue locally** — it's a global CDN script, use destructuring:
  `const { createApp, ref, reactive } = Vue;`
- **Don't use Options API** — always use `setup()` with Composition API
