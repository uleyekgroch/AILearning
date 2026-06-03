# Frontend Development Guidelines

> Vue 3 SPA served from C++ server. No build step, no npm.

---

## Overview

Single-page dashboard at `web/` — three files total (`index.html`, `app.js`, `style.css`). Loaded directly by the browser from the Crow static file server. No TypeScript, no bundler, no npm.

---

## Guidelines Index

| Guide | Description |
|-------|-------------|
| [Directory Structure](./directory-structure.md) | 3-file SPA layout |
| [Component Guidelines](./component-guidelines.md) | Vue 3 Options-free, setup() only |
| [Hook Guidelines](./hook-guidelines.md) | Native fetch + WebSocket patterns |
| [State Management](./state-management.md) | `ref()`/`reactive()` in setup() |
| [Quality Guidelines](./quality-guidelines.md) | ECharts integration, naming |
| [Type Safety](./type-safety.md) | No TypeScript, JSDoc optional |

---

## Key Conventions

- **Framework**: Vue 3 via CDN (`<script src="vue.global.prod.js">`)
- **Charts**: ECharts via CDN
- **API calls**: Native `fetch()` wrapped in `apiGet()`/`apiPost()` helpers
- **Real-time**: WebSocket to `/ws/events` and `/ws/stats`
- **No build step**: edit files, refresh browser
