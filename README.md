# TitanML — AI Knowledge Nexus

An interactive knowledge graph and visual encyclopedia for AI/ML concepts. Built with Next.js 16, featuring a canvas-based force-directed graph, architecture visualizer, and structured learning paths.

## Features

- **Interactive Knowledge Graph** — Canvas-rendered force-directed graph with physics simulation, spatial grid optimization, and smooth zoom/pan. Click nodes to explore term definitions, relationships, and code examples.
- **Architecture Visualizer** — Visual flowcharts of AI architectures (RAG, Agentic, fine-tuning pipelines) with step-by-step breakdowns.
- **Knowledge Paths** — Curated learning curricula with difficulty levels, from foundational ML concepts to advanced topics like multi-agent systems.
- **Dark Mode** — Comprehensive theme system with 50+ CSS variables, smooth transitions, and canvas-aware dark rendering.
- **Search** — Debounced full-text search across term names, descriptions, and tags with real-time graph highlighting.
- **Add Terms** — Contribute new AI concepts directly through the in-app modal form.
- **Keyboard Shortcuts** — `/` to focus search, `Escape` to close panels, `Alt+Arrow` for history navigation.
- **Touch Support** — Pinch-zoom and tap-to-select on mobile devices.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Framework | Next.js 16 (App Router, Turbopack) |
| Language | TypeScript |
| Styling | Custom CSS with CSS variables, `@property` animations |
| State | Zustand (atomic selectors for render optimization) |
| Graph Engine | Custom Canvas 2D — force-directed physics, spatial grid partitioning, lerp-smoothed camera |
| Font | Plus Jakarta Sans (Google Fonts) |

## Project Structure

```
src/
├── app/
│   ├── globals.css          # Base reset, loading screen, toast, panel/modal animations,
│   │                         # scrollbar, code blocks, tooltip, accessibility
│   ├── layout.tsx           # Root layout (meta, fonts, CSS imports)
│   └── page.tsx             # Main page — LoadingScreen, Toast, Home (data orchestrator),
│                             # AppContent (all UI split into prop-driven component)
├── lib/
│   └── graph-engine.ts      # Canvas knowledge graph engine (~1010 lines)
│       ├── Physics: spatial grid O(n) neighbor lookup, repulsion/attraction/gravity
│       ├── Rendering: radial gradient bg, cached dot grid, curved edges with arrows,
│       │             quantum packet animations, gradient-filled nodes with glow
│       ├── Interaction: mouse drag/pan, wheel zoom (lerp-smoothed), touch pinch-zoom
│       ├── Performance: dirty flag + idle frame skipping with full rAF stop/wake,
│       │              label truncation cache, sorted node draw order cache
│       └── Dark mode: full theme swap via setDarkMode()
├── store/
│   └── useTitanMLStore.ts   # Zustand store — all app state and actions
└── public/
    ├── css/
    │   ├── styles.css           # Main component styles + dark mode overrides
    │   ├── architectureStyles.css  # Architecture gallery & flow visualizer
    │   └── knowledgePath.css       # Knowledge path gallery & timeline
    ├── data/
    │   ├── graphData.json      # Categories & terms with relationships
    │   ├── architecture.json   # Architecture definitions with steps
    │   └── knowledgePath.json  # Learning paths with steps/subpaths
    └── assets/
```

## Architecture Decisions

### Render Optimization
- **Atomic Zustand selectors** — Each UI section subscribes to only the state slices it needs, preventing unnecessary re-renders when unrelated state changes.
- **AppContent extraction** — The main UI is a separate function component from the data-loading orchestrator, allowing React to skip the entire AppContent tree during loading.
- **useMemo for derived data** — `relatedTerms`, `selectedTermCategory`, and `currentArch` are memoized to avoid recomputation on every render.
- **Graph engine idle stopping** — The canvas render loop fully stops after 60 idle frames (~1s at 60fps), then wakes on user interaction via `_wake()`. Saves CPU and battery.
- **Sorted node draw cache** — Node sort order (for z-ordering) is cached and only recomputed when selection changes via `_sortedDirty` flag.
- **Label truncation cache** — Truncated labels are cached by `nodeId:maxLabelPx` and cleared on data reload.

### Memory Management
- **Proper cleanup** — Graph engine `destroy()` clears all arrays, maps, caches, event listeners, and canvas references. `ResizeObserver` is disconnected on unmount.
- **No dead code** — Unused files (`db.ts`, `utils.ts`, `api/route.ts`, `components/ui/`, `hooks/`) removed. Unused `codeBlockRef` eliminated.
- **Event listener tracking** — Graph engine tracks all bound listeners in `_boundListeners[]` for complete cleanup.

### Animation System
- **Loading screen** — Dual-ring spinner with CSS `@keyframes`, emoji rotation, fade-in on mount via `loaderFadeIn` keyframe, and fade-out + scale transition on data load. Loader is conditionally rendered (removed from DOM after fade-out).
- **Toast** — Spring-based entrance (`translateY + scale + opacity`) and smooth exit animation with 300ms crossfade.
- **View transitions** — Architecture and knowledge path views conditionally render with `key` prop, triggering `viewFadeIn` animation on each navigation.
- **Panel** — Slides in with spring transition from `styles.css`. Panel sections stagger-animate on open.
- **Modal** — Overlay fades in, modal scales up with spring. Form fields stagger-animate.
- **Interactive elements** — Category/filter items translate on hover, buttons scale on press, related terms lift with shadow.

### Bug Fixes (Godmode Audit)
- **L1: Hydration mismatch** — `suppressHydrationWarning` on both `<html>` and `<body>` to handle browser extension attribute injection.
- **L1: _drawNodes array mutation** — Node drawing uses a separate `_sortedNodes` copy instead of mutating `this.nodes` during iteration.
- **L2: Critical `getTerm` ReferenceError** — `graph.onNodeSelect` callback was calling undefined `getTerm()`. Fixed to use `useTitanMLStore.getState().terms.find()`.
- **L2: Loading screen double-trigger** — `loaderFading` state was set but never applied to className. Now conditionally renders the fade-out loader div.
- **L2: Canvas render loop** — Added full rAF stop after 60 idle frames + `_wake()` mechanism on all interaction handlers.
- **L2: Label truncation cache** — `_labelCache` now cleared in `loadData()` to prevent stale entries after data reload.
- **L2: Panel/Modal entrance animations** — Added staggered section animations and form field animations.
- **L3: Visitor counter** — Repositioned to bottom-right corner, subtler opacity.
- **L3: Code block** — Removed unused `codeBlockRef`. Copy button is fully inline React state. Added `padding-top: 40px` for button clearance.
- **L3: View transitions** — Architecture and knowledge path views now conditionally render with keys, triggering proper `viewFadeIn` animation each time.
- **L3: Accessibility** — Added `:focus-visible` styles and `:focus:not(:focus-visible)` reset.
- **L3: Dead code** — Removed `db.ts`, `utils.ts`, `api/route.ts`, `components/ui/`, `hooks/`.

## Getting Started

```bash
# Install dependencies
npm install

# Development server
npm run dev

# Production build
npm run build
npm start
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## Credits

Created by **Abhishek Shah** — [Portfolio](https://abhiverse01.github.io) | [Email](mailto:abhishek.aimarine@gmail.com)

## License

All rights reserved.