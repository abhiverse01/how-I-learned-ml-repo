## Phase 1 Report — Audit & Performance

### Findings summary

| # | File | Category | Issue | Severity | Status |
|---|------|----------|-------|----------|--------|
| 1 | `src/lib/graph-engine.ts` | Correctness | `stopAnimation()` cancelled rAF but left `_isRunning = true`, so `visibilitychange` resume and `_wake()` both no-op'd — animation permanently died after any tab switch | **L1** | **Fixed** |
| 2 | `src/app/page.tsx` | Correctness | `onHoverChange` callback cast event to `MouseEvent` unconditionally; when triggered from `_onTouchEnd` (TouchEvent), `clientX` was `undefined`, producing `NaN` tooltip positioning | **L1** | **Fixed** |
| 3 | `src/lib/graph-engine.ts` | Correctness | `_onTouchEnd` did not call `e.preventDefault()`, allowing browser to fire a synthetic `click` event ~300ms after touch tap — caused double node-select pulse on mobile | **L2** | **Fixed** |
| 4 | `src/lib/graph-engine.ts` | Memory | Two `setTimeout` calls in `_onClick` and `_onTouchEnd` (tap pulse reset, 220ms) were untracked — if `destroy()` ran before the timer fired, the closure retained references to `this` and `node`, preventing GC | **L2** | **Fixed** |
| 5 | `src/lib/graph-engine.ts` | Algorithmic | `_drawEdges` creates a `ctx.createLinearGradient()` per visible edge per frame (~227 gradient objects/frame). Non-active edges all render at low opacity; a pre-computed single-color stroke would be cheaper. | L3 | Not fixed — acceptable at 87 nodes, degrades at 500+ |
| 6 | `src/lib/graph-engine.ts` | Algorithmic | `findNodeAt` is O(n) linear scan. At 87 nodes this is ~87 iterations per click; at 500 nodes it's 500. The spatial grid exists for physics but is not used for hit-testing. | L3 | Not fixed — acceptable at current scale |
| 7 | `src/lib/graph-engine.ts` | Algorithmic | `loadData` line 293 calls `categories.indexOf(category)` after building `catMap` — redundant O(n) scan. One-time init cost, negligible. | L3 | Not fixed — one-time cost |
| 8 | `src/app/page.tsx` | Algorithmic | Stats computation (byCategory + edge dedup) is duplicated between `page.tsx` lines 270-291 and `useTitanMLStore.ts` `computeStats()`. Both produce identical results; `computeStats` is only used by `addTerm`. | L3 | Not fixed — runs once on load, no perf impact |
| 9 | `public/data/graphData.json` | Correctness | 71 orphaned `related[]` references point to term IDs that do not exist in the dataset (e.g. `evaluation`, `alignment`, `vae`, `llama`). 4 of these (`evaluation`, `safety`, `multimodal`, `training`) are category IDs used as term IDs. Code handles this gracefully (edges to missing nodes are skipped), but these represent broken relationship links. | L2 | **Not fixed — needs your input** (create missing terms, or remove dangling refs) |
| 10 | `public/data/graphData.json` | Correctness | 2 terms (`dspy`, `llamaindex`) have `type: "framework"` which is not in `options.nodeRadius` — they fall back to default radius 16px instead of having a configured size. | L2 | **Not fixed — needs your input** (add `"framework"` to radius map, or reclassify) |
| 11 | `public/data/graphData.json` | Correctness | All 87 terms are missing the `createdAt` field. Code falls back to `new Date().toISOString()` at parse time (page.tsx line 259), so no runtime error, but all terms show "just now" timestamps. | L3 | **Not fixed — needs your input** (batch-add timestamps) |
| 12 | `public/data/graphData.json` | Redundancy | 49 relationship pairs are bidirectional (A→B and B→A). Code deduplicates via `edgeSet` in `loadData` (line 333), so no visual duplication, but 17.8% of relationship data is redundant. | L3 | Not fixed — data convention, not a bug |
| 13 | `src/hooks/use-toast.ts` | Memory/Correctness | `useEffect` dependency array is `[state]` instead of `[]`, causing listener teardown+re-attach on every state change. `TOAST_REMOVE_DELAY` is 1000000ms (~16.7 min). | L2 | Not fixed — **dead code** (never imported by active code; app uses its own Toast component) |
| 14 | `src/lib/db.ts` | Dead code | Imports `PrismaClient` from `@prisma/client`. Prisma is not used anywhere in the active application code. | — | **Flagged for removal** |
| 15 | `src/components/ui/*` (44 files) | Dead code | Full shadcn/ui component library (button, dialog, dropdown-menu, form, etc.) imported by nothing in the active codebase. App uses custom CSS only. | — | **Flagged for removal** |
| 16 | `src/hooks/use-mobile.ts` | Dead code | `useIsMobile()` hook — never imported. App does its own mobile detection inline in `page.tsx` line 201-206. | — | **Flagged for removal** |
| 17 | `src/lib/utils.ts` | Dead code | `cn()` utility using `clsx` + `tailwind-merge` — only consumed by the dead `src/components/ui/*` files. | — | **Flagged for removal** |
| 18 | `package.json` deps | Dead code | `class-variance-authority`, `clsx`, `tailwind-merge` are transitive deps only used by dead shadcn/ui components. `@prisma/client` is unused. Removing all would reduce install size but requires `npm uninstall`. | — | **Flagged for removal** (requires your sign-off) |

### Benchmarks

> **Note:** A genuine browser profiling environment is not available in this session. The following is **static complexity analysis and allocation counting**, not measured frame times. Label accordingly.

| Metric | Analysis (87 nodes) | Analysis (200 nodes) | Analysis (500 nodes) | Technique used |
|--------|---------------------|----------------------|----------------------|----------------|
| Physics per-frame | O(n × k) where k = avg grid neighbors (~6-8) = ~600 ops | ~1600 ops | ~4000 ops | Spatial grid (CELL=200) limits repulsion checks from O(n²) to O(n × k) |
| Edge rendering per-frame | 227 edges × (gradient alloc + bezier draw) | ~400 edges × same | ~1000 edges × same | Each edge creates a `createLinearGradient()` — could be solid color for non-active |
| Node rendering per-frame | 87 nodes × (radialGradient + shadow + fill + label) | 200 × same | 500 × same | Label cache hits after radii stabilize; shadow per node is the main GPU cost |
| Hit detection (click/tap) | O(n) linear scan = 87 iterations | 200 iterations | 500 iterations | `findNodeAt` iterates all nodes; spatial grid not used here |
| Zoom/pan interaction latency | Lerp-based (0.18 factor/rAF) — perceived latency ~3-4 frames = 50-67ms at 60fps | Same | Same | No computation bottleneck; limited by rAF cadence |
| Memory per frame (allocations) | ~227 gradient objects + ~87 radial gradient objects + grid cache (reused) | ~400 + ~200 | ~1000 + ~500 | Gradient objects are GC'd each frame; grid cache is reused until dirty |
| Spatial grid: neighbors checked per node | ~6-10 (CELL=200, nodes spread across ~20-30 cells) | ~8-14 | ~10-18 | Grid cell size 200px vs node spacing ~180px means most neighbors are in 3×3 neighborhood |

**Key bottleneck at scale:** `createLinearGradient()` per edge per frame. At 500 nodes with ~1000 edges, this is 1000 gradient allocations per frame. Replacing non-active edge strokes with a single pre-computed RGBA color (via `_hexToRgba` which is already used elsewhere) would eliminate ~997 gradient allocations per frame.

### Verified-still-fixed (from README's prior audit)

| Item | Status | Evidence |
|------|--------|----------|
| Hydration mismatch | ✅ Still fixed | `isMobile` initialized `useState(false)`, set via client-only `useEffect` (page.tsx 201-206) |
| `_drawNodes` mutation | ✅ Still fixed | Uses `this.nodes.slice()` to create `_sortedNodes` (graph-engine.ts 900), splices from copy |
| `getTerm` ReferenceError | ✅ Still fixed | Uses `get().terms.find()` (useTitanMLStore.ts 265), safe getter pattern |
| Loading screen double-trigger | ✅ Still fixed | Guard `if (!isLoaded \|\| !canvasRef.current \|\| graphRef.current) return` (page.tsx 320) |
| Render loop idle-stop | ✅ Still fixed | `_idleFrames >= _MAX_IDLE` check stops rAF (graph-engine.ts 728), `_wake()` restarts on interaction |
| Label cache | ✅ Still fixed | `_labelCache` Map with composite key `${node.id}:${maxLabelPx}` (graph-engine.ts 1002-1008) |
| Panel/modal animations | ✅ Still fixed | `.panel.open .panel-section` staggered animation (globals.css 151-158), `.modal-overlay.open .form-group` staggered (globals.css 172-178) |
| Visitor counter | ✅ Still fixed | DOM element at page.tsx 1302-1304, styled via globals.css 181-199 |
| Code block ref | ✅ Still fixed | Copy button uses inline React state `copyState` (page.tsx 439-467), comment at line 190 confirms removal |
| View transitions | ✅ Still fixed | `.view-container` class with `viewFadeIn` animation (globals.css 135-148) |
| Focus-visible | ✅ Still fixed | `:focus-visible` styles in globals.css 392-396, plus styles.css 278-283 |
| Dead code removal | ✅ Still fixed (partial) | `codeBlockRef` removed (line 190 comment). Additional dead code identified in this audit (see #14-18) |

### Fixes NOT made (and why)

| Issue | Why not fixed |
|-------|--------------|
| 71 orphaned relationship refs in graphData.json | Data quality decision — requires your input on whether to create missing terms or remove dangling refs. Code handles missing refs gracefully (edges skipped). |
| 2 invalid term types (`dspy`, `llamaindex` type `"framework"`) | Requires your decision: add `"framework"` to `options.nodeRadius` map, or reclassify these terms. Currently they render at default 16px radius. |
| Missing `createdAt` on all 87 terms | Data migration task — needs your preferred timestamp. Code falls back gracefully. |
| 49 bidirectional redundant edges | Data convention — code deduplicates at load time via `edgeSet`. Removing one direction from each pair is a data decision. |
| `use-toast.ts` dependency array bug and TOAST_REMOVE_DELAY | Dead code — this hook is never imported. The app has its own Toast component. Removing the file is recommended. |
| Edge gradient per-frame allocation (L3) | Acceptable at current 87 nodes. At 500+ nodes, replace non-active edge `createLinearGradient()` with solid `strokeStyle = _hexToRgba(color, opacity)`. Flagged for future optimization. |
| `findNodeAt` O(n) hit detection (L3) | Acceptable at current scale. Spatial grid exists for physics but not hit detection. Future optimization if node count grows significantly. |
| 44 dead shadcn/ui component files + db.ts + hooks + utils.ts | **Flagged for your sign-off.** These files and their package dependencies (`class-variance-authority`, `clsx`, `tailwind-merge`, `@prisma/client`) are never imported by active code. Removing them reduces install size and eliminates confusion about what's in use. I recommend removal but will not delete without your approval since this is a production codebase about to ship. |
| `handleAddTerm` graph reload resets all positions | When a user adds a term via the modal, `graphRef.current.loadData(cats, termsData)` reinitializes the entire graph — all node positions reset and physics re-simulates from scratch. Fixing this requires implementing an incremental `addNode()` method. Flagged for your input on desired behavior. |