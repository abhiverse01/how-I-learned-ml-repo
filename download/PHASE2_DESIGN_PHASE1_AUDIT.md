## Phase 1 — Visual Debt Audit

### Effect inventory (by surface)

| Surface | Gradients | Box-Shadows | Backdrop-Filters | Glow Effects |
|---|---|---|---|---|
| Header/Sidebar | 23 | 36 | 3 | 12 |
| Graph Controls/Legend | 8 | 4 | 4 | 1 |
| Detail Panel | 6 | 12 | 2 | 0 |
| Modal/Form | 1 | 10 | 3 | 0 |
| Architecture Visualizer | 40 | 57 | 3 | 30 |
| Knowledge Path | 25 | 32 | 0 | 12 |
| Footer/Creator Card | 3 | 4 | 0 | 2 |
| Global/Shared | 8 | 18 | 1 | 10 |
| **TOTAL** | **114** | **173** | **16** | **67** |

### Duplicated cross-file patterns (12 found)

| # | Pattern | Files | Fix |
|---|---------|-------|-----|
| D1 | Scrollbar gradient thumb | styles.css, arch.css, kp.css | Shared global rule |
| D2 | 5-layer grid background (radial + 4 linear) | arch.css, kp.css | Shared `.view-surface` class |
| D3 | Prismatic 3-color top bar | arch.css, kp.css | `--prismatic-bar` variable |
| D4 | Shimmer sweep gradient | styles.css, arch.css, kp.css | `--shimmer-gradient` variable |
| D5 | Card hover shadow (20px/48px accent blur) | arch.css, kp.css | `--card-hover-shadow` variable |
| D6 | Glass shine overlay | arch.css, kp.css | Shared class |
| D7 | Separator gradient line | arch.css, kp.css | `--separator-gradient` variable |
| D8 | Multi-stop spine gradient | arch.css, kp.css | `--spine-gradient` variable |
| D9 | Accent pip gradient (primary→secondary) | styles.css, arch.css, kp.css | `--accent-pip` variable |
| D10 | Dark mode grid background | arch.css, kp.css | Same as D2 |
| D11 | Glow halo under separator | arch.css, kp.css | Shared pseudo-element |
| D12 | Spine dot glow (3-ring shadow) | arch.css, kp.css | `--spine-dot-glow` variable |

### Scaffolding verdict: ALL DEAD

| Item | Verdict | Evidence |
|------|---------|----------|
| shadcn/ui (`src/components/ui/*`, 48 files) | **DEAD** | Zero imports from `src/app/` or `src/store/` |
| Tailwind CSS | **DEAD** | No `@tailwind` directives in any CSS; all classes are custom |
| Prisma (`src/lib/db.ts`) | **DEAD** | Zero imports from live code; no API routes exist |
| `src/lib/utils.ts` (`cn()`) | **DEAD** | Only consumed by dead shadcn components |
| `src/hooks/use-toast.ts`, `use-mobile.ts` | **DEAD** | Zero imports; app uses own toast + inline mobile detection |
| `examples/websocket/` | **DEAD** | Zero references from `src/` |
| Framer Motion | **DEAD** | Zero imports; all animations via CSS @keyframes |
| `class-variance-authority`, `clsx`, `tailwind-merge` | **DEAD** | Only imported by dead shadcn files |

### Fixed-grid flexibility risks found

| File:Line | Class | Current | Action |
|-----------|-------|---------|--------|
| styles.css:1420 | `.related-grid` | `repeat(2, 1fr)` | **Convert** to `repeat(auto-fill, minmax(160px, 1fr))` |
| styles.css:920 | `.stats-grid` | `repeat(3, 1fr)` | Keep (intentional 3-column layout) |
| styles.css:1721 | `.form-row` | `repeat(2, 1fr)` | Keep (intentional 2-column form layout) |
| architectureStyles.css:1052 | `.parallel-lanes` | `1fr 1fr` | **Convert** to `repeat(auto-fill, minmax(280px, 1fr))` |
| arch.css gallery, kp.css gallery | `.arch-gallery`, `.path-gallery` | Already use `auto-fill`/`auto-fit` | No change needed |

### Dark mode variable coverage table

| Variable | Light Value | Dark Value | Status | Notes |
|----------|-------------|------------|--------|-------|
| `--bg-primary` | `#f8f9fb` | `#09090b` | ✅ Covered | |
| `--bg-secondary` | `#ffffff` | `#111113` | ✅ Covered | |
| `--bg-tertiary` | `#eef1f5` | `#18181b` | ✅ Covered | |
| `--bg-hover` | `#e6eaf0` | `#1e1e22` | ✅ Covered | |
| `--bg-active` | `#dce1e8` | `#27272a` | ✅ Covered | |
| `--text-primary` | `#111827` | `#f4f4f5` | ✅ Covered | 18.1:1 on bg-primary ✅ |
| `--text-secondary` | `#374151` | `#d4d4d8` | ✅ Covered | 12.76:1 on bg-secondary ✅ |
| `--text-tertiary` | `#6b7280` | `#a1a1aa` | ✅ Covered | 7.76:1 on bg-primary ✅ |
| `--text-muted` | `#9ca3af` | `#71717a` | ✅ Covered | **4.12:1 on bg-primary ❌ FAIL WCAG AA** |
| `--border-light` | `#dde5f0` | `#27272a` | ✅ Covered | |
| `--border-medium` | `#bfcfe0` | `#3f3f46` | ✅ Covered | |
| `--border-dark` | `#8fa3bb` | `#52525b` | ✅ Covered | |
| `--accent-primary` | `#0891b2` | *(inherited)* | ⚠️ No override | 5.4:1 ✅ |
| `--accent-secondary` | `#6366f1` | *(inherited)* | ⚠️ No override | |
| `--accent-tertiary` | `#8b5cf6` | *(inherited)* | ⚠️ No override | |
| `--accent-success` | `#10b981` | *(inherited)* | ⚠️ No override | |
| `--accent-warning` | `#f59e0b` | *(inherited)* | ⚠️ No override | 9.26:1 ✅ |
| `--accent-danger` | `#ef4444` | *(inherited)* | ⚠️ No override | |
| `--accent-primary-rgb` | `6, 182, 212` | *(inherited)* | **🔴 BUG** | Should be `8, 145, 178` to match `#0891b2` |
| `--accent-secondary-rgb` | `100, 116, 139` | *(inherited)* | **🔴 BUG** | Should be `99, 102, 241` to match `#6366f1` |
| `--accent-tertiary-rgb` | `71, 85, 105` | *(inherited)* | **🔴 BUG** | Should be `139, 92, 246` to match `#8b5cf6` |
| `--cat-*` (8 vars) | various | *(inherited)* | ⚠️ No override | Acceptable for category identification |
| `--glow-*` (3 vars) | uses -rgb vars | *(inherited)* | ⚠️ No override | Will auto-fix when RGB triplets are corrected |
| All shadow vars | various | ✅ All covered | ✅ | Dark shadows have dramatically higher opacity |
| All glass vars | various | ✅ All covered | ✅ | Dark glass uses near-black base |
| All spacing/radius/z-index/easing/font/layout/safe-area vars | — | — | N/A | Non-thematic, correctly omitted from dark block |

### Critical bug: Accent RGB triplets don't match hex values

| Variable | Hex says | RGB actually is | Effect |
|----------|----------|-----------------|--------|
| `--accent-primary-rgb` | `#0891b2` → (8,145,178) | `6,182,212` → `#06b6d4` (Cyan-500) | Slight mismatch |
| `--accent-secondary-rgb` | `#6366f1` → (99,102,241) | `100,116,139` → `#64748b` (Slate-500) | **SEVERE** — renders Slate gray glows, not Indigo |
| `--accent-tertiary-rgb` | `#8b5cf6` → (139,92,246) | `71,85,105` → `#475569` (Slate-600) | **SEVERE** — renders Slate gray glows, not Violet |

### Hardcoded values needing variables (~45+ instances)

- `code-block`: 4 hardcoded hex values for bg/border/fg with no CSS variable (`#08101f`, `#152035`, `#c5d5f0`, dark `#020509`, `#0e1a2e`)
- `knowledgePath.css`: **26 instances** of `rgba(99, 102, 241, ...)` — should use `rgba(var(--accent-secondary-rgb), ...)`
- Modal overlay: `rgba(10, 18, 35, 0.52)` — no variable
- Sidebar overlay: `rgba(8, 16, 32, 0.42)` — no variable
- Dark logo icon gradient: `#22d3ee, #818cf8` — should use accent variables