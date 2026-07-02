"""
Phase 2: Scientific/Technical Design System Overhaul
Transforms styles.css for instrument-panel aesthetic.
"""

import re

with open('/home/z/my-project/public/css/styles.css', 'r') as f:
    css = f.read()

# ============================================================
# 1. FONT SYSTEM — Add monospace pairing
# ============================================================
css = css.replace(
    "  --font-mono:   'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace;",
    "  --font-mono:   'JetBrains Mono', ui-monospace, 'Cascadia Code', 'Fira Code', monospace;\n  --font-mono-weight: 500;"
)

# ============================================================
# 2. COLOR PALETTE — Scientific/Technical instrument panel
# ============================================================
css = css.replace(
    """  --bg-primary:   #f7f9fc;
  --bg-secondary: #ffffff;
  --bg-tertiary:  #f0f4fa;
  --bg-hover:     #eaf0f8;
  --bg-active:    #dde7f4;""",
    """  --bg-primary:   #f8f9fb;
  --bg-secondary: #ffffff;
  --bg-tertiary:  #eef1f5;
  --bg-hover:     #e6eaf0;
  --bg-active:    #dce1e8;"""
)

css = css.replace(
    """  --text-primary:   #0d1829;
  --text-secondary: #3d526b;
  --text-tertiary:  #607080;
  --text-muted:     #8fa3bb;""",
    """  --text-primary:   #111827;
  --text-secondary: #374151;
  --text-tertiary:  #6b7280;
  --text-muted:     #9ca3af;"""
)

css = css.replace(
    """  --accent-primary:   #0891b2;
  --accent-secondary: #6366f1;
  --accent-tertiary:  #8b5cf6;""",
    """  --accent-primary:   #06b6d4;
  --accent-secondary: #64748b;
  --accent-tertiary:  #475569;"""
)

css = css.replace(
    """  --accent-primary-rgb:   8, 145, 178;
  --accent-secondary-rgb: 99, 102, 241;
  --accent-tertiary-rgb:  139, 92, 246;""",
    """  --accent-primary-rgb:   6, 182, 212;
  --accent-secondary-rgb: 100, 116, 139;
  --accent-tertiary-rgb:  71, 85, 105;"""
)

# ============================================================
# 3. SHADOW SYSTEM — Tight, precise instead of diffuse
# ============================================================
css = css.replace(
    """  --shadow-xs:  0 1px 2px rgba(0, 0, 0, 0.04);
  --shadow-sm:  0 1px 3px rgba(0, 0, 0, 0.06), 0 1px 2px rgba(0, 0, 0, 0.04);
  --shadow-md:  0 4px 12px rgba(0, 0, 0, 0.06), 0 2px 4px rgba(0, 0, 0, 0.04);
  --shadow-lg:  0 12px 24px rgba(0, 0, 0, 0.07), 0 4px 8px rgba(0, 0, 0, 0.04);
  --shadow-xl:  0 20px 40px rgba(0, 0, 0, 0.09), 0 8px 16px rgba(0, 0, 0, 0.04);
  --shadow-2xl: 0 32px 64px rgba(0, 0, 0, 0.14), 0 16px 32px rgba(0, 0, 0, 0.06);""",
    """  --shadow-xs:  0 1px 2px rgba(0, 0, 0, 0.05);
  --shadow-sm:  0 1px 3px rgba(0, 0, 0, 0.07), 0 1px 2px rgba(0, 0, 0, 0.04);
  --shadow-md:  0 2px 8px rgba(0, 0, 0, 0.08), 0 1px 3px rgba(0, 0, 0, 0.04);
  --shadow-lg:  0 4px 16px rgba(0, 0, 0, 0.08), 0 2px 4px rgba(0, 0, 0, 0.04);
  --shadow-xl:  0 8px 24px rgba(0, 0, 0, 0.10), 0 4px 8px rgba(0, 0, 0, 0.04);
  --shadow-2xl: 0 16px 40px rgba(0, 0, 0, 0.12), 0 8px 16px rgba(0, 0, 0, 0.04);"""
)

# ============================================================
# 4. GLOW SYSTEM — Restrained 1px accent borders
# ============================================================
css = css.replace(
    """  --glow-primary:   0 0 0 1px rgba(var(--accent-primary-rgb), 0.15),
                    0 0 20px rgba(var(--accent-primary-rgb), 0.25),
                    0 0 40px rgba(var(--accent-primary-rgb), 0.10);
  --glow-secondary: 0 0 0 1px rgba(var(--accent-secondary-rgb), 0.15),
                    0 0 20px rgba(var(--accent-secondary-rgb), 0.25);
  --glow-active:    0 0 8px rgba(var(--accent-primary-rgb), 0.4),
                    0 0 20px rgba(var(--accent-primary-rgb), 0.15);""",
    """  --glow-primary:   0 0 0 1px rgba(var(--accent-primary-rgb), 0.25);
  --glow-secondary: 0 0 0 1px rgba(var(--accent-secondary-rgb), 0.20);
  --glow-active:    0 0 0 1px rgba(var(--accent-primary-rgb), 0.5),
                    0 0 6px rgba(var(--accent-primary-rgb), 0.12);"""
)

# ============================================================
# 5. GLASS — Less blur, more opacity
# ============================================================
css = css.replace(
    """  --glass-light: rgba(255, 255, 255, 0.82);
  --glass-medium: rgba(255, 255, 255, 0.65);
  --glass-dark: rgba(255, 255, 255, 0.45);
  --glass-blur: blur(22px) saturate(180%);
  --glass-blur-heavy: blur(32px) saturate(200%);""",
    """  --glass-light: rgba(255, 255, 255, 0.92);
  --glass-medium: rgba(255, 255, 255, 0.80);
  --glass-dark: rgba(255, 255, 255, 0.60);
  --glass-blur: blur(16px) saturate(160%);
  --glass-blur-heavy: blur(24px) saturate(180%);"""
)

# ============================================================
# 6. SPACING SCALE — Document 4px base, tighten radii
# ============================================================
css = css.replace(
    """  /* ── Sizing ── */
  --radius-xs:  4px;
  --radius-sm:  6px;
  --radius-md:  10px;
  --radius-lg:  14px;
  --radius-xl:  18px;
  --radius-2xl: 26px;
  --radius-full: 9999px;""",
    """  /* ── Spacing Scale (4px base unit) ── */
  --space-1:  4px;   --space-2:  8px;   --space-3:  12px;
  --space-4:  16px;  --space-5:  20px;  --space-6:  24px;
  --space-8:  32px;  --space-10: 40px; --space-12: 48px;
  --space-16: 64px;

  /* ── Sizing ── */
  --radius-xs:  4px;
  --radius-sm:  6px;
  --radius-md:  8px;
  --radius-lg:  12px;
  --radius-xl:  16px;
  --radius-2xl: 24px;
  --radius-full: 9999px;"""
)

# ============================================================
# 7. DARK MODE — True near-black
# ============================================================
css = css.replace(
    """[data-theme="dark"] {
  color-scheme: dark;

  --bg-primary:   #0a1120;
  --bg-secondary: #111827;
  --bg-tertiary:  #1e2d40;
  --bg-hover:     #1e2d40;
  --bg-active:    #2d3f58;

  --text-primary:   #e8f0fe;
  --text-secondary: #b4c6e0;
  --text-tertiary:  #7c96b4;
  --text-muted:     #4e6480;

  --border-light:  #162030;
  --border-medium: #1e2d40;
  --border-dark:   #2d3f58;

  --glass-light:  rgba(17, 24, 39, 0.88);
  --glass-medium: rgba(17, 24, 39, 0.72);
  --glass-dark:   rgba(17, 24, 39, 0.55);

  --shadow-xs:  0 1px 2px rgba(0, 0, 0, 0.5);
  --shadow-sm:  0 1px 3px rgba(0, 0, 0, 0.6), 0 1px 2px rgba(0, 0, 0, 0.35);
  --shadow-md:  0 4px 12px rgba(0, 0, 0, 0.55), 0 2px 4px rgba(0, 0, 0, 0.35);
  --shadow-lg:  0 12px 24px rgba(0, 0, 0, 0.65), 0 4px 8px rgba(0, 0, 0, 0.35);
  --shadow-xl:  0 20px 40px rgba(0, 0, 0, 0.75), 0 8px 16px rgba(0, 0, 0, 0.35);
  --shadow-2xl: 0 32px 64px rgba(0, 0, 0, 0.85), 0 16px 32px rgba(0, 0, 0, 0.5);
}""",
    """[data-theme="dark"] {
  color-scheme: dark;

  /* True near-black — instrument panel dark */
  --bg-primary:   #09090b;
  --bg-secondary: #111113;
  --bg-tertiary:  #18181b;
  --bg-hover:     #1e1e22;
  --bg-active:    #27272a;

  --text-primary:   #f4f4f5;
  --text-secondary: #d4d4d8;
  --text-tertiary:  #a1a1aa;
  --text-muted:     #71717a;

  --border-light:  #27272a;
  --border-medium: #3f3f46;
  --border-dark:   #52525b;

  --glass-light:  rgba(9, 9, 11, 0.90);
  --glass-medium: rgba(9, 9, 11, 0.78);
  --glass-dark:   rgba(9, 9, 11, 0.60);

  --shadow-xs:  0 1px 2px rgba(0, 0, 0, 0.40);
  --shadow-sm:  0 1px 3px rgba(0, 0, 0, 0.50), 0 1px 2px rgba(0, 0, 0, 0.30);
  --shadow-md:  0 2px 8px rgba(0, 0, 0, 0.45), 0 1px 3px rgba(0, 0, 0, 0.25);
  --shadow-lg:  0 4px 16px rgba(0, 0, 0, 0.50), 0 2px 4px rgba(0, 0, 0, 0.25);
  --shadow-xl:  0 8px 24px rgba(0, 0, 0, 0.55), 0 4px 8px rgba(0, 0, 0, 0.25);
  --shadow-2xl: 0 16px 40px rgba(0, 0, 0, 0.60), 0 8px 16px rgba(0, 0, 0, 0.30);
}"""
)

# ============================================================
# 8. HEADER — Flatten, precise border
# ============================================================
css = css.replace(
    """.header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: var(--header-height);
  padding: 0 16px;
  background: rgba(255, 255, 255, 0.78);
  backdrop-filter: var(--glass-blur);
  -webkit-backdrop-filter: var(--glass-blur);
  border-bottom: 1px solid rgba(255, 255, 255, 0.9);
  flex-shrink: 0;
  z-index: var(--z-overlay);
  position: sticky;
  top: 0;
  isolation: isolate;
  /* Layered shadow for perceived elevation */
  box-shadow:
    0 1px 0 rgba(255, 255, 255, 0.8) inset,
    0 -1px 0 rgba(0, 0, 0, 0.03) inset,
    var(--shadow-sm);""",
    """.header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: var(--header-height);
  padding: 0 var(--space-4);
  background: var(--bg-secondary);
  border-bottom: 1px solid var(--border-light);
  flex-shrink: 0;
  z-index: var(--z-overlay);
  position: sticky;
  top: 0;
  isolation: isolate;
  box-shadow: var(--shadow-xs);"""
)

# ============================================================
# 9. LOGO — Solid accent, no gradient
# ============================================================
css = css.replace(
    """.logo-icon {
  width: 36px;
  height: 36px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
  border-radius: var(--radius-lg);
  color: white;
  box-shadow:
    0 4px 14px rgba(var(--accent-primary-rgb), 0.35),
    inset 0 1px 0 rgba(255, 255, 255, 0.25);
  transition:
    transform var(--transition-base),
    box-shadow var(--transition-base);
  flex-shrink: 0;
  position: relative;
  overflow: hidden;
}""",
    """.logo-icon {
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--accent-primary);
  border-radius: var(--radius-md);
  color: white;
  box-shadow: none;
  transition:
    transform var(--transition-base),
    box-shadow var(--transition-base);
  flex-shrink: 0;
  position: relative;
}"""
)

# Remove shine overlay
css = css.replace(
    """/* Shine overlay on logo */
.logo-icon::before {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(135deg, rgba(255,255,255,0.3) 0%, transparent 60%);
  pointer-events: none;
}""",
    "/* Logo: clean, no decorative overlay */"
)

# Simplify hover
css = css.replace(
    """.logo-icon:hover {
  transform: scale(1.1) rotate(-5deg);
  box-shadow:
    0 8px 24px rgba(var(--accent-primary-rgb), 0.5),
    inset 0 1px 0 rgba(255, 255, 255, 0.3);
  animation: logoFloat 2s var(--ease-spring) infinite;
}""",
    """.logo-icon:hover {
  transform: scale(1.05);
  box-shadow: 0 2px 8px rgba(var(--accent-primary-rgb), 0.3);
}"""
)

# ============================================================
# 10. HEADER SHIMMER — Single accent line
# ============================================================
css = css.replace(
    """/* Prismatic shimmer accent on header bottom edge */
.header::after {
  content: '';
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  height: 1px;
  background: linear-gradient(
    90deg,
    transparent 0%,
    rgba(var(--accent-primary-rgb), 0.4) 30%,
    rgba(var(--accent-secondary-rgb), 0.4) 70%,
    transparent 100%
  );
  opacity: 0.6;
}""",
    """/* Header bottom accent — single precise line */
.header::after {
  content: '';
  position: absolute;
  bottom: -1px;
  left: 0;
  right: 0;
  height: 1px;
  background: var(--accent-primary);
  opacity: 0.2;
}"""
)

# ============================================================
# 11. LOGO TEXT — Solid, no gradient text
# ============================================================
css = css.replace(
    """.logo-text {
  font-size: var(--font-size-md);
  font-weight: 800;
  letter-spacing: -0.03em;
  background: linear-gradient(
    135deg,
    var(--text-primary) 0%,
    var(--text-secondary) 100%
  );
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
  white-space: nowrap;
}""",
    """.logo-text {
  font-size: var(--font-size-md);
  font-weight: 700;
  letter-spacing: -0.03em;
  color: var(--text-primary);
  white-space: nowrap;
}"""
)

# ============================================================
# 12. GRAPH CONTROLS — Tighter
# ============================================================
css = css.replace(
    """.graph-controls {
  position: absolute;
  bottom: var(--space-4);
  right: var(--space-4);
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
  z-index: 10;
}

.control-btn {
  width: 36px;
  height: 36px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--glass-light);
  backdrop-filter: var(--glass-blur);
  -webkit-backdrop-filter: var(--glass-blur);
  border: 1px solid var(--border-light);
  border-radius: var(--radius-md);
  color: var(--text-secondary);
  cursor: pointer;
  transition: all var(--transition-base);
  box-shadow: var(--shadow-sm);
}""",
    """.graph-controls {
  position: absolute;
  bottom: var(--space-4);
  right: var(--space-4);
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
  z-index: 10;
}

.control-btn {
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--bg-secondary);
  border: 1px solid var(--border-light);
  border-radius: var(--radius-sm);
  color: var(--text-secondary);
  cursor: pointer;
  transition: all var(--transition-base);
  box-shadow: var(--shadow-xs);
}"""
)

# ============================================================
# 13. STAT VALUE — Monospace
# ============================================================
css = css.replace(
    """.stat-value {
  font-family: var(--font-family);
  font-size: var(--font-size-xl);
  font-weight: 700;
  letter-spacing: -0.02em;""",
    """.stat-value {
  font-family: var(--font-mono);
  font-size: var(--font-size-xl);
  font-weight: var(--font-mono-weight);
  letter-spacing: -0.03em;
  font-variant-numeric: tabular-nums;"""
)

# ============================================================
# 14. CATEGORY COUNT — Monospace
# ============================================================
css = css.replace(
    """.category-count {
  margin-left: auto;
  font-size: var(--font-size-xs);
  font-weight: 600;
  color: var(--text-muted);
  background: var(--bg-tertiary);
  padding: 2px 8px;
  border-radius: var(--radius-full);
  min-width: 28px;
  text-align: center;""",
    """.category-count {
  margin-left: auto;
  font-family: var(--font-mono);
  font-size: var(--font-size-xs);
  font-weight: var(--font-mono-weight);
  color: var(--text-muted);
  background: var(--bg-tertiary);
  padding: 2px 8px;
  border-radius: var(--radius-full);
  min-width: 28px;
  text-align: center;
  font-variant-numeric: tabular-nums;"""
)

# ============================================================
# 15. FILTER COUNT — Monospace
# ============================================================
css = css.replace(
    """.filter-count {
  margin-left: auto;
  font-size: var(--font-size-xs);
  font-weight: 600;
  color: var(--text-muted);
  font-variant-numeric: tabular-nums;""",
    """.filter-count {
  margin-left: auto;
  font-family: var(--font-mono);
  font-size: var(--font-size-xs);
  font-weight: var(--font-mono-weight);
  color: var(--text-muted);
  font-variant-numeric: tabular-nums;"""
)

# ============================================================
# 16. TAG — Monospace
# ============================================================
css = css.replace(
    """.tag {
  display: inline-block;
  padding: 3px 10px;
  font-size: var(--font-size-xs);
  font-weight: 600;
  color: var(--accent-primary);
  background: rgba(var(--accent-primary-rgb), 0.08);
  border-radius: var(--radius-full);
  border: 1px solid rgba(var(--accent-primary-rgb), 0.15);
  letter-spacing: 0.01em;
  white-space: nowrap;""",
    """.tag {
  display: inline-block;
  padding: 3px 10px;
  font-family: var(--font-mono);
  font-size: var(--font-size-xs);
  font-weight: var(--font-mono-weight);
  color: var(--accent-primary);
  background: rgba(var(--accent-primary-rgb), 0.06);
  border-radius: var(--radius-full);
  border: 1px solid rgba(var(--accent-primary-rgb), 0.15);
  letter-spacing: 0.02em;
  white-space: nowrap;"""
)

# ============================================================
# 17. VISITOR COUNTER — Monospace
# ============================================================
css = css.replace(
    """.visitor-counter {
  position: fixed;
  bottom: 6px;
  right: 12px;
  text-align: right;
  padding: 4px 8px;
  font-size: 0.7rem;
  color: var(--text-muted);
  opacity: 0.5;
  z-index: 50;
  font-family: 'Plus Jakarta Sans', sans-serif;
  pointer-events: none;""",
    """.visitor-counter {
  position: fixed;
  bottom: 6px;
  right: 12px;
  text-align: right;
  padding: 4px 8px;
  font-family: var(--font-mono);
  font-size: 0.65rem;
  color: var(--text-muted);
  opacity: 0.4;
  z-index: 50;
  pointer-events: none;"""
)

# ============================================================
# 18. PANEL — Tighter shadow
# ============================================================
css = css.replace(
    """  box-shadow: -8px 0 32px rgba(0, 0, 0, 0.08), -2px 0 8px rgba(0, 0, 0, 0.04);
  transform: translateX(100%);
  transition: transform 420ms var(--ease-spring);
  z-index: var(--z-panel);""",
    """  box-shadow: -4px 0 16px rgba(0, 0, 0, 0.06);
  transform: translateX(100%);
  transition: transform 380ms var(--ease-spring);
  z-index: var(--z-panel);"""
)

# ============================================================
# 19. SIDEBAR — Flat, no glass
# ============================================================
css = css.replace(
    """.sidebar {
  width: var(--sidebar-width);
  background: var(--glass-light);
  backdrop-filter: var(--glass-blur);
  -webkit-backdrop-filter: var(--glass-blur);
  border-right: 1px solid var(--border-light);
  box-shadow: var(--shadow-md);""",
    """.sidebar {
  width: var(--sidebar-width);
  background: var(--bg-secondary);
  border-right: 1px solid var(--border-light);
  box-shadow: var(--shadow-sm);"""
)

# Dark sidebar
css = css.replace(
    """[data-theme="dark"] .sidebar {
  background: rgba(10, 17, 32, 0.92);
  border-right: 1px solid rgba(255, 255, 255, 0.05);""",
    """[data-theme="dark"] .sidebar {
  background: var(--bg-secondary);
  border-right: 1px solid var(--border-light);"""
)

# ============================================================
# 20. SEARCH KBD — Monospace
# ============================================================
css = css.replace(
    """  font-family: inherit;
  font-weight: 600;
  color: var(--text-muted);
  background: var(--bg-secondary);
  border: 1px solid var(--border-light);
  border-radius: var(--radius-sm);
  box-shadow:
    0 1px 0 rgba(0, 0, 0, 0.08),
    0 2px 0 rgba(0, 0, 0, 0.05),
    inset 0 -1px 0 rgba(0, 0, 0, 0.06);""",
    """  font-family: var(--font-mono);
  font-weight: var(--font-mono-weight);
  color: var(--text-muted);
  background: var(--bg-secondary);
  border: 1px solid var(--border-light);
  border-radius: var(--radius-sm);
  box-shadow: var(--shadow-xs);"""
)

# ============================================================
# 21. MODAL — Tighter
# ============================================================
css = css.replace(
    """  box-shadow: var(--shadow-2xl);
  position: relative;
}""",
    """  box-shadow: var(--shadow-lg);
  position: relative;
}""")

# ============================================================
# 22. STAT CARD — No shadow, border only
# ============================================================
css = css.replace(
    """  box-shadow: var(--shadow-sm);
  border: 1px solid var(--border-light);
  transition: all var(--transition-base);
}""",  # First occurrence (stat-card)
    """  box-shadow: none;
  border: 1px solid var(--border-light);
  transition: all var(--transition-base);
}""")

# ============================================================
# 23. LEGEND — Flat, no glass
# ============================================================
css = css.replace(
    """.legend {
  position: absolute;
  top: var(--space-4);
  left: var(--space-4);
  background: var(--glass-light);
  backdrop-filter: var(--glass-blur);
  -webkit-backdrop-filter: var(--glass-blur);
  border: 1px solid var(--border-light);
  border-radius: var(--radius-lg);
  padding: var(--space-3) var(--space-4);
  box-shadow: var(--shadow-md);""",
    """.legend {
  position: absolute;
  top: var(--space-4);
  left: var(--space-4);
  background: var(--bg-secondary);
  border: 1px solid var(--border-light);
  border-radius: var(--radius-md);
  padding: var(--space-3) var(--space-4);
  box-shadow: var(--shadow-xs);"""
)

# ============================================================
# 24. DARK HEADER
# ============================================================
css = css.replace(
    """[data-theme="dark"] .header {
  background: rgba(10, 17, 32, 0.86);
  border-bottom: 1px solid rgba(255, 255, 255, 0.04);""",
    """[data-theme="dark"] .header {
  background: var(--bg-secondary);
  border-bottom: 1px solid var(--border-light);"""
)

# ============================================================
# 25. PULSE GLOW — Replace with opacity pulse
# ============================================================
css = css.replace(
    """@keyframes pulseGlow {
  0%, 100% { box-shadow: 0 0 6px currentColor; }
  50%       { box-shadow: 0 0 14px currentColor, 0 0 28px currentColor; }
}""",
    """@keyframes pulseGlow {
  0%, 100% { opacity: 1; }
  50%       { opacity: 0.7; }
}"""
)

# ============================================================
# 26. DARK MODE PANEL
# ============================================================
css = css.replace(
    """[data-theme="dark"] .panel {
  background: rgba(10, 17, 32, 0.96);
  border-left: 1px solid rgba(255, 255, 255, 0.05);""",
    """[data-theme="dark"] .panel {
  background: var(--bg-secondary);
  border-left: 1px solid var(--border-light);"""
)

# ============================================================
# 27. DARK MODE LEGEND
# ============================================================
css = css.replace(
    """[data-theme="dark"] .legend {
  background: rgba(10, 17, 32, 0.90);
  border: 1px solid rgba(255, 255, 255, 0.05);""",
    """[data-theme="dark"] .legend {
  background: var(--bg-secondary);
  border: 1px solid var(--border-light);"""
)

# ============================================================
# 28. DARK MODE MODAL
# ============================================================
css = css.replace(
    """[data-theme="dark"] .modal {
  background: rgba(10, 17, 32, 0.96);
  border: 1px solid rgba(255, 255, 255, 0.05);""",
    """[data-theme="dark"] .modal {
  background: var(--bg-secondary);
  border: 1px solid var(--border-light);"""
)

# ============================================================
# 29. DARK MODE CONTROLS
# ============================================================
css = css.replace(
    """[data-theme="dark"] .graph-controls {
  background: rgba(10, 17, 32, 0.80);
  border: 1px solid rgba(255, 255, 255, 0.05);""",
    """[data-theme="dark"] .graph-controls {
  background: var(--bg-secondary);
  border: 1px solid var(--border-light);"""
)

# ============================================================
# 30. DARK MODE STAT CARD
# ============================================================
css = css.replace(
    """[data-theme="dark"] .stat-card {
  background: rgba(10, 17, 32, 0.70);
  border: 1px solid rgba(255, 255, 255, 0.05);""",
    """[data-theme="dark"] .stat-card {
  background: var(--bg-tertiary);
  border: 1px solid var(--border-light);"""
)

# ============================================================
# 31. DARK MODE TOOLTIP
# ============================================================
css = css.replace(
    """[data-theme="dark"] .tooltip {
  background: rgba(10, 17, 32, 0.96);
  border: 1px solid rgba(255, 255, 255, 0.08);""",
    """[data-theme="dark"] .tooltip {
  background: var(--bg-tertiary);
  border: 1px solid var(--border-light);"""
)

# ============================================================
# WRITE OUTPUT
# ============================================================
with open('/home/z/my-project/public/css/styles.css', 'w') as f:
    f.write(css)

print("styles.css: Phase 2 transformation complete")
print(f"Total lines: {len(css.splitlines())}")