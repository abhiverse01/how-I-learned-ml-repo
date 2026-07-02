/**
 * AI Knowledge Graph Visualization Engine
 * Converted from graph.js - Canvas-based force-directed graph with physics simulation.
 */

export interface GraphNode {
  id: string
  x: number
  y: number
  vx: number
  vy: number
  radius: number
  term: {
    id: string
    name: string
    fullName: string
    category: string
    type: string
    shortDesc: string
    tags: string[]
  }
  color: string
  highlighted: boolean
  visible: boolean
  currentRadius: number
  targetRadius: number
}

export interface GraphEdge {
  source: string
  target: string
  strength: number
}

export interface GraphCategory {
  id: string
  name: string
  color: string
}

export type NodeSelectCallback = (term: GraphNode['term']) => void
export type HoverChangeCallback = (node: GraphNode | null, e: MouseEvent | TouchEvent | null) => void

export class KnowledgeGraph {
  canvas: HTMLCanvasElement | null = null
  ctx: CanvasRenderingContext2D | null = null

  theme = {
    bg: '#f8f9fb',
    bgGradientCenter: '#ffffff',
    gridDot: '#d4d8e0',
    text: '#374151',
    nodeBorderDefault: '#d1d5db',
    nodeLabelDefault: '#6b7280',
    nodeBodyStart: '#ffffff',
    nodeBodyEnd: '#f3f4f6',
    dimmedBodyStart: 'rgba(255,255,255,0.4)',
    dimmedBodyEnd: 'rgba(243,244,246,0.3)',
  }

  private _lightTheme = { ...this.theme }

  private _darkTheme = {
    bg: '#09090b',
    bgGradientCenter: '#111113',
    gridDot: '#1c1c1f',
    text: '#f4f4f5',
    nodeBorderDefault: '#27272a',
    nodeLabelDefault: '#a1a1aa',
    nodeBodyStart: '#18181b',
    nodeBodyEnd: '#111113',
    dimmedBodyStart: 'rgba(24,24,27,0.4)',
    dimmedBodyEnd: 'rgba(17,17,19,0.3)',
  }

  options = {
    nodeRadius: { core: 26, technique: 18, infrastructure: 14, application: 12 } as Record<string, number>,
    fontSize: 10,
    padding: 80,
    zoomMin: 0.15,
    zoomMax: 5,
  }

  nodes: GraphNode[] = []
  edges: GraphEdge[] = []
  nodeMap = new Map<string, GraphNode>()
  zoom = 1
  panX = 0
  panY = 0
  hoveredNode: GraphNode | null = null
  private _selectedNode: GraphNode | null = null
  get selectedNode(): GraphNode | null { return this._selectedNode }
  set selectedNode(value: GraphNode | null) {
    if (this._selectedNode === value) return
    this._selectedNode = value
    this._sortedDirty = true
    this._isDirty = true
    this._wake()
  }
  isDragging = false
  lastMouse = { x: 0, y: 0 }
  animationId: number | null = null

  width = 0
  height = 0
  centerX = 0
  centerY = 0
  dpr = 1

  time = 0
  private _lastTs: number | null = null

  physics = {
    enabled: true,
    repulsion: 800,
    attraction: 0.005,
    centerGravity: 0.01,
    damping: 0.85,
    minVelocity: 0.05,
    maxVelocity: 10,
  }

  private _zoomTarget = 1
  private _panTargetX = 0
  private _panTargetY = 0
  private _pinchDist: number | null = null
  private _pinchZoom = 1
  private _touchStartPos: { x: number; y: number } | null = null
  private _touchStartTime = 0
  private _TAP_MAX_DIST = 10
  private _TAP_MAX_MS = 250
  private _gridCanvas: HTMLCanvasElement | null = null
  private _gridDirty = true
  private _boundListeners: { target: EventTarget; type: string; fn: EventListener; opts?: AddEventListenerOptions }[] = []
  private _isDark = false
  private _sortedNodes: GraphNode[] = []
  private _sortedDirty = true
  private _labelCache = new Map<string, string>()
  private _isDirty = true
  private _idleFrames = 0
  private _MAX_IDLE = 60 // Stop rAF after 60 idle frames (~1s at 60fps)
  private _isRunning = false
  private _mouseDownPos: { x: number; y: number } | null = null
  private _resizeTimer: ReturnType<typeof setTimeout> | null = null
  private _warmUpTimers: ReturnType<typeof setTimeout>[] = []
  private _tapResetTimers: ReturnType<typeof setTimeout>[] = []

  /** Wake the animation loop if it was stopped */
  private _wake() {
    if (!this._isRunning) {
      this._isDirty = true
      this._idleFrames = 0
      this._startAnimation()
    }
  }

  onNodeSelect: NodeSelectCallback | null = null
  onHoverChange: HoverChangeCallback | null = null

  constructor() {}

  init(canvas: HTMLCanvasElement) {
    this.canvas = canvas
    this.ctx = canvas.getContext('2d')
    if (!this.ctx) {
      console.error('[KnowledgeGraph] Could not get 2d context')
      return
    }
    this._handleResize()
    this._bindEvents()
    this._startAnimation()
  }

  findNode(id: string): GraphNode | null {
    return this.nodeMap.get(id) ?? null
  }

  resize() {
    this._handleResize()
  }

  zoomIn() {
    const newZoom = Math.min(this.options.zoomMax, this.zoom * 1.3)
    this._animateZoom(newZoom, this.width / 2, this.height / 2)
  }

  zoomOut() {
    const newZoom = Math.max(this.options.zoomMin, this.zoom / 1.3)
    this._animateZoom(newZoom, this.width / 2, this.height / 2)
  }

  resetView() {
    this._zoomTarget = 1
    this._panTargetX = 0
    this._panTargetY = 0
  }

  setDarkMode(isDark: boolean) {
    if (this._isDark === isDark) return
    this._isDark = isDark
    const t = isDark ? this._darkTheme : this._lightTheme
    Object.assign(this.theme, t)
    this._gridDirty = true
    this._isDirty = true
    this._wake()
  }

  highlightNodes(query: string | null) {
    const q = query ? String(query).toLowerCase().trim() : ''
    if (!q) {
      for (let i = 0; i < this.nodes.length; i++) this.nodes[i].highlighted = false
    } else {
      for (let i = 0; i < this.nodes.length; i++) {
        const node = this.nodes[i]
        node.highlighted =
          node.term.name.toLowerCase().includes(q) ||
          (node.term.shortDesc ?? '').toLowerCase().includes(q) ||
          (node.term.tags ?? []).some((t) => t.toLowerCase().includes(q))
      }
    }
    this._isDirty = true
    this._wake()
  }

  filterByCategory(categoryId: string | null) {
    this.nodes.forEach((node) => {
      node.visible = !categoryId || node.term.category === categoryId
    })
    this._isDirty = true
    this._wake()
  }

  filterByType(type: string | null) {
    this.nodes.forEach((node) => {
      node.visible = !type || node.term.type === type
    })
    this._isDirty = true
    this._wake()
  }

  stopAnimation() {
    if (this.animationId) {
      cancelAnimationFrame(this.animationId)
      this.animationId = null
    }
    this._isRunning = false
  }

  destroy() {
    this.stopAnimation()
    this._isRunning = false
    for (const { target, type, fn, opts } of this._boundListeners) {
      target.removeEventListener(type, fn, opts)
    }
    this._boundListeners = []
    if (this._resizeTimer) {
      clearTimeout(this._resizeTimer)
      this._resizeTimer = null
    }
    for (const id of this._warmUpTimers) clearTimeout(id)
    this._warmUpTimers = []
    for (const id of this._tapResetTimers) clearTimeout(id)
    this._tapResetTimers = []
    this._gridCanvas = null
    this.nodes = []
    this.edges = []
    this.nodeMap.clear()
    this._sortedNodes = []
    this._labelCache.clear()
    this.canvas = null
    this.ctx = null
  }

  loadData(categories: GraphCategory[], terms: GraphNode['term'][]) {
    this.nodes = []
    this.edges = []
    this.nodeMap.clear()
    this._labelCache.clear()
    this._sortedNodes = []
    this._sortedDirty = true

    if (!categories.length || !terms.length) return

    // Pre-build category index and group terms by category (O(n) instead of O(n²))
    const catMap = new Map<string, GraphCategory>()
    for (let ci = 0; ci < categories.length; ci++) catMap.set(categories[ci].id, categories[ci])
    const catTermsMap = new Map<string, { term: GraphNode['term']; idx: number }[]>()
    for (let ti = 0; ti < terms.length; ti++) {
      const t = terms[ti]
      const arr = catTermsMap.get(t.category)
      if (arr) arr.push({ term: t, idx: ti })
      else catTermsMap.set(t.category, [{ term: t, idx: ti }])
    }

    terms.forEach((term, termGlobalIdx) => {
      const category = catMap.get(term.category)
      const categoryIndex = category ? categories.indexOf(category) : 0
      const total = categories.length || 1

      const baseAngle = (categoryIndex / total) * Math.PI * 2 - Math.PI / 2
      const catTermsArr = catTermsMap.get(term.category)
      const termIndex = catTermsArr ? catTermsArr.findIndex((e) => e.idx === termGlobalIdx) : 0

      const radius = this.options.nodeRadius[term.type] ?? 16
      const distance = term.type === 'core' ? 160 : term.type === 'technique' ? 220 : 280

      const spreadAngle = Math.PI / 4
      const catTermsLen = catTermsArr ? catTermsArr.length : 1
      const angleOffset =
        catTermsLen > 1
          ? ((termIndex - (catTermsLen - 1) / 2) * spreadAngle) / catTermsLen
          : 0
      const angle = baseAngle + angleOffset

      const jx = (Math.random() - 0.5) * 40
      const jy = (Math.random() - 0.5) * 40

      const node: GraphNode = {
        id: term.id,
        x: this.centerX + Math.cos(angle) * distance + jx,
        y: this.centerY + Math.sin(angle) * distance + jy,
        vx: 0,
        vy: 0,
        radius,
        term,
        color: category?.color ?? '#94a3b8',
        highlighted: false,
        visible: true,
        currentRadius: radius,
        targetRadius: radius,
      }

      this.nodes.push(node)
      this.nodeMap.set(term.id, node)
    })

    const edgeSet = new Set<string>()
    terms.forEach((term) => {
      const relatedIds = (term as unknown as { related?: string[] }).related ?? []
      relatedIds.forEach((relId: string) => {
        const key = [term.id, relId].sort().join('|')
        if (!edgeSet.has(key)) {
          edgeSet.add(key)
          this.edges.push({ source: term.id, target: relId, strength: 1 })
        }
      })
    })

    this._warmUpPhysics(120)
  }

  private _warmUpPhysics(iterations: number) {
    const CHUNK = 60
    let done = 0
    const run = () => {
      const end = Math.min(done + CHUNK, iterations)
      for (; done < end; done++) this._simulatePhysics(0.15)
      if (done < iterations) {
        const id = setTimeout(run, 0)
        this._warmUpTimers.push(id)
      }
    }
    run()
  }

  private _simulatePhysics(dt = 1): boolean {
    if (!this.physics.enabled) return false

    const nodes = this.nodes
    const CELL = 200
    const eps = 0.001

    const grid = new Map<string, GraphNode[]>()
    const cellOf = (x: number, y: number) => `${Math.floor(x / CELL)},${Math.floor(y / CELL)}`

    nodes.forEach((node) => {
      if (!node.visible) return
      const key = cellOf(node.x, node.y)
      if (!grid.has(key)) grid.set(key, [])
      grid.get(key)!.push(node)
    })

    nodes.forEach((node) => {
      if (!node.visible) return

      node.vx += (this.centerX - node.x) * this.physics.centerGravity * dt
      node.vy += (this.centerY - node.y) * this.physics.centerGravity * dt

      // Inline neighbor iteration — avoids allocating a new array per node per frame
      const cx = Math.floor(node.x / CELL)
      const cy = Math.floor(node.y / CELL)
      for (let dx = -1; dx <= 1; dx++) {
        for (let dy = -1; dy <= 1; dy++) {
          const key = `${cx + dx},${cy + dy}`
          const cell = grid.get(key)
          if (!cell) continue
          for (let ci = 0; ci < cell.length; ci++) {
            const other = cell[ci]
            if (node.id === other.id) continue
            const ddx = node.x - other.x
            const ddy = node.y - other.y
            const distSq = ddx * ddx + ddy * ddy
            const dist = Math.max(Math.sqrt(distSq), 1)
            if (dist < CELL) {
              const force = this.physics.repulsion / (distSq + eps)
              node.vx += (ddx / dist) * force * dt
              node.vy += (ddy / dist) * force * dt
            }
          }
        }
      }
    })

    this.edges.forEach((edge) => {
      const src = this.nodeMap.get(edge.source)
      const tgt = this.nodeMap.get(edge.target)
      if (!src || !tgt || !src.visible || !tgt.visible) return

      const dx = tgt.x - src.x
      const dy = tgt.y - src.y
      const dist = Math.max(Math.sqrt(dx * dx + dy * dy), 1)
      const targetDist = 180
      const force = (dist - targetDist) * this.physics.attraction

      src.vx += (dx / dist) * force * dt
      src.vy += (dy / dist) * force * dt
      tgt.vx -= (dx / dist) * force * dt
      tgt.vy -= (dy / dist) * force * dt
    })

    const maxV = this.physics.maxVelocity
    const padding = this.options.padding
    const minV = this.physics.minVelocity
    let moved = false

    nodes.forEach((node) => {
      if (!node.visible) return

      node.vx *= this.physics.damping
      node.vy *= this.physics.damping

      const v = Math.sqrt(node.vx * node.vx + node.vy * node.vy)
      if (v > maxV) {
        node.vx = (node.vx / v) * maxV
        node.vy = (node.vy / v) * maxV
      }

      if (Math.abs(node.vx) > minV) { node.x += node.vx; moved = true }
      if (Math.abs(node.vy) > minV) { node.y += node.vy; moved = true }

      node.x = Math.max(padding, Math.min(this.width - padding, node.x))
      node.y = Math.max(padding, Math.min(this.height - padding, node.y))

      const radiusDelta = (node.targetRadius - node.currentRadius) * 0.12
      if (Math.abs(radiusDelta) > 0.01) {
        node.currentRadius += radiusDelta
        moved = true
      }
    })

    return moved
  }

  screenToWorld(sx: number, sy: number) {
    return { x: (sx - this.panX) / this.zoom, y: (sy - this.panY) / this.zoom }
  }

  private findNodeAt(wx: number, wy: number, extra = 0): GraphNode | null {
    for (let i = this.nodes.length - 1; i >= 0; i--) {
      const node = this.nodes[i]
      if (!node.visible) continue
      const dx = wx - node.x
      const dy = wy - node.y
      if (Math.sqrt(dx * dx + dy * dy) <= node.radius + 5 + extra) return node
    }
    return null
  }

  private _clientPos(e: MouseEvent) {
    const rect = this.canvas!.getBoundingClientRect()
    return { x: e.clientX - rect.left, y: e.clientY - rect.top }
  }

  private _onMouseMove = (e: MouseEvent) => {
    this._wake()
    const { x, y } = this._clientPos(e)
    if (this.isDragging) {
      const dx = e.clientX - this.lastMouse.x
      const dy = e.clientY - this.lastMouse.y
      this.panX += dx
      this.panY += dy
      this._panTargetX = this.panX
      this._panTargetY = this.panY
      this.lastMouse = { x: e.clientX, y: e.clientY }
    } else {
      const world = this.screenToWorld(x, y)
      const hovered = this.findNodeAt(world.x, world.y)
      if (hovered !== this.hoveredNode) {
        this.hoveredNode = hovered
        this.canvas!.style.cursor = hovered ? 'pointer' : 'grab'
        this._isDirty = true
        if (this.onHoverChange) this.onHoverChange(hovered, e)
      }
    }
  }

  private _onMouseDown = (e: MouseEvent) => {
    this._wake()
    this.isDragging = true
    this._mouseDownPos = { x: e.clientX, y: e.clientY }
    this.lastMouse = { x: e.clientX, y: e.clientY }
    this.canvas!.style.cursor = 'grabbing'
  }

  private _onMouseUp = () => {
    this.isDragging = false
    this.canvas!.style.cursor = this.hoveredNode ? 'pointer' : 'grab'
  }

  private _onMouseLeave = () => {
    this.isDragging = false
    this.hoveredNode = null
    if (this.onHoverChange) this.onHoverChange(null, null)
  }

  private _onWheel = (e: WheelEvent) => {
    this._wake()
    e.preventDefault()
    const { x, y } = this._clientPos(e)
    const delta = e.deltaY > 0 ? 0.9 : 1.1
    const newZoom = Math.max(this.options.zoomMin, Math.min(this.options.zoomMax, this.zoom * delta))
    this._animateZoom(newZoom, x, y)
  }

  private _onClick = (e: MouseEvent) => {
    if (e.clientX === undefined && e.clientY === undefined) return
    // If mouse moved more than 5px from mousedown position, this was a drag/pan — skip click
    if (this._mouseDownPos) {
      const dx = e.clientX - this._mouseDownPos.x
      const dy = e.clientY - this._mouseDownPos.y
      if (Math.sqrt(dx * dx + dy * dy) > 5) return
    }
    const { x, y } = this._clientPos(e)
    const world = this.screenToWorld(x, y)
    const node = this.findNodeAt(world.x, world.y)

    if (node) {
      this.selectedNode = node
      node.targetRadius = node.radius * 1.2
      const resetId = setTimeout(() => {
        if (this.selectedNode === node) node.targetRadius = node.radius
      }, 220)
      this._tapResetTimers.push(resetId)
      if (this.onNodeSelect) this.onNodeSelect(node.term)
    } else {
      this.selectedNode = null
    }
  }

  private _onTouchStart = (e: TouchEvent) => {
    this._wake()
    if (e.touches.length === 1) {
      e.preventDefault()
      const t = e.touches[0]
      const rect = this.canvas!.getBoundingClientRect()
      this._touchStartPos = { x: t.clientX - rect.left, y: t.clientY - rect.top }
      this._touchStartTime = performance.now()
      this.isDragging = true
      this.lastMouse = { x: t.clientX, y: t.clientY }
    } else if (e.touches.length === 2) {
      e.preventDefault()
      this.isDragging = false
      const dx = e.touches[0].clientX - e.touches[1].clientX
      const dy = e.touches[0].clientY - e.touches[1].clientY
      this._pinchDist = Math.sqrt(dx * dx + dy * dy)
      this._pinchZoom = this.zoom
    }
  }

  private _onTouchMove = (e: TouchEvent) => {
    if (e.touches.length === 1 && this.isDragging) {
      e.preventDefault()
      const t = e.touches[0]
      const dx = t.clientX - this.lastMouse.x
      const dy = t.clientY - this.lastMouse.y
      this.panX += dx
      this.panY += dy
      this._panTargetX = this.panX
      this._panTargetY = this.panY
      this.lastMouse = { x: t.clientX, y: t.clientY }
    } else if (e.touches.length === 2 && this._pinchDist) {
      e.preventDefault()
      const dx = e.touches[0].clientX - e.touches[1].clientX
      const dy = e.touches[0].clientY - e.touches[1].clientY
      const dist = Math.sqrt(dx * dx + dy * dy)
      const factor = dist / this._pinchDist
      const newZoom = Math.max(this.options.zoomMin, Math.min(this.options.zoomMax, this._pinchZoom * factor))
      this._animateZoom(newZoom, this.width / 2, this.height / 2)
    }
  }

  private _onTouchEnd = (e: TouchEvent) => {
    e.preventDefault()
    if (e.changedTouches.length > 0 && this._touchStartPos) {
      const t = e.changedTouches[0]
      const rect = this.canvas!.getBoundingClientRect()
      const ex = t.clientX - rect.left
      const ey = t.clientY - rect.top

      const distMoved = Math.sqrt((ex - this._touchStartPos.x) ** 2 + (ey - this._touchStartPos.y) ** 2)
      const elapsed = performance.now() - this._touchStartTime

      if (distMoved < this._TAP_MAX_DIST && elapsed < this._TAP_MAX_MS) {
        const world = this.screenToWorld(ex, ey)
        const node = this.findNodeAt(world.x, world.y, 12)
        if (node) {
          this.hoveredNode = node
          this.selectedNode = node
          node.targetRadius = node.radius * 1.2
          const tapResetId = setTimeout(() => {
            if (this.selectedNode === node) node.targetRadius = node.radius
          }, 220)
          this._tapResetTimers.push(tapResetId)
          if (this.onNodeSelect) this.onNodeSelect(node.term)
          if (this.onHoverChange) this.onHoverChange(node, e)
        } else {
          this.selectedNode = null
          this.hoveredNode = null
          if (this.onHoverChange) this.onHoverChange(null, null)
        }
      }
    }

    this.isDragging = false
    this._pinchDist = null
    this._touchStartPos = null
    this.canvas!.style.cursor = 'grab'
  }

  private _bindEvents() {
    if (!this.canvas) return
    const add = (target: EventTarget, type: string, fn: EventListener, opts?: AddEventListenerOptions) => {
      target.addEventListener(type, fn, opts)
      this._boundListeners.push({ target, type, fn, opts })
    }

    add(window, 'resize', () => {
      if (this._resizeTimer) clearTimeout(this._resizeTimer)
      this._resizeTimer = setTimeout(() => this._handleResize(), 200)
    })
    add(this.canvas, 'mousemove', this._onMouseMove as EventListener)
    add(this.canvas, 'mousedown', this._onMouseDown as EventListener)
    add(this.canvas, 'mouseup', this._onMouseUp as EventListener)
    add(this.canvas, 'mouseleave', this._onMouseLeave as EventListener)
    add(this.canvas, 'wheel', this._onWheel as EventListener, { passive: false })
    add(this.canvas, 'click', this._onClick as EventListener)
    add(this.canvas, 'touchstart', this._onTouchStart as EventListener, { passive: false })
    add(this.canvas, 'touchmove', this._onTouchMove as EventListener, { passive: false })
    add(this.canvas, 'touchend', this._onTouchEnd as EventListener)

    add(document, 'visibilitychange', () => {
      if (document.hidden) {
        this.stopAnimation()
      } else {
        this._lastTs = null
        this._startAnimation()
      }
    })
  }

  private _handleResize() {
    if (!this.canvas) return
    const parent = this.canvas.parentElement
    if (!parent) return
    const rect = parent.getBoundingClientRect()

    this.width = rect.width || 800
    this.height = rect.height || 600
    this.centerX = this.width / 2
    this.centerY = this.height / 2
    this.dpr = window.devicePixelRatio || 1

    this.canvas.width = Math.round(this.width * this.dpr)
    this.canvas.height = Math.round(this.height * this.dpr)
    this.canvas.style.width = `${this.width}px`
    this.canvas.style.height = `${this.height}px`

    this._gridDirty = true
    this._isDirty = true
  }

  private _animateZoom(newZoom: number, pivotX: number, pivotY: number) {
    const scale = newZoom / this.zoom
    this._panTargetX = pivotX - (pivotX - this.panX) * scale
    this._panTargetY = pivotY - (pivotY - this.panY) * scale
    this._zoomTarget = newZoom
  }

  private _startAnimation() {
    if (this._isRunning) return

    const tick = (ts: number) => {
      const dt = this._lastTs ? Math.min((ts - this._lastTs) / 1000, 0.05) : 0.016
      this._lastTs = ts

      this.time += dt

      const LERP = 0.18
      const zooming = Math.abs(this._zoomTarget - this.zoom) > 0.0005
      const panning = Math.abs(this._panTargetX - this.panX) > 0.1 || Math.abs(this._panTargetY - this.panY) > 0.1
      if (zooming || panning) {
        this.zoom += (this._zoomTarget - this.zoom) * LERP
        this.panX += (this._panTargetX - this.panX) * LERP
        this.panY += (this._panTargetY - this.panY) * LERP
        this._isDirty = true
      }

      const hadMovement = this._simulatePhysics(dt * 60)
      if (hadMovement) this._isDirty = true

      // Skip render when idle (nothing changed for several frames)
      const shouldRender = this._isDirty || this.hoveredNode || this.selectedNode || this._idleFrames < 3
      if (shouldRender) {
        this._render()
        this._idleFrames = 0
      } else {
        this._idleFrames++
      }

      this._isDirty = false

      // Full stop when truly idle — saves CPU/battery
      if (this._idleFrames >= this._MAX_IDLE) {
        this._isRunning = false
        this.animationId = null
        return
      }

      this.animationId = requestAnimationFrame(tick)
    }

    this._isRunning = true
    this.animationId = requestAnimationFrame(tick)
  }

  private _rebuildGridCache() {
    const w = this.width
    const h = this.height
    const cw = Math.round(w * this.dpr)
    const ch = Math.round(h * this.dpr)

    // Reuse existing canvas if dimensions haven't changed
    if (!this._gridCanvas) {
      this._gridCanvas = document.createElement('canvas')
      this._gridCanvas.width = cw
      this._gridCanvas.height = ch
    } else if (this._gridCanvas.width !== cw || this._gridCanvas.height !== ch) {
      this._gridCanvas.width = cw
      this._gridCanvas.height = ch
    }
    // else: dimensions match, just repaint

    const gc = this._gridCanvas.getContext('2d')!
    gc.setTransform(this.dpr, 0, 0, this.dpr, 0, 0)
    gc.fillStyle = this.theme.gridDot

    const gap = 40
    for (let x = gap; x < w; x += gap) {
      for (let y = gap; y < h; y += gap) {
        gc.beginPath()
        gc.arc(x, y, 1.5, 0, Math.PI * 2)
        gc.fill()
      }
    }

    this._gridDirty = false
  }

  private _render() {
    const ctx = this.ctx
    if (!ctx || !this.canvas) return

    // Track selection changes for sort optimization
    const curSelId = this.selectedNode?.id ?? null
    if (curSelId !== this._prevSelectedId) {
      this._sortedDirty = true
      this._prevSelectedId = curSelId
    }

    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0)

    const bgGrad = ctx.createRadialGradient(this.centerX, this.centerY, 0, this.centerX, this.centerY, Math.max(this.width, this.height) * 0.7)
    bgGrad.addColorStop(0, this.theme.bgGradientCenter)
    bgGrad.addColorStop(1, this.theme.bg)
    ctx.fillStyle = bgGrad
    ctx.fillRect(0, 0, this.width, this.height)

    if (this._gridDirty || !this._gridCanvas) this._rebuildGridCache()
    if (this._gridCanvas) ctx.drawImage(this._gridCanvas, 0, 0, this.width, this.height)

    ctx.save()
    ctx.translate(this.panX, this.panY)
    ctx.scale(this.zoom, this.zoom)

    this._drawEdges(ctx)
    this._drawNodes(ctx)

    ctx.restore()
  }

  private _drawEdges(ctx: CanvasRenderingContext2D) {
    const selected = this.selectedNode
    const hovered = this.hoveredNode

    this.edges.forEach((edge) => {
      const src = this.nodeMap.get(edge.source)
      const tgt = this.nodeMap.get(edge.target)
      if (!src || !tgt || !src.visible || !tgt.visible) return

      const srcActive = selected && (selected.id === src.id || selected.id === tgt.id)
      const hovActive = hovered && (hovered.id === src.id || hovered.id === tgt.id)

      let opacity: number
      if (selected) {
        opacity = srcActive ? 1.0 : 0.06
      } else if (hovered) {
        opacity = hovActive ? 0.85 : 0.12
      } else {
        opacity = 0.55
      }

      const dx = tgt.x - src.x
      const dy = tgt.y - src.y
      const dist = Math.max(Math.sqrt(dx * dx + dy * dy), 1)

      const mx = (src.x + tgt.x) / 2
      const my = (src.y + tgt.y) / 2
      // Edge curve direction: since edges are bidirectional (no inherent direction),
      // we use a deterministic but arbitrary ID-based comparison so that the curve bends
      // consistently for each pair rather than randomly flipping.
      const off = dist * 0.15
      const dir = src.id < tgt.id ? 1 : -1
      const cx = mx + (dy / dist) * off * dir
      const cy = my - (dx / dist) * off * dir

      const grad = ctx.createLinearGradient(src.x, src.y, tgt.x, tgt.y)
      grad.addColorStop(0, this._hexToRgba(src.color, opacity))
      grad.addColorStop(1, this._hexToRgba(tgt.color, opacity))

      ctx.beginPath()
      ctx.moveTo(src.x, src.y)
      ctx.quadraticCurveTo(cx, cy, tgt.x, tgt.y)
      ctx.strokeStyle = grad
      ctx.lineWidth = srcActive || hovActive ? 2.5 : 1.5
      ctx.lineCap = 'round'
      ctx.setLineDash([])
      ctx.stroke()

      // Arrowhead
      const tangX = tgt.x - cx
      const tangY = tgt.y - cy
      const angle = Math.atan2(tangY, tangX)
      const arrowX = tgt.x - Math.cos(angle) * (tgt.radius + 6)
      const arrowY = tgt.y - Math.sin(angle) * (tgt.radius + 6)

      ctx.save()
      ctx.translate(arrowX, arrowY)
      ctx.rotate(angle)
      ctx.beginPath()
      ctx.moveTo(0, 0)
      ctx.lineTo(-8, 4)
      ctx.lineTo(-8, -4)
      ctx.closePath()
      ctx.fillStyle = this._hexToRgba(tgt.color, opacity)
      ctx.fill()
      ctx.restore()

      // Animated packets on active edges — crisp dots, no glow
      if (srcActive || hovActive) {
        ctx.save()

        const packetColor = this._isDark ? '#22d3ee' : '#06b6d4'
        for (let i = 0; i < 3; i++) {
          const t = ((this.time * 0.6 + i / 3)) % 1
          const pos = this._bezierPoint(t, src.x, src.y, cx, cy, tgt.x, tgt.y)
          const sz = 2 + Math.sin(this.time * 10 + i) * 0.5

          ctx.beginPath()
          ctx.arc(pos.x, pos.y, sz, 0, Math.PI * 2)
          ctx.fillStyle = packetColor
          ctx.fill()
        }

        ctx.restore()
        ctx.shadowBlur = 0
      }
    })
  }

  private _drawNodes(ctx: CanvasRenderingContext2D) {
    // Re-sort when selection changes — separate sorting from drawing to avoid mutation
    if (this._sortedDirty) {
      this._sortedNodes = this.nodes.slice()
      // Move selected node to end for draw order (on top)
      const selId = this.selectedNode?.id
      if (selId) {
        const selIdx = this._sortedNodes.findIndex((n) => n.id === selId)
        if (selIdx !== -1 && selIdx !== this._sortedNodes.length - 1) {
          const [selNode] = this._sortedNodes.splice(selIdx, 1)
          this._sortedNodes.push(selNode)
        }
      }
      this._sortedDirty = false
    }

    const nodes = this._sortedNodes
    const selId = this.selectedNode?.id
    const hovId = this.hoveredNode?.id
    const hasSelection = !!selId
    const fontSize = this.options.fontSize
    const fontFamily = "'JetBrains Mono', ui-monospace, monospace"
    const time = this.time
    const isDark = this._isDark

    for (let ni = 0; ni < nodes.length; ni++) {
      const node = nodes[ni]
      if (!node.visible) continue

      const isSel = selId === node.id
      const isHov = hovId === node.id
      const dimmed = hasSelection && !isSel && !node.highlighted

      let r = Math.max(1, node.currentRadius + (isSel ? Math.sin(time * 4) * 1.5 : 0))

      // Outer scanning rings (selected)
      if (isSel) {
        ctx.save()
        ctx.translate(node.x, node.y)
        ctx.globalAlpha = 0.6
        ctx.strokeStyle = node.color
        ctx.lineWidth = 1.5
        ctx.setLineDash([4, 8])
        ctx.rotate(-time)
        ctx.beginPath()
        ctx.arc(0, 0, r + 8, 0, Math.PI * 2)
        ctx.stroke()

        ctx.setLineDash([])
        ctx.lineWidth = 1
        ctx.globalAlpha = 0.3 + Math.sin(time * 5) * 0.2
        ctx.rotate(time * 2)
        ctx.beginPath()
        ctx.arc(0, 0, r + 4 + Math.sin(time * 3) * 3, 0, Math.PI * 2)
        ctx.stroke()
        ctx.restore()
      }

      // Highlighted ring (search result)
      if (node.highlighted && !isSel) {
        ctx.save()
        ctx.globalAlpha = 0.7 + Math.sin(time * 6) * 0.2
        ctx.strokeStyle = node.color
        ctx.lineWidth = 2
        ctx.setLineDash([3, 5])
        ctx.beginPath()
        ctx.arc(node.x, node.y, r + 5, 0, Math.PI * 2)
        ctx.stroke()
        ctx.setLineDash([])
        ctx.restore()
      }

      // Shadow — tight, precise, no diffuse glow
      ctx.shadowColor = isSel || isHov ? node.color : 'rgba(0,0,0,0.06)'
      ctx.shadowBlur = isSel || isHov ? 4 : 2

      // Node body
      ctx.beginPath()
      ctx.arc(node.x, node.y, r, 0, Math.PI * 2)

      const innerR = Math.max(r * 0.1, 0.5)
      const grad = ctx.createRadialGradient(node.x - r * 0.3, node.y - r * 0.3, innerR, node.x, node.y, Math.max(r, 1))

      if (isSel || isHov || node.highlighted) {
        grad.addColorStop(0, isDark ? '#334155' : '#ffffff')
        grad.addColorStop(0.4, this._hexToRgba(node.color, dimmed ? 0.3 : 1))
        grad.addColorStop(1, this._darkenColor(node.color, 20))
      } else {
        grad.addColorStop(0, dimmed ? this.theme.dimmedBodyStart : this.theme.nodeBodyStart)
        grad.addColorStop(1, dimmed ? this.theme.dimmedBodyEnd : this.theme.nodeBodyEnd)
      }

      ctx.globalAlpha = dimmed ? 0.35 : 1
      ctx.fillStyle = grad
      ctx.fill()
      ctx.globalAlpha = 1
      ctx.shadowBlur = 0

      // Border
      ctx.strokeStyle = isSel || isHov ? node.color : this.theme.nodeBorderDefault
      ctx.lineWidth = isSel || isHov ? 2.5 : 1
      ctx.stroke()

      // Label (with cache)
      const maxLabelPx = Math.round((r - 3) * 2)
      const cacheKey = `${node.id}:${maxLabelPx}`
      let label = this._labelCache.get(cacheKey)
      if (label === undefined) {
        ctx.font = `600 ${fontSize}px ${fontFamily}`
        label = this._truncateLabel(ctx, node.term.name, maxLabelPx)
        this._labelCache.set(cacheKey, label)
      }
      ctx.font = `600 ${fontSize}px ${fontFamily}`
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'
      ctx.fillStyle = isSel ? (isDark ? '#f4f4f5' : '#111827') : isHov ? node.color : this.theme.nodeLabelDefault

      ctx.shadowColor = 'rgba(0,0,0,0.6)'
      ctx.shadowBlur = 2
      ctx.fillText(label, node.x, node.y)
      ctx.shadowBlur = 0
    }
  }

  private _prevSelectedId: string | null = null

  private _truncateLabel(ctx: CanvasRenderingContext2D, text: string, maxPx: number): string {
    if (ctx.measureText(text).width <= maxPx) return text
    let t = text
    while (t.length > 1 && ctx.measureText(t + '\u2026').width > maxPx) {
      t = t.slice(0, -1)
    }
    return t + '\u2026'
  }

  private _bezierPoint(t: number, sx: number, sy: number, cx: number, cy: number, tx: number, ty: number) {
    const u = 1 - t
    return {
      x: u * u * sx + 2 * u * t * cx + t * t * tx,
      y: u * u * sy + 2 * u * t * cy + t * t * ty,
    }
  }

  private _hexToRgba(hex: string, alpha = 1): string {
    if (typeof hex !== 'string' || !hex.startsWith('#')) return `rgba(148,163,184,${alpha})`

    let h = hex.slice(1)
    if (h.length === 3) h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2]
    if (h.length !== 6) return `rgba(148,163,184,${alpha})`

    const r = parseInt(h.slice(0, 2), 16)
    const g = parseInt(h.slice(2, 4), 16)
    const b = parseInt(h.slice(4, 6), 16)

    if (isNaN(r) || isNaN(g) || isNaN(b)) return `rgba(148,163,184,${alpha})`
    return `rgba(${r},${g},${b},${alpha})`
  }

  private _darkenColor(hex: string, percent: number): string {
    if (typeof hex !== 'string' || !hex.startsWith('#')) return '#94a3b8'

    let h = hex.slice(1)
    if (h.length === 3) h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2]
    if (h.length !== 6) return '#94a3b8'

    const amt = Math.round(2.55 * percent)
    const num = parseInt(h, 16)
    const R = Math.max(0, (num >> 16) - amt)
    const G = Math.max(0, ((num >> 8) & 0x00ff) - amt)
    const B = Math.max(0, (num & 0xff) - amt)
    return '#' + (0x1000000 + R * 0x10000 + G * 0x100 + B).toString(16).slice(1)
  }
}