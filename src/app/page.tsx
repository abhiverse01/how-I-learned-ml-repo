'use client'

import { useEffect, useRef, useState, useCallback, useMemo } from 'react'
import { useTitanMLStore, type Term, type Category, type KnowledgePath } from '@/store/useTitanMLStore'
import { KnowledgeGraph } from '@/lib/graph-engine'

// ─── Utility functions ───
function escapeHTML(text: string): string {
  return String(text ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')
}

function parseMarkdown(text: string): string {
  if (!text) return ''
  let out = escapeHTML(text)
  out = out.replace(/```([\s\S]*?)```/g, (_, code) => `<pre><code>${code.trim()}</code></pre>`)
  out = out.replace(/`([^`\n]+)`/g, '<code>$1</code>')
  out = out.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
  out = out.replace(/(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)/g, '<em>$1</em>')
  out = out.replace(/^### (.+)$/gm, '<h4>$1</h4>')
  out = out.replace(/^## (.+)$/gm, '<h3>$1</h3>')
  out = out.replace(/^# (.+)$/gm, '<h2>$1</h2>')
  out = out.replace(
    /(?:^- .+\n?)+/gm,
    (block) =>
      '<ul>' +
      block
        .trim()
        .split('\n')
        .map((l) => `<li>${l.replace(/^- /, '').trim()}</li>`)
        .join('') +
      '</ul>'
  )
  out = out.replace(
    /(?:^\d+\. .+\n?)+/gm,
    (block) =>
      '<ol>' +
      block
        .trim()
        .split('\n')
        .map((l) => `<li>${l.replace(/^\d+\. /, '').trim()}</li>`)
        .join('') +
      '</ol>'
  )
  const BLOCK_START = /^<(ul|ol|pre|h[1-6]|blockquote)/
  out = out
    .split(/\n{2,}/)
    .map((block) => {
      const trimmed = block.trim()
      if (!trimmed) return ''
      if (BLOCK_START.test(trimmed)) return trimmed
      return `<p>${trimmed.replace(/\n/g, '<br>')}</p>`
    })
    .filter(Boolean)
    .join('')
  return out
}

// ─── Architecture step renderer ───
function renderArchSteps(steps: { label: string; desc?: string; type?: string; inputId?: string; children?: typeof steps }[]): string {
  if (!steps || !steps.length) return ''
  return steps
    .map((step) => {
      let typeClass = `step-type-${step.type || 'process'}`
      let childrenHtml = ''
      if (step.children && step.children.length > 0) {
        typeClass += ' step-type-container'
        childrenHtml = `<div class="step-children">${renderArchSteps(step.children)}</div>`
      }
      return `
        <div class="flow-step">
          <div class="step-box ${typeClass}">
            <div class="step-label">${escapeHTML(step.label)}</div>
            ${step.desc ? `<div class="step-desc">${escapeHTML(step.desc)}</div>` : ''}
          </div>
          ${childrenHtml}
        </div>
      `
    })
    .join('')
}

// ─── Loading Screen Component ───
const EMOJIS = ['\uD83E\uDDE0', '\uD83E\uDD16', '\uD83D\uDCA1', '\u26A1']

function LoadingScreen() {
  const [emojiIndex, setEmojiIndex] = useState(0)

  useEffect(() => {
    const id = setInterval(() => setEmojiIndex((i) => (i + 1) % EMOJIS.length), 800)
    return () => clearInterval(id)
  }, [])

  return (
    <div id="initial-loader">
      <div className="loader-spinner" />
      <p className="loader-text">
        Initializing Knowledge Base... {EMOJIS[emojiIndex]}
      </p>
    </div>
  )
}

// ─── Toast Component ───
interface ToastData {
  message: string
  type: string
  key: number
  exiting: boolean
}

function Toast({ toast }: { toast: ToastData }) {
  const bg =
    toast.type === 'success'
      ? 'var(--accent-success, #10b981)'
      : toast.type === 'error'
        ? 'var(--accent-danger, #ef4444)'
        : 'var(--accent-primary, #0891b2)'

  return (
    <div
      key={toast.key}
      className={`app-toast ${toast.exiting ? 'toast-exit' : ''}`}
      style={{ background: bg }}
      role="status"
      aria-live="polite"
    >
      {toast.message}
    </div>
  )
}

// ─── Main Page Component ───
export default function Home() {
  // Select only the state slices we need to minimize re-renders
  const isLoaded = useTitanMLStore((s) => s.isLoaded)
  const isDarkMode = useTitanMLStore((s) => s.isDarkMode)
  const activeView = useTitanMLStore((s) => s.activeView)
  const detailPanelOpen = useTitanMLStore((s) => s.detailPanelOpen)
  const addModalOpen = useTitanMLStore((s) => s.addModalOpen)
  const selectedTerm = useTitanMLStore((s) => s.selectedTerm)
  const selectedCategory = useTitanMLStore((s) => s.selectedCategory)
  const selectedFilter = useTitanMLStore((s) => s.selectedFilter)
  const searchQuery = useTitanMLStore((s) => s.searchQuery)
  const sidebarOpen = useTitanMLStore((s) => s.sidebarOpen)
  const sidebarCollapsed = useTitanMLStore((s) => s.sidebarCollapsed)
  const categories = useTitanMLStore((s) => s.categories)
  const stats = useTitanMLStore((s) => s.stats)
  const archCurrentView = useTitanMLStore((s) => s.archCurrentView)
  const archCurrentId = useTitanMLStore((s) => s.archCurrentId)
  const pathMode = useTitanMLStore((s) => s.pathMode)
  const activePath = useTitanMLStore((s) => s.activePath)
  const pathSubPaths = useTitanMLStore((s) => s.pathSubPaths)
  const knowledgePaths = useTitanMLStore((s) => s.knowledgePaths)
  const architectures = useTitanMLStore((s) => s.architectures)
  const terms = useTitanMLStore((s) => s.terms)

  // Actions
  const setCategories = useTitanMLStore((s) => s.setCategories)
  const setTerms = useTitanMLStore((s) => s.setTerms)
  const setDataLoaded = useTitanMLStore((s) => s.setDataLoaded)
  const setArchitectures = useTitanMLStore((s) => s.setArchitectures)
  const setKnowledgePaths = useTitanMLStore((s) => s.setKnowledgePaths)
  const setStats = useTitanMLStore((s) => s.setStats)
  const addTerm = useTitanMLStore((s) => s.addTerm)
  const setActiveView = useTitanMLStore((s) => s.setActiveView)
  const setSearchQuery = useTitanMLStore((s) => s.setSearchQuery)
  const setSelectedCategory = useTitanMLStore((s) => s.setSelectedCategory)
  const setSelectedTerm = useTitanMLStore((s) => s.setSelectedTerm)
  const setSelectedFilter = useTitanMLStore((s) => s.setSelectedFilter)
  const toggleSidebar = useTitanMLStore((s) => s.toggleSidebar)
  const closeSidebar = useTitanMLStore((s) => s.closeSidebar)
  const setDetailPanelOpen = useTitanMLStore((s) => s.setDetailPanelOpen)
  const setAddModalOpen = useTitanMLStore((s) => s.setAddModalOpen)
  const toggleDarkMode = useTitanMLStore((s) => s.toggleDarkMode)
  const navigateTerm = useTitanMLStore((s) => s.navigateTerm)
  const setArchCurrentView = useTitanMLStore((s) => s.setArchCurrentView)
  const setArchCurrentId = useTitanMLStore((s) => s.setArchCurrentId)
  const setPathMode = useTitanMLStore((s) => s.setPathMode)
  const setActivePath = useTitanMLStore((s) => s.setActivePath)
  const setPathSubPaths = useTitanMLStore((s) => s.setPathSubPaths)

  const canvasRef = useRef<HTMLCanvasElement>(null)
  const graphRef = useRef<KnowledgeGraph | null>(null)
  const tooltipRef = useRef<HTMLDivElement>(null)
  // codeBlockRef removed — copy button is now inline React state
  const searchInputRef = useRef<HTMLInputElement>(null)
  const toastTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const toastExitTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const resizeObserverRef = useRef<ResizeObserver | null>(null)
  const [toast, setToast] = useState<ToastData | null>(null)
  const [loaderFading, setLoaderFading] = useState(false)
  const [isMobile, setIsMobile] = useState(false)
  const toastKeyRef = useRef(0)

  // ─── Detect mobile (client-only, no hydration issue) ───
  useEffect(() => {
    const check = () => setIsMobile(window.innerWidth < 768)
    check()
    window.addEventListener('resize', check)
    return () => window.removeEventListener('resize', check)
  }, [])

  // ─── Toast ───
  const showToast = useCallback((message: string, type = 'info') => {
    if (toastTimerRef.current) clearTimeout(toastTimerRef.current)
    if (toastExitTimerRef.current) clearTimeout(toastExitTimerRef.current)
    const key = ++toastKeyRef.current
    setToast({ message, type, key, exiting: false })
    toastTimerRef.current = setTimeout(() => {
      setToast((prev) => (prev ? { ...prev, exiting: true } : null))
      toastExitTimerRef.current = setTimeout(() => setToast(null), 300)
    }, 2700)
  }, [])

  // ─── Cleanup timers on unmount ───
  useEffect(() => {
    return () => {
      if (toastTimerRef.current) clearTimeout(toastTimerRef.current)
      if (toastExitTimerRef.current) clearTimeout(toastExitTimerRef.current)
      if (searchTimerRef.current) clearTimeout(searchTimerRef.current)
    }
  }, [])

  // ─── Data Loading ───
  useEffect(() => {
    let cancelled = false

    async function loadData() {
      try {
        const [graphRes, archRes, pathRes] = await Promise.all([
          fetch('/data/graphData.json'),
          fetch('/data/architecture.json'),
          fetch('/data/knowledgePath.json'),
        ])

        const graphData = await graphRes.json()
        const archData = await archRes.json()
        const pathData = await pathRes.json()

        if (cancelled) return

        const cats: Category[] = graphData.categories || []
        const ts: Term[] = (graphData.terms || []).map((t: Record<string, unknown>) => ({
          id: t.id || '',
          name: t.name || '',
          fullName: (t.fullName as string) || (t.name as string) || '',
          category: (t.category as string) || 'general',
          type: (t.type as string) || 'technique',
          shortDesc: (t.shortDesc as string) || '',
          definition: (t.definition as string) || '',
          related: (t.related as string[]) || [],
          tags: (t.tags as string[]) || [],
          codeExample: (t.codeExample as string) || '',
          createdAt: (t.createdAt as string) || new Date().toISOString(),
          importance: (t.importance as number) || 0,
        }))

        setCategories(cats)
        setTerms(ts)
        setDataLoaded()
        setArchitectures(archData)
        setKnowledgePaths(pathData.paths || pathData)

        // Compute stats — deduplicate connections (each edge counted once)
        const byCategory: Record<string, number> = {}
        for (let i = 0; i < cats.length; i++) {
          let count = 0
          for (let j = 0; j < ts.length; j++) {
            if (ts[j].category === cats[i].id) count++
          }
          byCategory[cats[i].id] = count
        }
        const edgeSet = new Set<string>()
        let connections = 0
        for (let i = 0; i < ts.length; i++) {
          const related = ts[i].related
          if (!related) continue
          for (let j = 0; j < related.length; j++) {
            const key = [ts[i].id, related[j]].sort().join('|')
            if (!edgeSet.has(key)) {
              edgeSet.add(key)
              connections++
            }
          }
        }
        setStats({ categories: cats.length, terms: ts.length, connections, byCategory })
      } catch (err) {
        console.error('Failed to load data:', err)
        if (!cancelled) {
          setCategories([])
          setTerms([])
          setDataLoaded()
        }
      }
    }

    loadData()
    return () => { cancelled = true }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // ─── Loading screen fade-out trigger ───
  useEffect(() => {
    if (!isLoaded) return
    // Small delay to let the fade-in animation play, then trigger fade-out
    const t = setTimeout(() => {
      setLoaderFading(true)
    }, 400)
    return () => clearTimeout(t)
  }, [isLoaded])

  // ─── Graph initialization ───
  useEffect(() => {
    if (!isLoaded || !canvasRef.current || graphRef.current) return

    const graph = new KnowledgeGraph()
    graph.init(canvasRef.current)
    graphRef.current = graph

    const cats = categories.map((c) => ({ id: c.id, name: c.name, color: c.color }))
    const termsData = terms.map((t) => ({
      id: t.id, name: t.name, fullName: t.fullName,
      category: t.category, type: t.type, shortDesc: t.shortDesc,
      tags: t.tags, related: t.related,
    }))

    graph.loadData(cats, termsData)

    // Set initial dark mode on graph
    if (isDarkMode) graph.setDarkMode(true)

    graph.onNodeSelect = (term) => {
      const fullTerm = useTitanMLStore.getState().terms.find((t) => t.id === term.id)
      if (fullTerm) {
        setSelectedTerm(fullTerm)
        navigateTerm(fullTerm.id)
      }
    }

    graph.onHoverChange = (node, e) => {
      const tooltip = tooltipRef.current
      if (!tooltip) return
      if (node?.term) {
        tooltip.innerHTML = `<strong>${escapeHTML(node.term.name)}</strong><br><span style="color:var(--text-muted);">${escapeHTML(node.term.shortDesc ?? '')}</span>`
        const tipW = 220
        let clientX = 0
        let clientY = 0
        if (e && 'clientX' in e) {
          clientX = (e as MouseEvent).clientX
          clientY = (e as MouseEvent).clientY
        } else if (e && 'changedTouches' in e) {
          const touch = (e as TouchEvent).changedTouches[0]
          if (touch) { clientX = touch.clientX; clientY = touch.clientY }
        }
        const x = Math.min(clientX + 12, window.innerWidth - tipW - 8)
        const y = clientY + 12
        tooltip.style.left = `${x}px`
        tooltip.style.top = `${y}px`
        tooltip.classList.add('visible')
      } else {
        tooltip.classList.remove('visible')
      }
    }

    // Resize observer
    if (canvasRef.current.parentElement && window.ResizeObserver) {
      let resizeTimer: ReturnType<typeof setTimeout>
      resizeObserverRef.current = new ResizeObserver(() => {
        clearTimeout(resizeTimer)
        resizeTimer = setTimeout(() => graph.resize(), 200)
      })
      resizeObserverRef.current.observe(canvasRef.current.parentElement)
    }

    return () => {
      graph.destroy()
      graphRef.current = null
      if (resizeObserverRef.current) {
        resizeObserverRef.current.disconnect()
        resizeObserverRef.current = null
      }
    }
  }, [isLoaded]) // eslint-disable-line react-hooks/exhaustive-deps

  // ─── Sync graph dark mode ───
  useEffect(() => {
    graphRef.current?.setDarkMode(isDarkMode)
  }, [isDarkMode])

  // ─── Sync selectedNode with store ───
  useEffect(() => {
    if (!graphRef.current || !selectedTerm) return
    graphRef.current.selectedNode = graphRef.current.findNode(selectedTerm.id)
  }, [selectedTerm])

  // ─── Search ───
  const handleSearch = useCallback(
    (value: string) => {
      setSearchQuery(value)
      graphRef.current?.highlightNodes(value || null)
    },
    [setSearchQuery]
  )

  const searchTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const onSearchInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const value = e.target.value
      if (searchTimerRef.current) clearTimeout(searchTimerRef.current)
      searchTimerRef.current = setTimeout(() => handleSearch(value), 150)
    },
    [handleSearch]
  )

  // ─── Keyboard shortcuts ───
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const active = document.activeElement
      const isTyping = active?.matches('input,textarea,[contenteditable]')

      if (e.key === '/' && !isTyping) {
        e.preventDefault()
        searchInputRef.current?.focus()
      }
      if (e.key === 'Escape') {
        setDetailPanelOpen(false)
        setAddModalOpen(false)
        searchInputRef.current?.blur()
      }
      if (e.key === 'ArrowLeft' && e.altKey) {
        e.preventDefault()
        useTitanMLStore.getState().goBack()
      }
      if (e.key === 'ArrowRight' && e.altKey) {
        e.preventDefault()
        useTitanMLStore.getState().goForward()
      }
    }
    document.addEventListener('keydown', handler)
    return () => document.removeEventListener('keydown', handler)
  }, [setDetailPanelOpen, setAddModalOpen])

  // ─── Copy code state ───
  const [copyState, setCopyState] = useState<'idle' | 'copied' | 'failed'>('idle')

  const handleCopyCode = useCallback(async () => {
    const code = selectedTerm?.codeExample
    if (!code) return
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(code)
      } else {
        const ta = document.createElement('textarea')
        ta.value = code
        ta.style.cssText = 'position:fixed;top:-999px;left:-999px;opacity:0;'
        document.body.appendChild(ta)
        ta.select()
        document.execCommand('copy')
        ta.remove()
      }
      setCopyState('copied')
      setTimeout(() => setCopyState('idle'), 1500)
    } catch {
      setCopyState('failed')
      setTimeout(() => setCopyState('idle'), 1500)
    }
  }, [selectedTerm?.codeExample])

  // Reset copy state when term changes
  useEffect(() => {
    setCopyState('idle')
  }, [selectedTerm])

  // ─── Handlers ───
  const handleCategoryClick = useCallback((categoryId: string) => {
    if (selectedCategory === categoryId) {
      setSelectedCategory(null)
      graphRef.current?.filterByCategory(null)
    } else {
      setSelectedCategory(categoryId)
      graphRef.current?.filterByCategory(categoryId)
    }
  }, [selectedCategory, setSelectedCategory])

  const handleFilterClick = useCallback((filter: string) => {
    setSelectedFilter(filter)
    graphRef.current?.filterByType(filter === 'all' ? null : filter)
    if (filter === 'all') setSelectedCategory(null)
  }, [setSelectedFilter, setSelectedCategory])

  const handleRelatedClick = useCallback((termId: string) => {
    const term = terms.find((t) => t.id === termId)
    if (term) {
      setSelectedTerm(term)
      navigateTerm(term.id)
      if (graphRef.current) {
        graphRef.current.selectedNode = graphRef.current.findNode(term.id)
      }
    }
  }, [terms, setSelectedTerm, navigateTerm])

  const handleAddTerm = useCallback((e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    const formData = new FormData(e.currentTarget)
    const name = formData.get('name')?.toString().trim() ?? ''
    const categoryId = formData.get('category')?.toString().trim() ?? ''
    const shortDesc = formData.get('shortDesc')?.toString().trim() ?? ''
    const definition = formData.get('definition')?.toString().trim() ?? ''
    const relatedStr = formData.get('related')?.toString().trim() ?? ''
    const tagsStr = formData.get('tags')?.toString().trim() ?? ''

    if (!name || !categoryId || !shortDesc || !definition) {
      showToast('Please fill all required fields.', 'error')
      return
    }

    const baseId = name.toLowerCase().replace(/\s+/g, '-').replace(/[^a-z0-9-]/g, '')
    if (!baseId) {
      showToast('Could not generate a valid ID from that name.', 'error')
      return
    }

    let id = baseId
    let suffix = 1
    while (terms.find((t) => t.id === id)) {
      id = `${baseId}-${++suffix}`
    }

    const related = relatedStr
      .split(',')
      .map((s) => s.trim().toLowerCase().replace(/\s+/g, '-').replace(/[^a-z0-9-]/g, ''))
      .filter(Boolean)

    const tags = tagsStr
      .split(',')
      .map((s) => s.trim())
      .filter(Boolean)

    addTerm({
      id, name, fullName: name, category: categoryId, type: 'technique',
      shortDesc, definition, related, tags, codeExample: '',
      createdAt: new Date().toISOString(), importance: 0,
    })

    if (graphRef.current) {
      const cats = categories.map((c) => ({ id: c.id, name: c.name, color: c.color }))
      const updatedTerms = useTitanMLStore.getState().terms
      const termsData = updatedTerms.map((t) => ({
        id: t.id, name: t.name, fullName: t.fullName,
        category: t.category, type: t.type, shortDesc: t.shortDesc,
        tags: t.tags, related: t.related,
      }))
      graphRef.current.loadData(cats, termsData)
    }

    setAddModalOpen(false)
    showToast(`"${name}" added successfully!`, 'success')
  }, [addTerm, categories, terms, setAddModalOpen, showToast])

  // Architecture handlers
  const showArchGallery = useCallback(() => {
    setActiveView('architecture')
    setArchCurrentView('gallery')
    setArchCurrentId(null)
  }, [setActiveView, setArchCurrentView, setArchCurrentId])

  const showArchDetail = useCallback((archId: string) => {
    setArchCurrentId(archId)
    setArchCurrentView('architecture')
  }, [setArchCurrentId, setArchCurrentView])

  const hideArchView = useCallback(() => {
    setActiveView('graph')
    setArchCurrentId(null)
  }, [setActiveView, setArchCurrentId])

  // Knowledge path handlers
  const showPathsView = useCallback(() => {
    setActiveView('knowledge-paths')
    setPathMode('gallery')
    setActivePath(null)
    setPathSubPaths([])
  }, [setActiveView, setPathMode, setActivePath, setPathSubPaths])

  const startPath = useCallback((path: KnowledgePath) => {
    if (path.paths && !path.steps) {
      setPathMode('subpaths')
      setPathSubPaths(path.paths)
      setActivePath(path)
    } else {
      setPathMode('steps')
      setActivePath(path)
    }
  }, [setPathMode, setPathSubPaths, setActivePath])

  const focusTerm = useCallback((termName: string) => {
    setActiveView('graph')
    const term = terms.find((t) => t.name === termName || t.fullName === termName)
    if (term) {
      setSelectedTerm(term)
      navigateTerm(term.id)
    }
  }, [terms, setActiveView, setSelectedTerm, navigateTerm])

  // ─── Memoized derived data ───
  const selectedTermCategory = useMemo(() => {
    if (!selectedTerm) return null
    return categories.find((c) => c.id === selectedTerm.category) ?? null
  }, [selectedTerm, categories])

  const relatedTerms = useMemo(() => {
    if (!selectedTerm) return []
    const results: { term: Term; cat: Category | undefined }[] = []
    for (let i = 0; i < selectedTerm.related.length; i++) {
      const rt = terms.find((t) => t.id === selectedTerm.related[i])
      if (!rt) continue
      const rc = categories.find((c) => c.id === rt.category)
      results.push({ term: rt, cat: rc })
    }
    return results
  }, [selectedTerm, terms, categories])

  const currentArch = useMemo(() => {
    if (!archCurrentId) return null
    return architectures.find((a) => a.id === archCurrentId) ?? null
  }, [archCurrentId, architectures])

  // ─── Loading screen ───
  if (!isLoaded) {
    return <LoadingScreen />
  }

  // Show fade-out loader + app content simultaneously
  return (
    <>
      {loaderFading && (
        <div id="initial-loader" className="fade-out">
          <div className="loader-spinner" />
          <p className="loader-text">
            Initializing Knowledge Base...
          </p>
        </div>
      )}
      <AppContent
          // Header
          searchQuery={searchQuery} onSearchInput={onSearchInput} searchInputRef={searchInputRef}
          toggleSidebar={toggleSidebar} onAddTerm={() => setAddModalOpen(true)}
          sidebarOpen={sidebarOpen} sidebarCollapsed={sidebarCollapsed} closeSidebar={closeSidebar}
          categories={categories} selectedCategory={selectedCategory} stats={stats}
          selectedFilter={selectedFilter} isDarkMode={isDarkMode} toggleDarkMode={toggleDarkMode}
          handleCategoryClick={handleCategoryClick} handleFilterClick={handleFilterClick}
          showArchGallery={showArchGallery} showPathsView={showPathsView}
          // Graph
          canvasRef={canvasRef} tooltipRef={tooltipRef} graphRef={graphRef}
          // Architecture
          activeView={activeView} archCurrentView={archCurrentView} currentArch={currentArch}
          showArchDetail={showArchDetail} hideArchView={hideArchView}
          architectures={architectures} setArchCurrentView={setArchCurrentView}
          // Knowledge paths
          pathMode={pathMode} activePath={activePath}
          pathSubPaths={pathSubPaths} knowledgePaths={knowledgePaths}
          startPath={startPath} focusTerm={focusTerm}
          setPathMode={setPathMode} setActivePath={setActivePath} setPathSubPaths={setPathSubPaths}
          setActiveView={setActiveView}
          // Terms (for path step existence check)
          terms={terms}
          // Detail panel
          detailPanelOpen={detailPanelOpen} selectedTerm={selectedTerm}
          selectedTermCategory={selectedTermCategory} relatedTerms={relatedTerms}
          copyState={copyState}
          onCopyCode={handleCopyCode}
          onRelatedClick={handleRelatedClick} onClosePanel={() => setDetailPanelOpen(false)}
          // Modal
          addModalOpen={addModalOpen} categoriesForModal={categories}
          handleAddTerm={handleAddTerm} onCloseModal={() => setAddModalOpen(false)}
          // Toast
          toast={toast}
          // Meta
          isMobile={isMobile}
        />
      </>
    )
}

// ─── App Content (rendered after loading) ───
interface AppContentProps {
  // Header
  searchQuery: string
  onSearchInput: (e: React.ChangeEvent<HTMLInputElement>) => void
  searchInputRef: React.RefObject<HTMLInputElement | null>
  toggleSidebar: () => void
  onAddTerm: () => void
  // Sidebar
  sidebarOpen: boolean
  sidebarCollapsed: boolean
  closeSidebar: () => void
  categories: Category[]
  selectedCategory: string | null
  stats: { categories: number; terms: number; connections: number; byCategory: Record<string, number> } | null
  selectedFilter: string
  isDarkMode: boolean
  toggleDarkMode: () => void
  handleCategoryClick: (id: string) => void
  handleFilterClick: (filter: string) => void
  showArchGallery: () => void
  showPathsView: () => void
  // Graph
  canvasRef: React.RefObject<HTMLCanvasElement | null>
  tooltipRef: React.RefObject<HTMLDivElement | null>
  graphRef: React.RefObject<KnowledgeGraph | null>
  // Architecture
  activeView: string
  archCurrentView: string
  setArchCurrentView: (view: 'gallery' | 'architecture') => void
  currentArch: { id: string; name: string; category: string; shortDesc: string; steps: { label: string; desc?: string; type?: string; children?: unknown[] }[] } | null
  architectures: { id: string; name: string; category: string; shortDesc: string; steps: unknown[] }[]
  showArchDetail: (id: string) => void
  hideArchView: () => void
  // Knowledge paths
  pathMode: string
  activePath: KnowledgePath | null
  pathSubPaths: KnowledgePath[]
  knowledgePaths: KnowledgePath[]
  startPath: (path: KnowledgePath) => void
  focusTerm: (name: string) => void
  setPathMode: (mode: 'gallery' | 'steps' | 'subpaths') => void
  setActivePath: (path: KnowledgePath | null) => void
  setPathSubPaths: (paths: KnowledgePath[]) => void
  setActiveView: (view: 'graph' | 'architecture' | 'knowledge-paths') => void
  // Terms
  terms: Term[]
  // Detail panel
  detailPanelOpen: boolean
  selectedTerm: Term | null
  selectedTermCategory: Category | null
  relatedTerms: { term: Term; cat: Category | undefined }[]
  copyState: 'idle' | 'copied' | 'failed'
  onCopyCode: () => void
  onRelatedClick: (id: string) => void
  onClosePanel: () => void
  // Modal
  addModalOpen: boolean
  categoriesForModal: Category[]
  handleAddTerm: (e: React.FormEvent<HTMLFormElement>) => void
  onCloseModal: () => void
  // Toast
  toast: { message: string; type: string; key: number; exiting: boolean } | null
  // Meta
  isMobile: boolean
}

function AppContent({
  searchQuery, onSearchInput, searchInputRef, toggleSidebar, onAddTerm,
  sidebarOpen, sidebarCollapsed, closeSidebar, categories, selectedCategory, stats,
  selectedFilter, isDarkMode, toggleDarkMode, handleCategoryClick, handleFilterClick,
  showArchGallery, showPathsView,
  canvasRef, tooltipRef, graphRef,
  activeView, archCurrentView, setArchCurrentView, currentArch, architectures, showArchDetail, hideArchView,
  pathMode, activePath, pathSubPaths, knowledgePaths, startPath, focusTerm,
  setPathMode, setActivePath, setPathSubPaths, setActiveView,
  terms,
  detailPanelOpen, selectedTerm, selectedTermCategory, relatedTerms,
  copyState, onCopyCode,
  onRelatedClick, onClosePanel,
  addModalOpen, categoriesForModal, handleAddTerm, onCloseModal,
  toast, isMobile,
}: AppContentProps) {
  return (
    <>
      <div id="app" className="app">
        {/* Header */}
        <header className="header">
          <div className="header-left">
            <div className="logo">
              <div className="logo-icon">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="12" r="3" />
                  <path d="M12 2v4m0 12v4M2 12h4m12 0h4" />
                  <circle cx="12" cy="12" r="9" strokeDasharray="4 2" />
                </svg>
              </div>
              <span className="logo-text">TitanML</span>
            </div>
          </div>

          <div className="header-center">
            <div className="search-wrapper">
              <svg className="search-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="11" cy="11" r="8" />
                <path d="m21 21-4.35-4.35" />
              </svg>
              <input
                type="text"
                className="search-input"
                ref={searchInputRef}
                placeholder="Search concepts, terms, relationships..."
                aria-label="Search"
                onChange={onSearchInput}
                value={searchQuery}
              />
              <kbd className="search-kbd">/</kbd>
            </div>
          </div>

          <div className="header-right">
            <button className="btn btn-ghost" id="toggleSidebar" aria-label="Toggle sidebar" onClick={toggleSidebar}>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <rect x="3" y="3" width="18" height="18" rx="2" />
                <path d="M9 3v18" />
              </svg>
            </button>
            <button className="btn btn-primary" id="addTermBtn" onClick={onAddTerm}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M12 5v14m-7-7h14" />
              </svg>
              <span className="btn-text">Add Term</span>
            </button>
          </div>
        </header>

        {/* Main Layout */}
        <div className="layout">
          {/* Sidebar overlay for mobile */}
          <div
            className={`sidebar-overlay ${sidebarOpen ? 'active' : ''}`}
            onClick={closeSidebar}
          />

          {/* Sidebar */}
          <aside
            className={`sidebar ${sidebarOpen ? 'active' : ''} ${sidebarCollapsed ? 'collapsed' : ''}`}
            id="sidebar"
          >
            <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
              <div style={{ flex: 1, overflowY: 'auto' }}>
                {/* Categories */}
                <div className="sidebar-section">
                  <h3 className="sidebar-title">Categories</h3>
                  <div className="category-list" id="categoryList">
                    {categories.map((cat) => (
                      <div
                        key={cat.id}
                        className={`category-item ${selectedCategory === cat.id ? 'active' : ''}`}
                        data-category={cat.id}
                        onClick={() => handleCategoryClick(cat.id)}
                      >
                        <div className="category-dot" style={{ background: cat.color }} />
                        <span className="category-name">{cat.name}</span>
                        <span className="category-count">{stats?.byCategory[cat.id] ?? 0}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Quick Filters */}
                <div className="sidebar-section">
                  <h3 className="sidebar-title">Quick Filters</h3>
                  <div className="filter-list" id="filterList">
                    {[
                      { filter: 'all', label: 'All Terms', color: 'var(--accent-primary)', count: stats?.terms ?? 0 },
                      { filter: 'core', label: 'Core Concepts', color: 'var(--accent-warning)' },
                      { filter: 'technique', label: 'Techniques', color: 'var(--accent-secondary)' },
                    ].map((f) => (
                      <div
                        key={f.filter}
                        className={`filter-item ${selectedFilter === f.filter ? 'active' : ''}`}
                        data-filter={f.filter}
                        onClick={() => handleFilterClick(f.filter)}
                      >
                        <div className="filter-dot" style={{ background: f.color }} />
                        <span>{f.label}</span>
                        {f.count !== undefined && <span className="filter-count">{f.count}</span>}
                      </div>
                    ))}
                  </div>
                </div>

                {/* Explore */}
                <div className="sidebar-section">
                  <h3 className="sidebar-title">Explore</h3>
                  <div className="category-list">
                    <div className="category-item" style={{ cursor: 'pointer' }} onClick={showArchGallery}>
                      <div className="category-dot" style={{ background: 'var(--accent-secondary)' }} />
                      <span className="category-name">Architectures</span>
                    </div>
                  </div>
                </div>

                {/* Knowledge Paths */}
                <div className="sidebar-section">
                  <h3 className="sidebar-title">Curriculum</h3>
                  <div className="filter-list">
                    <div className="filter-item" style={{ cursor: 'pointer' }} onClick={showPathsView}>
                      <div className="filter-dot" style={{ background: 'var(--accent-secondary)' }} />
                      <span>Knowledge Paths</span>
                    </div>
                  </div>
                </div>

                {/* Statistics */}
                <div className="sidebar-section">
                  <h3 className="sidebar-title">Statistics</h3>
                  <div className="stats-grid">
                    <div className="stat-card">
                      <div className="stat-value">{stats?.categories ?? 0}</div>
                      <div className="stat-label">Categories</div>
                    </div>
                    <div className="stat-card">
                      <div className="stat-value">{stats?.terms ?? 0}</div>
                      <div className="stat-label">Terms</div>
                    </div>
                    <div className="stat-card">
                      <div className="stat-value">{stats?.connections ?? 0}</div>
                      <div className="stat-label">Connections</div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Sidebar Footer */}
              <div className="sidebar-footer">
                <div className="creator-card">
                  <div className="creator-avatar">AS</div>
                  <div className="creator-info">
                    <h4>Abhishek Shah</h4>
                    <p>Creator &amp; Developer</p>
                  </div>
                </div>

                <div className="creator-links">
                  <a href="mailto:abhishek.aimarine@gmail.com" className="creator-link" title="Email">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z" />
                      <polyline points="22,6 12,13 2,6" />
                    </svg>
                    Email
                  </a>
                  <a href="https://abhiverse01.github.io" target="_blank" rel="noopener noreferrer" className="creator-link" title="Portfolio">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
                      <polyline points="15 3 21 3 21 9" />
                      <line x1="10" y1="14" x2="21" y2="3" />
                    </svg>
                    Portfolio
                  </a>
                </div>

                <button
                  id="themeToggle"
                  onClick={toggleDarkMode}
                >
                  {isDarkMode ? '\u2600\uFE0F Light Mode' : '\uD83C\uDF19 Dark Mode'}
                </button>

                <div className="copyright-text" style={{ marginTop: 12 }}>
                  {'\u00A9'} {new Date().getFullYear()} Abhishek Shah. All rights reserved.
                </div>
              </div>
            </div>
          </aside>

          {/* Graph View */}
          <main
            className="content view-container"
            id="content"
            style={{ display: activeView === 'graph' ? 'flex' : 'none' }}
          >
            <div className="graph-container" id="graphContainer">
              <canvas id="graphCanvas" ref={canvasRef} />

              {/* Legend */}
              <div className="legend" id="legend">
                <div className="legend-title">Node Types</div>
                <div className="legend-items" id="legendItems">
                  {categories.map((cat) => (
                    <div key={cat.id} className="legend-item">
                      <div className="legend-dot" style={{ background: cat.color }} />
                      <span>{cat.name}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Controls */}
              <div className="graph-controls">
                <button className="control-btn" aria-label="Zoom in" onClick={() => graphRef.current?.zoomIn()}>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="11" cy="11" r="8" />
                    <path d="m21 21-4.35-4.35M11 8v6m-3-3h6" />
                  </svg>
                </button>
                <button className="control-btn" aria-label="Zoom out" onClick={() => graphRef.current?.zoomOut()}>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="11" cy="11" r="8" />
                    <path d="m21 21-4.35-4.35M8 11h6" />
                  </svg>
                </button>
                <button className="control-btn" aria-label="Reset view" onClick={() => graphRef.current?.resetView()}>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
                    <path d="M3 3v5h5" />
                  </svg>
                </button>
              </div>
            </div>
          </main>

        {/* Architecture View */}
        {activeView === 'architecture' && (
        <div
          id="archView"
          key="view-architecture"
          className="arch-visualizer view-container"
        >
          <div className="viz-header">
            <button className="viz-back-btn" onClick={archCurrentView === 'gallery' ? hideArchView : () => setArchCurrentView('gallery')}>
              &larr; Back to {archCurrentView === 'gallery' ? 'Graph' : 'Gallery'}
            </button>
            <div>
              <h2 style={{ fontSize: '1.5rem', color: 'var(--text-primary)' }}>
                {archCurrentView === 'gallery' ? 'AI Architectures' : currentArch?.name ?? ''}
              </h2>
              <p style={{ color: 'var(--text-tertiary)' }}>
                {archCurrentView === 'gallery'
                  ? 'Explore the blueprints of modern AI.'
                  : currentArch?.shortDesc ?? ''}
              </p>
            </div>
          </div>

          {/* Gallery */}
          <div
            id="archGallery"
            className="arch-gallery"
            style={{ display: archCurrentView === 'gallery' ? 'grid' : 'none' }}
          >
            {architectures.map((arch) => (
              <div key={arch.id} className="arch-card" data-id={arch.id} onClick={() => showArchDetail(arch.id)}>
                <div className="arch-category">{arch.category}</div>
                <div className="arch-title">{arch.name}</div>
                <div className="arch-desc">{arch.shortDesc}</div>
              </div>
            ))}
          </div>

          {/* Visualizer */}
          <div
            id="archVisualizer"
            style={{ display: archCurrentView === 'architecture' && currentArch ? 'block' : 'none' }}
          >
            {currentArch && (
              <div className="flow-container" dangerouslySetInnerHTML={{ __html: renderArchSteps(currentArch.steps as Parameters<typeof renderArchSteps>[0]) }} />
            )}
          </div>
          </div>
        )}

        {/* Knowledge Paths View */}
        {activeView === 'knowledge-paths' && (
        <div
          id="pathView"
          key="view-knowledge-paths"
          className="arch-visualizer view-container"
        >
          <div className="viz-header">
            <button
              className="viz-back-btn"
              onClick={() => {
                if (pathMode !== 'gallery') {
                  setPathMode('gallery')
                  setActivePath(null)
                  setPathSubPaths([])
                } else {
                  setActiveView('graph')
                }
              }}
            >
              &larr; {pathMode !== 'gallery' ? 'Back' : 'Back to Graph'}
            </button>
            <div>
              <h2 style={{ fontSize: '1.5rem', color: 'var(--text-primary)' }}>
                {pathMode === 'gallery'
                  ? 'Knowledge Paths'
                  : pathMode === 'subpaths'
                    ? 'Specializations'
                    : activePath?.title ?? ''}
              </h2>
              <p style={{ color: 'var(--text-tertiary)' }}>
                {pathMode === 'gallery'
                  ? 'Structured learning paths to master AI.'
                  : activePath?.description ?? ''}
              </p>
            </div>
          </div>

          {/* Path Gallery */}
          <div
            id="pathGallery"
            className="path-gallery"
            style={{
              display: pathMode === 'gallery' || pathMode === 'subpaths' ? 'grid' : 'none',
            }}
          >
            {(pathMode === 'gallery' ? knowledgePaths : pathSubPaths).map((path) => (
              <div key={path.id} className="path-card">
                <div className="path-title">{path.title}</div>
                <div className="path-desc">{path.description}</div>
                <div className="path-meta">
                  <span className="badge">{path.difficulty}</span>
                </div>
                <button className="path-start" onClick={() => startPath(path)}>
                  {pathMode === 'gallery' ? (path.paths ? 'Explore' : 'Start Path') : 'Explore'}
                </button>
              </div>
            ))}
          </div>

          {/* Path Steps */}
          <div id="pathViewer" style={{ display: pathMode === 'steps' ? 'block' : 'none' }}>
            {pathMode === 'steps' && activePath?.steps?.length ? (
              <div className="path-steps-container">
                {activePath.steps.map((step, i) => {
                  const exists = terms.some((t) => t.name === step || t.fullName === step)
                  return (
                    <div key={i} className="path-step">
                      <div className="step-index">{i + 1}</div>
                      <div className="step-name">{step}</div>
                      {exists ? (
                        <button className="step-open" onClick={() => focusTerm(step)}>Open</button>
                      ) : (
                        <span className="step-open" style={{ opacity: 0.4, cursor: 'default' }}>N/A</span>
                      )}
                    </div>
                  )
                })}
              </div>
            ) : pathMode === 'steps' ? (
              <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-muted)' }}>
                This path doesn't have detailed steps yet. Explore the graph to discover related concepts.
              </div>
            ) : null}
          </div>
        </div>
        )}
        </div>

        {/* Node Detail Panel */}
        <div className={`panel ${detailPanelOpen ? 'open' : ''}`} id="detailPanel">
          <div className="panel-header">
            <button className="panel-close" id="closePanel" aria-label="Close" onClick={onClosePanel}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M18 6 6 18M6 6l12 12" />
              </svg>
            </button>
            <div
              className="panel-badge"
              id="panelBadge"
              style={{
                color: selectedTermCategory?.color ?? '#6b7280',
                background: selectedTermCategory ? `${selectedTermCategory.color}15` : 'var(--bg-tertiary)',
              }}
            >
              {selectedTermCategory?.name ?? 'General'}
            </div>
            <h2 className="panel-title" id="panelTitle">
              {selectedTerm?.fullName || selectedTerm?.name || 'Term Name'}
            </h2>
            <p className="panel-subtitle" id="panelSubtitle">
              {selectedTerm?.shortDesc ?? 'Description'}
            </p>
          </div>
          <div className="panel-body">
            <section className="panel-section">
              <h4 className="section-title">Definition</h4>
              <p
                className="section-text"
                id="panelDefinition"
                dangerouslySetInnerHTML={{
                  __html: parseMarkdown(selectedTerm?.definition ?? ''),
                }}
              />
            </section>

            <section className="panel-section">
              <h4 className="section-title">Related Terms</h4>
              <div className="related-grid" id="relatedTerms">
                {relatedTerms.length > 0 ? (
                  relatedTerms.map(({ term: rt, cat: rc }) => (
                    <div
                      key={rt.id}
                      className="related-item"
                      role="button"
                      tabIndex={0}
                      onClick={() => onRelatedClick(rt.id)}
                      onKeyDown={(e) => { if (e.key === 'Enter') onRelatedClick(rt.id) }}
                    >
                      <div className="related-name">{rt.name}</div>
                      <div className="related-type">{rc?.name ?? 'General'}</div>
                    </div>
                  ))
                ) : (
                  <p style={{ color: 'var(--text-muted)', fontSize: 'var(--font-size-sm, 0.85rem)' }}>No related terms</p>
                )}
              </div>
            </section>

            <section className="panel-section">
              <h4 className="section-title">Code Example</h4>
              <div className="code-block">
                <button
                  className="copy-btn-code"
                  onClick={onCopyCode}
                  style={{ color: copyState === 'copied' ? '#4ade80' : copyState === 'failed' ? '#f87171' : undefined }}
                >
                  {copyState === 'copied' ? 'Copied' : copyState === 'failed' ? 'Failed' : 'Copy'}
                </button>
                <pre>
                  <code id="panelCode">{selectedTerm?.codeExample || '// No code example available'}</code>
                </pre>
              </div>
            </section>

            <section className="panel-section">
              <h4 className="section-title">Tags</h4>
              <div className="tag-list" id="panelTags">
                {selectedTerm?.tags?.map((tag) => <span key={tag} className="tag">{tag}</span>) ?? []}
              </div>
            </section>
          </div>
        </div>

        {/* Add Term Modal */}
        <div
          className={`modal-overlay ${addModalOpen ? 'open' : ''}`}
          id="addModal"
          onClick={(e) => {
            if (e.target === e.currentTarget) onCloseModal()
          }}
        >
          <div className="modal">
            <div className="modal-header">
              <h3 className="modal-title">Add New Term</h3>
              <button className="modal-close" aria-label="Close" onClick={onCloseModal}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M18 6 6 18M6 6l12 12" />
                </svg>
              </button>
            </div>
            <form className="modal-form" id="addTermForm" onSubmit={handleAddTerm}>
              <div className="form-group">
                <label className="form-label">Term Name *</label>
                <input type="text" name="name" className="form-input" required />
              </div>
              <div className="form-group">
                <label className="form-label">Category *</label>
                <select name="category" className="form-select" id="categorySelect" required defaultValue="">
                  <option value="" disabled>
                    Select a category...
                  </option>
                  {categoriesForModal.map((cat) => (
                    <option key={cat.id} value={cat.id}>
                      {cat.name}
                    </option>
                  ))}
                </select>
              </div>
              <div className="form-group">
                <label className="form-label">Short Description *</label>
                <input type="text" name="shortDesc" className="form-input" required />
              </div>
              <div className="form-group">
                <label className="form-label">Full Definition *</label>
                <textarea name="definition" className="form-textarea" rows={4} required />
              </div>
              <div className="form-row">
                <div className="form-group">
                  <label className="form-label">Related Terms</label>
                  <input type="text" name="related" className="form-input" placeholder="term1, term2" />
                </div>
                <div className="form-group">
                  <label className="form-label">Tags</label>
                  <input type="text" name="tags" className="form-input" placeholder="tag1, tag2" />
                </div>
              </div>
              <div className="form-actions">
                <button type="button" className="btn btn-ghost" onClick={onCloseModal}>
                  Cancel
                </button>
                <button type="submit" className="btn btn-primary">
                  Add Term
                </button>
              </div>
            </form>
          </div>
        </div>

        {/* Tooltip */}
        <div className="tooltip" id="tooltip" ref={tooltipRef} />
      </div>

      {/* Toast */}
      {toast && <Toast toast={toast} />}

      {/* Visitor Counter */}
      <div className="visitor-counter">
        {'\uD83E\uDDE0'} TitanML Minds Visited: <span id="visitorCount">{'\u2014'}</span>
      </div>

      {/* Desktop Note for mobile */}
      <div id="desktopNote">
        {'\u26A1'} For the best experience, view TitanML on desktop
      </div>
    </>
  )
}