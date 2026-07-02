import { create } from 'zustand'

export interface Term {
  id: string
  name: string
  fullName: string
  category: string
  type: string
  shortDesc: string
  definition: string
  related: string[]
  tags: string[]
  codeExample: string
  createdAt: string
  importance: number
}

export interface Category {
  id: string
  name: string
  fullName: string
  color: string
  description: string
}

export interface Architecture {
  id: string
  name: string
  category: string
  shortDesc: string
  steps: ArchStep[]
}

export interface ArchStep {
  label: string
  desc?: string
  type?: string
  inputId?: string
  children?: ArchStep[]
}

export interface KnowledgePath {
  id: string
  title: string
  description: string
  difficulty: string
  steps?: string[]
  paths?: KnowledgePath[]
}

interface Stats {
  categories: number
  terms: number
  connections: number
  byCategory: Record<string, number>
}

type AppView = 'graph' | 'architecture' | 'knowledge-paths'

interface TitanMLState {
  // Data
  categories: Category[]
  terms: Term[]
  isLoaded: boolean

  // Architecture data
  architectures: Architecture[]
  architecturesLoaded: boolean

  // Knowledge paths data
  knowledgePaths: KnowledgePath[]
  pathsLoaded: boolean

  // UI State
  activeView: AppView
  searchQuery: string
  selectedCategory: string | null
  selectedTerm: Term | null
  selectedFilter: string
  sidebarOpen: boolean
  sidebarCollapsed: boolean
  detailPanelOpen: boolean
  addModalOpen: boolean
  isDarkMode: boolean
  isLoading: boolean
  stats: Stats | null

  // Architecture sub-state
  archCurrentView: 'gallery' | 'architecture'
  archCurrentId: string | null

  // Knowledge paths sub-state
  pathMode: 'gallery' | 'steps' | 'subpaths'
  activePath: KnowledgePath | null
  pathSubPaths: KnowledgePath[]

  // History
  history: string[]
  historyIndex: number

  // Actions - Data
  setCategories: (cats: Category[]) => void
  setTerms: (terms: Term[]) => void
  setDataLoaded: () => void
  setArchitectures: (archs: Architecture[]) => void
  setKnowledgePaths: (paths: KnowledgePath[]) => void
  addTerm: (term: Term) => void

  // Actions - UI
  setActiveView: (view: AppView) => void
  setSearchQuery: (q: string) => void
  setSelectedCategory: (cat: string | null) => void
  setSelectedTerm: (term: Term | null) => void
  setSelectedFilter: (filter: string) => void
  toggleSidebar: () => void
  closeSidebar: () => void
  toggleSidebarCollapsed: () => void
  setDetailPanelOpen: (open: boolean) => void
  setAddModalOpen: (open: boolean) => void
  toggleDarkMode: () => void
  setIsLoading: (loading: boolean) => void
  setStats: (stats: Stats) => void

  // Architecture actions
  setArchCurrentView: (view: 'gallery' | 'architecture') => void
  setArchCurrentId: (id: string | null) => void

  // Path actions
  setPathMode: (mode: 'gallery' | 'steps' | 'subpaths') => void
  setActivePath: (path: KnowledgePath | null) => void
  setPathSubPaths: (paths: KnowledgePath[]) => void

  // History actions
  navigateTerm: (termId: string) => void
  goBack: () => void
  goForward: () => void

  // Utility
  getTerm: (id: string) => Term | undefined
  getTermsByCategory: (categoryId: string) => Term[]
  searchTerms: (query: string) => Term[]
}

export const useTitanMLStore = create<TitanMLState>((set, get) => ({
  // Data
  categories: [],
  terms: [],
  isLoaded: false,

  // Architecture data
  architectures: [],
  architecturesLoaded: false,

  // Knowledge paths data
  knowledgePaths: [],
  pathsLoaded: false,

  // UI State
  activeView: 'graph',
  searchQuery: '',
  selectedCategory: null,
  selectedTerm: null,
  selectedFilter: 'all',
  sidebarOpen: false,
  sidebarCollapsed: false,
  detailPanelOpen: false,
  addModalOpen: false,
  isDarkMode: false,
  isLoading: true,
  stats: null,

  // Architecture sub-state
  archCurrentView: 'gallery',
  archCurrentId: null,

  // Knowledge paths sub-state
  pathMode: 'gallery',
  activePath: null,
  pathSubPaths: [],

  // History
  history: [],
  historyIndex: -1,

  // Actions - Data
  setCategories: (cats) => set({ categories: cats }),
  setTerms: (terms) => set({ terms }),
  setDataLoaded: () => set({ isLoaded: true }),
  setArchitectures: (archs) => set({ architectures: archs, architecturesLoaded: true }),
  setKnowledgePaths: (paths) => set({ knowledgePaths: paths, pathsLoaded: true }),

  addTerm: (term) =>
    set((state) => {
      if (state.terms.find((t) => t.id === term.id)) return state
      const newTerms = [...state.terms, term]
      const newStats = computeStats(state.categories, newTerms)
      return { terms: newTerms, stats: newStats }
    }),

  // Actions - UI
  setActiveView: (view) => set({ activeView: view }),
  setSearchQuery: (q) => set({ searchQuery: q }),
  setSelectedCategory: (cat) => set({ selectedCategory: cat }),
  setSelectedTerm: (term) => set({ selectedTerm: term, detailPanelOpen: !!term }),
  setSelectedFilter: (filter) => set({ selectedFilter: filter }),
  toggleSidebar: () =>
    set((state) => {
      if (typeof window !== 'undefined' && window.innerWidth < 768) {
        return { sidebarOpen: !state.sidebarOpen }
      }
      return { sidebarCollapsed: !state.sidebarCollapsed }
    }),
  closeSidebar: () => set({ sidebarOpen: false }),
  toggleSidebarCollapsed: () => set((s) => ({ sidebarCollapsed: !s.sidebarCollapsed })),
  // Fix: don't clear selectedTerm when closing panel — keep it for re-open
  setDetailPanelOpen: (open) => set({ detailPanelOpen: open }),
  setAddModalOpen: (open) => set({ addModalOpen: open }),
  toggleDarkMode: () =>
    set((state) => {
      const newDark = !state.isDarkMode
      if (typeof document !== 'undefined') {
        document.documentElement.setAttribute('data-theme', newDark ? 'dark' : 'light')
      }
      return { isDarkMode: newDark }
    }),
  setIsLoading: (loading) => set({ isLoading: loading }),
  setStats: (stats) => set({ stats }),

  // Architecture actions
  setArchCurrentView: (view) => set({ archCurrentView: view }),
  setArchCurrentId: (id) => set({ archCurrentId: id }),

  // Path actions
  setPathMode: (mode) => set({ pathMode: mode }),
  setActivePath: (path) => set({ activePath: path }),
  setPathSubPaths: (paths) => set({ pathSubPaths: paths }),

  // History actions
  navigateTerm: (termId) =>
    set((state) => {
      const newHistory = state.history.slice(0, state.historyIndex + 1)
      if (newHistory[newHistory.length - 1] !== termId) {
        newHistory.push(termId)
      }
      return { history: newHistory, historyIndex: newHistory.length - 1 }
    }),
  goBack: () =>
    set((state) => {
      if (state.historyIndex <= 0) return state
      const newIndex = state.historyIndex - 1
      const termId = state.history[newIndex]
      const term = state.terms.find((t) => t.id === termId)
      return { historyIndex: newIndex, selectedTerm: term ?? null, detailPanelOpen: !!term }
    }),
  goForward: () =>
    set((state) => {
      if (state.historyIndex >= state.history.length - 1) return state
      const newIndex = state.historyIndex + 1
      const termId = state.history[newIndex]
      const term = state.terms.find((t) => t.id === termId)
      return { historyIndex: newIndex, selectedTerm: term ?? null, detailPanelOpen: !!term }
    }),

  // Utility
  getTerm: (id) => get().terms.find((t) => t.id === id),
  getTermsByCategory: (categoryId) => get().terms.filter((t) => t.category === categoryId),
  searchTerms: (query) => {
    if (!query) return get().terms
    const q = query.toLowerCase()
    return get().terms.filter(
      (t) =>
        t.name.toLowerCase().includes(q) ||
        t.shortDesc.toLowerCase().includes(q) ||
        t.fullName.toLowerCase().includes(q) ||
        t.tags.some((tag) => tag.toLowerCase().includes(q))
    )
  },
}))

function computeStats(categories: Category[], terms: Term[]): Stats {
  const byCategory: Record<string, number> = {}
  for (let i = 0; i < categories.length; i++) {
    const catId = categories[i].id
    let count = 0
    for (let j = 0; j < terms.length; j++) {
      if (terms[j].category === catId) count++
    }
    byCategory[catId] = count
  }
  // Deduplicate connections — each edge is counted once regardless of direction
  const edgeSet = new Set<string>()
  let connections = 0
  for (let i = 0; i < terms.length; i++) {
    const related = terms[i].related
    if (!related) continue
    for (let j = 0; j < related.length; j++) {
      const key = [terms[i].id, related[j]].sort().join('|')
      if (!edgeSet.has(key)) {
        edgeSet.add(key)
        connections++
      }
    }
  }
  return {
    categories: categories.length,
    terms: terms.length,
    connections,
    byCategory,
  }
}