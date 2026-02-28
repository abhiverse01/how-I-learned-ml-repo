/**
 * AI Knowledge Nexus - Main Application (FIXED)
 */

class App {
    constructor() {
        this.graph = null;
        this.state = {
            searchQuery: '',
            selectedCategory: null,
            selectedTerm: null
        };
        this.initialized = false;
    }
    
    init() {
        // Wait for DOM ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.setup());
        } else {
            this.setup();
        }
    }
    
    setup() {
        console.log('Setting up app...');
        
        // Check dependencies
        if (!window.KnowledgeBase) {
            console.error('KnowledgeBase not loaded');
            return;
        }
        if (!window.KnowledgeGraph) {
            console.error('KnowledgeGraph not loaded');
            return;
        }
        if (!window.KnowledgeUtils) {
            console.error('KnowledgeUtils not loaded');
            return;
        }
        
        // Initialize graph
        this.graph = new KnowledgeGraph('graphCanvas');
        this.graph.loadData();
        
        // Set callbacks
        this.graph.onNodeSelect = (term) => this.showTermDetail(term);
        this.graph.onHoverChange = (node, e) => this.handleHover(node, e);
        
        // Render UI
        this.renderCategories();
        this.renderLegend();
        this.updateStats();
        this.populateCategorySelect();
        
        // Bind events
        this.bindEvents();
        
        this.initialized = true;
        console.log('App initialized');
    }
    
    bindEvents() {
        // Search
        const searchInput = document.getElementById('searchInput');
        if (searchInput) {
            searchInput.addEventListener('input', (e) => {
                this.state.searchQuery = e.target.value;
                if (this.graph) this.graph.highlightNodes(this.state.searchQuery);
            });
        }
        
        // Keyboard
        document.addEventListener('keydown', (e) => {
            if (e.key === '/' && document.activeElement !== searchInput) {
                e.preventDefault();
                if (searchInput) searchInput.focus();
            }
            if (e.key === 'Escape') {
                this.closeDetailPanel();
                this.closeModal();
                if (searchInput) searchInput.blur();
            }
        });
        
        // Sidebar toggle - FIXED for mobile/desktop compatibility
        const toggleBtn = document.getElementById('toggleSidebar');
        if (toggleBtn) {
            toggleBtn.addEventListener('click', () => {
                const sidebar = document.getElementById('sidebar');
                
                // On desktop, we toggle 'collapsed'. On mobile, we toggle 'open'.
                if (window.innerWidth <= 1024) {
                    sidebar.classList.toggle('open'); 
                } else {
                    sidebar.classList.toggle('collapsed');
                }
            });
        }
        // REMOVED THE EXTRA } HERE
        
        // Category list
        const categoryList = document.getElementById('categoryList');
        if (categoryList) {
            categoryList.addEventListener('click', (e) => {
                const item = e.target.closest('.category-item');
                if (!item) return;
                
                document.querySelectorAll('.category-item').forEach(i => i.classList.remove('active'));
                
                const categoryId = item.dataset.category;
                if (this.state.selectedCategory === categoryId) {
                    this.state.selectedCategory = null;
                    if (this.graph) this.graph.filterByCategory(null);
                } else {
                    item.classList.add('active');
                    this.state.selectedCategory = categoryId;
                    if (this.graph) this.graph.filterByCategory(categoryId);
                }
            });
        }
        
        // Filters
        const filterList = document.getElementById('filterList');
        if (filterList) {
            filterList.addEventListener('click', (e) => {
                const item = e.target.closest('.filter-item');
                if (!item) return;
                
                document.querySelectorAll('.filter-item').forEach(i => i.classList.remove('active'));
                item.classList.add('active');
                
                const filter = item.dataset.filter;
                if (this.graph) {
                    if (filter === 'all') {
                        this.state.selectedCategory = null;
                        this.graph.nodes.forEach(n => n.visible = true);
                    } else {
                        this.graph.nodes.forEach(n => n.visible = n.term.type === filter);
                    }
                }
            });
        }
        
        // Graph controls
        const zoomIn = document.getElementById('zoomIn');
        const zoomOut = document.getElementById('zoomOut');
        const resetView = document.getElementById('resetView');
        
        if (zoomIn) zoomIn.addEventListener('click', () => this.graph && this.graph.zoomIn());
        if (zoomOut) zoomOut.addEventListener('click', () => this.graph && this.graph.zoomOut());
        if (resetView) resetView.addEventListener('click', () => this.graph && this.graph.resetView());
        
        // Detail panel
        const closePanel = document.getElementById('closePanel');
        if (closePanel) {
            closePanel.addEventListener('click', () => this.closeDetailPanel());
        }
        
        // Related terms
        const relatedTerms = document.getElementById('relatedTerms');
        if (relatedTerms) {
            relatedTerms.addEventListener('click', (e) => {
                const item = e.target.closest('.related-item');
                if (!item) return;
                
                const termId = item.dataset.termId;
                const term = KnowledgeUtils.getTerm(termId);
                if (term) {
                    this.showTermDetail(term);
                    if (this.graph) this.graph.selectedNode = this.graph.findNode(termId);
                }
            });
        }
        
        // Add term modal
        const addTermBtn = document.getElementById('addTermBtn');
        const closeModalBtn = document.getElementById('closeModal');
        const cancelAdd = document.getElementById('cancelAdd');
        const addModal = document.getElementById('addModal');
        
        if (addTermBtn) addTermBtn.addEventListener('click', () => this.openModal());
        if (closeModalBtn) closeModalBtn.addEventListener('click', () => this.closeModal());
        if (cancelAdd) cancelAdd.addEventListener('click', () => this.closeModal());
        if (addModal) {
            addModal.addEventListener('click', (e) => {
                if (e.target.id === 'addModal') this.closeModal();
            });
        }
        
        // Add term form
        const addTermForm = document.getElementById('addTermForm');
        if (addTermForm) {
            addTermForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.handleAddTerm(new FormData(e.target));
            });
        }
    }
    
    renderCategories() {
        const container = document.getElementById('categoryList');
        if (!container || !window.KnowledgeUtils) return;
        
        const stats = KnowledgeUtils.getStats();
        
        container.innerHTML = KnowledgeBase.categories.map(cat => `
            <div class="category-item" data-category="${cat.id}">
                <div class="category-dot" style="background: ${cat.color};"></div>
                <span class="category-name">${cat.name}</span>
                <span class="category-count">${stats.byCategory[cat.id] || 0}</span>
            </div>
        `).join('');
    }
    
    renderLegend() {
        const container = document.getElementById('legendItems');
        if (!container || !window.KnowledgeBase) return;
        
        container.innerHTML = KnowledgeBase.categories.map(cat => `
            <div class="legend-item">
                <div class="legend-dot" style="background: ${cat.color};"></div>
                <span>${cat.name}</span>
            </div>
        `).join('');
    }
    
    updateStats() {
        if (!window.KnowledgeUtils) return;
        
        const stats = KnowledgeUtils.getStats();
        
        const statCategories = document.getElementById('statCategories');
        const statTerms = document.getElementById('statTerms');
        const statConnections = document.getElementById('statConnections');
        const totalCount = document.getElementById('totalCount');
        
        if (statCategories) statCategories.textContent = stats.categories;
        if (statTerms) statTerms.textContent = stats.terms;
        if (statConnections) statConnections.textContent = stats.connections;
        if (totalCount) totalCount.textContent = stats.terms;
    }
    
    populateCategorySelect() {
        const select = document.getElementById('categorySelect');
        if (!select || !window.KnowledgeBase) return;
        
        select.innerHTML = KnowledgeBase.categories.map(cat => 
            `<option value="${cat.id}">${cat.name}</option>`
        ).join('');
    }
    
    showTermDetail(term) {
        if (!term) return;
        
        const panel = document.getElementById('detailPanel');
        if (!panel) return;
        
        const category = KnowledgeBase.categories.find(c => c.id === term.category);
        
        // Badge
        const badge = document.getElementById('panelBadge');
        if (badge) {
            badge.textContent = category ? category.name : 'General';
            badge.style.color = category ? category.color : '#6b7280';
            badge.style.background = category ? category.color + '15' : '#f3f4f6';
        }
        
        // Title & subtitle
        const title = document.getElementById('panelTitle');
        const subtitle = document.getElementById('panelSubtitle');
        if (title) title.textContent = term.fullName || term.name;
        if (subtitle) subtitle.textContent = term.shortDesc;
        
        // Definition
        const definition = document.getElementById('panelDefinition');
        if (definition) definition.textContent = term.definition;
        
        // Related terms
        const relatedContainer = document.getElementById('relatedTerms');
        if (relatedContainer) {
            if (term.related && term.related.length > 0) {
                relatedContainer.innerHTML = term.related.map(relId => {
                    const relTerm = KnowledgeUtils.getTerm(relId);
                    if (!relTerm) return '';
                    const relCat = KnowledgeBase.categories.find(c => c.id === relTerm.category);
                    return `
                        <div class="related-item" data-term-id="${relId}">
                            <div class="related-name">${relTerm.name}</div>
                            <div class="related-type">${relCat ? relCat.name : 'General'}</div>
                        </div>
                    `;
                }).filter(Boolean).join('') || '<p style="color:var(--text-muted);font-size:var(--font-size-sm);">No related terms found</p>';
            } else {
                relatedContainer.innerHTML = '<p style="color:var(--text-muted);font-size:var(--font-size-sm);">No related terms</p>';
            }
        }
        
        // Code
        const codeContainer = document.getElementById('panelCode');
        if (codeContainer) {
            codeContainer.textContent = term.codeExample || '// No code example available';
        }
        
        // Tags
        const tagContainer = document.getElementById('panelTags');
        if (tagContainer) {
            tagContainer.innerHTML = term.tags.map(tag => 
                `<span class="tag">${tag}</span>`
            ).join('');
        }
        
        panel.classList.add('open');
        this.state.selectedTerm = term;
    }
    
    closeDetailPanel() {
        const panel = document.getElementById('detailPanel');
        if (panel) panel.classList.remove('open');
        this.state.selectedTerm = null;
        if (this.graph) this.graph.selectedNode = null;
    }
    
    handleHover(node, e) {
        const tooltip = document.getElementById('tooltip');
        if (!tooltip) return;
        
        if (node) {
            tooltip.innerHTML = `<strong>${node.term.name}</strong><br><span style="color:var(--text-muted);">${node.term.shortDesc}</span>`;
            tooltip.style.left = (e.clientX + 12) + 'px';
            tooltip.style.top = (e.clientY + 12) + 'px';
            tooltip.classList.add('visible');
        } else {
            tooltip.classList.remove('visible');
        }
    }
    
    openModal() {
        const modal = document.getElementById('addModal');
        if (modal) modal.classList.add('open');
    }
    
    closeModal() {
        const modal = document.getElementById('addModal');
        const form = document.getElementById('addTermForm');
        if (modal) modal.classList.remove('open');
        if (form) form.reset();
    }
    
    handleAddTerm(formData) {
        const name = formData.get('name');
        const categoryId = formData.get('category');
        const shortDesc = formData.get('shortDesc');
        const definition = formData.get('definition');
        const relatedStr = formData.get('related') || '';
        const tagsStr = formData.get('tags') || '';
        
        if (!name || !categoryId || !shortDesc || !definition) {
            alert('Please fill all required fields');
            return;
        }
        
        // Generate ID
        const id = name.toLowerCase().replace(/\s+/g, '-').replace(/[^a-z0-9-]/g, '');
        
        // Parse arrays
        const related = relatedStr.split(',').map(s => s.trim().toLowerCase().replace(/\s+/g, '-')).filter(Boolean);
        const tags = tagsStr.split(',').map(s => s.trim()).filter(Boolean);
        
        // Add term
        const success = KnowledgeUtils.addTerm({
            id,
            name,
            category: categoryId,
            type: 'technique',
            shortDesc,
            definition,
            related,
            tags
        });
        
        if (success) {
            // Reload graph
            if (this.graph) this.graph.loadData();
            this.updateStats();
            this.renderCategories();
            this.closeModal();
            
            // Show new term
            const newTerm = KnowledgeUtils.getTerm(id);
            if (newTerm) {
                setTimeout(() => this.showTermDetail(newTerm), 100);
            }
        } else {
            alert('Failed to add term. It may already exist.');
        }
    }
}

// Initialize
const app = new App();
