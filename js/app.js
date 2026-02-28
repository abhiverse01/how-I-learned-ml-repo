/**
 * AI Knowledge Nexus - Main Application
 */

class App {
    constructor() {
        this.graph = null;
        this.state = {
            searchQuery: '',
            selectedCategory: null,
            selectedTerm: null,
            sidebarOpen: true
        };
        
        this.init();
    }
    
    init() {
        // Wait for DOM and KnowledgeBase
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.setup());
        } else {
            this.setup();
        }
    }
    
    setup() {
        // Initialize graph
        this.graph = new KnowledgeGraph('graphCanvas');
        this.graph.loadData();
        
        // Setup callbacks
        this.graph.onNodeSelect = (term) => this.showTermDetail(term);
        this.graph.onHoverChange = (node, e) => this.handleHover(node, e);
        
        // Render UI
        this.renderCategories();
        this.renderLegend();
        this.updateStats();
        this.populateCategorySelect();
        
        // Setup event listeners
        this.setupEventListeners();
    }
    
    setupEventListeners() {
        // Search
        const searchInput = document.getElementById('searchInput');
        searchInput.addEventListener('input', (e) => {
            this.state.searchQuery = e.target.value;
            this.graph.highlightNodes(this.state.searchQuery);
        });
        
        // Keyboard shortcut
        document.addEventListener('keydown', (e) => {
            if (e.key === '/' && document.activeElement !== searchInput) {
                e.preventDefault();
                searchInput.focus();
            }
            if (e.key === 'Escape') {
                this.closeDetailPanel();
                this.closeModal();
                searchInput.blur();
            }
        });
        
        // Sidebar toggle
        document.getElementById('toggleSidebar').addEventListener('click', () => {
            document.getElementById('sidebar').classList.toggle('collapsed');
        });
        
        // Category selection
        document.getElementById('categoryList').addEventListener('click', (e) => {
            const item = e.target.closest('.category-item');
            if (!item) return;
            
            const categoryId = item.dataset.category;
            
            document.querySelectorAll('.category-item').forEach(i => i.classList.remove('active'));
            
            if (this.state.selectedCategory === categoryId) {
                this.state.selectedCategory = null;
                this.graph.filterByCategory(null);
            } else {
                item.classList.add('active');
                this.state.selectedCategory = categoryId;
                this.graph.filterByCategory(categoryId);
            }
        });
        
        // Filters
        document.getElementById('filterList').addEventListener('click', (e) => {
            const item = e.target.closest('.filter-item');
            if (!item) return;
            
            document.querySelectorAll('.filter-item').forEach(i => i.classList.remove('active'));
            item.classList.add('active');
            
            const filter = item.dataset.filter;
            if (filter === 'all') {
                this.state.selectedCategory = null;
                this.graph.filterByCategory(null);
            } else if (filter === 'core') {
                this.graph.nodes.forEach(n => n.visible = n.term.type === 'core');
            } else if (filter === 'technique') {
                this.graph.nodes.forEach(n => n.visible = n.term.type === 'technique');
            }
        });
        
        // Graph controls
        document.getElementById('zoomIn').addEventListener('click', () => this.graph.zoomIn());
        document.getElementById('zoomOut').addEventListener('click', () => this.graph.zoomOut());
        document.getElementById('resetView').addEventListener('click', () => this.graph.resetView());
        
        // Detail panel
        document.getElementById('closePanel').addEventListener('click', () => this.closeDetailPanel());
        
        // Related term clicks
        document.getElementById('relatedTerms').addEventListener('click', (e) => {
            const item = e.target.closest('.related-item');
            if (!item) return;
            
            const termId = item.dataset.termId;
            const term = KnowledgeUtils.getTerm(termId);
            if (term) {
                this.showTermDetail(term);
                this.graph.selectedNode = this.graph.findNode(termId);
            }
        });
        
        // Add term modal
        document.getElementById('addTermBtn').addEventListener('click', () => this.openModal());
        document.getElementById('closeModal').addEventListener('click', () => this.closeModal());
        document.getElementById('cancelAdd').addEventListener('click', () => this.closeModal());
        document.getElementById('addModal').addEventListener('click', (e) => {
            if (e.target.id === 'addModal') this.closeModal();
        });
        
        // Add term form
        document.getElementById('addTermForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.handleAddTerm(new FormData(e.target));
        });
        
        // Window resize
        window.addEventListener('resize', () => {
            this.graph.handleResize();
        });
    }
    
    renderCategories() {
        const container = document.getElementById('categoryList');
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
        
        container.innerHTML = KnowledgeBase.categories.map(cat => `
            <div class="legend-item">
                <div class="legend-dot" style="background: ${cat.color};"></div>
                <span>${cat.name}</span>
            </div>
        `).join('');
    }
    
    updateStats() {
        const stats = KnowledgeUtils.getStats();
        document.getElementById('statCategories').textContent = stats.categories;
        document.getElementById('statTerms').textContent = stats.terms;
        document.getElementById('statConnections').textContent = stats.connections;
        document.getElementById('totalCount').textContent = stats.terms;
    }
    
    populateCategorySelect() {
        const select = document.getElementById('categorySelect');
        select.innerHTML = KnowledgeBase.categories.map(cat => 
            `<option value="${cat.id}">${cat.name}</option>`
        ).join('');
    }
    
    showTermDetail(term) {
        const panel = document.getElementById('detailPanel');
        const category = KnowledgeBase.categories.find(c => c.id === term.category);
        
        // Update content
        document.getElementById('panelBadge').textContent = category ? category.name : 'General';
        document.getElementById('panelBadge').style.color = category ? category.color : '#6b7280';
        document.getElementById('panelBadge').style.background = category ? category.color + '15' : '#f3f4f6';
        document.getElementById('panelTitle').textContent = term.fullName || term.name;
        document.getElementById('panelSubtitle').textContent = term.shortDesc;
        document.getElementById('panelDefinition').textContent = term.definition;
        
        // Related terms
        const relatedContainer = document.getElementById('relatedTerms');
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
            }).join('');
        } else {
            relatedContainer.innerHTML = '<p style="color: var(--text-muted); font-size: var(--font-size-sm);">No related terms</p>';
        }
        
        // Code example
        const codeContainer = document.getElementById('panelCode');
        codeContainer.textContent = term.codeExample || '// No code example available';
        
        // Tags
        const tagContainer = document.getElementById('panelTags');
        tagContainer.innerHTML = term.tags.map(tag => 
            `<span class="tag">${tag}</span>`
        ).join('');
        
        panel.classList.add('open');
        this.state.selectedTerm = term;
    }
    
    closeDetailPanel() {
        document.getElementById('detailPanel').classList.remove('open');
        this.state.selectedTerm = null;
        this.graph.selectedNode = null;
    }
    
    handleHover(node, e) {
        const tooltip = document.getElementById('tooltip');
        
        if (node) {
            tooltip.innerHTML = `<strong>${node.term.name}</strong><br><span style="color: var(--text-muted);">${node.term.shortDesc}</span>`;
            tooltip.style.left = e.clientX + 12 + 'px';
            tooltip.style.top = e.clientY + 12 + 'px';
            tooltip.classList.add('visible');
        } else {
            tooltip.classList.remove('visible');
        }
    }
    
    openModal() {
        document.getElementById('addModal').classList.add('open');
    }
    
    closeModal() {
        document.getElementById('addModal').classList.remove('open');
        document.getElementById('addTermForm').reset();
    }
    
    handleAddTerm(formData) {
        const name = formData.get('name').trim();
        const categoryId = formData.get('category');
        const shortDesc = formData.get('shortDesc').trim();
        const definition = formData.get('definition').trim();
        const relatedStr = formData.get('related') || '';
        const tagsStr = formData.get('tags') || '';
        
        // Generate ID from name
        const id = name.toLowerCase().replace(/\s+/g, '-').replace(/[^a-z0-9-]/g, '');
        
        // Parse related and tags
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
            this.graph.loadData();
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

// Initialize app
const app = new App();
