/**
 * Knowledge Graph Visualization - OPTIMIZED & ENHANCED
 */

class KnowledgeGraph {
    constructor(canvasId, options = {}) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) {
            console.error('Canvas not found:', canvasId);
            return;
        }
        
        this.ctx = this.canvas.getContext('2d');
        
        // Options
        this.options = {
            nodeRadius: { core: 22, technique: 16, infrastructure: 14, application: 12 },
            fontSize: 10,
            padding: 60,
            ...options
        };
        
        // State
        this.nodes = [];
        this.edges = [];
        this.nodeMap = new Map(); // Performance: O(1) lookups
        this.zoom = 1;
        this.panX = 0;
        this.panY = 0;
        this.hoveredNode = null;
        this.selectedNode = null;
        this.isDragging = false;
        this.lastMouse = { x: 0, y: 0 };
        this.animationId = null;
        
        // Logical dimensions (independent of screen pixels)
        this.width = 0;
        this.height = 0;
        this.centerX = 0;
        this.centerY = 0;
        this.dpr = window.devicePixelRatio || 1;
        
        // Physics config
        this.physics = {
            enabled: true,
            repulsion: 600,
            attraction: 0.006,
            centerGravity: 0.008,
            damping: 0.88,
            minVelocity: 0.05
        };
        
        // Callbacks
        this.onNodeSelect = null;
        this.onHoverChange = null;
        
        this.init();
    }
    
    init() {
        this.handleResize();
        this.bindEvents();
        this.startAnimation();
    }
    
    bindEvents() {
        window.addEventListener('resize', () => this.handleResize());
        
        // Mouse Events
        this.canvas.addEventListener('mousemove', (e) => this.handleMouseMove(e));
        this.canvas.addEventListener('mousedown', (e) => this.handleMouseDown(e));
        this.canvas.addEventListener('mouseup', () => this.handleMouseUp());
        this.canvas.addEventListener('mouseleave', () => this.handleMouseLeave());
        this.canvas.addEventListener('wheel', (e) => this.handleWheel(e), { passive: false });
        this.canvas.addEventListener('click', (e) => this.handleClick(e));
        
        // Touch Events for Mobile
        this.canvas.addEventListener('touchstart', (e) => this.handleTouchStart(e), { passive: false });
        this.canvas.addEventListener('touchmove', (e) => this.handleTouchMove(e), { passive: false });
        this.canvas.addEventListener('touchend', () => this.handleMouseUp());
    }
    
    handleResize() {
        const rect = this.canvas.parentElement.getBoundingClientRect();
        this.width = rect.width || 800;
        this.height = rect.height || 600;
        this.centerX = this.width / 2;
        this.centerY = this.height / 2;
        
        // HiDPI / Retina support
        this.dpr = window.devicePixelRatio || 1;
        this.canvas.width = this.width * this.dpr;
        this.canvas.height = this.height * this.dpr;
        this.canvas.style.width = `${this.width}px`;
        this.canvas.style.height = `${this.height}px`;
        
        // Reset transform and scale for crisp drawing
        this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    }
    
    loadData() {
        if (!window.KnowledgeBase) {
            console.error('KnowledgeBase not found');
            return;
        }
        
        this.nodes = [];
        this.edges = [];
        this.nodeMap.clear();
        
        const categories = KnowledgeBase.categories;
        const terms = KnowledgeBase.terms;
        
        if (!categories.length || !terms.length) {
            console.error('No data in KnowledgeBase');
            return;
        }
        
        // Create nodes
        terms.forEach((term) => {
            const category = categories.find(c => c.id === term.category);
            const categoryIndex = categories.indexOf(category);
            const totalCategories = categories.length || 1;
            
            // Position calculation
            const baseAngle = (categoryIndex / totalCategories) * Math.PI * 2 - Math.PI / 2;
            const categoryTerms = terms.filter(t => t.category === term.category);
            const termIndex = categoryTerms.indexOf(term);
            
            const radius = this.options.nodeRadius[term.type] || 16;
            let distance = 180;
            if (term.type === 'core') distance = 140;
            else if (term.type === 'technique') distance = 200;
            else distance = 250;
            
            const spreadAngle = Math.PI / 5;
            const angleOffset = categoryTerms.length > 1 
                ? (termIndex - (categoryTerms.length - 1) / 2) * (spreadAngle / categoryTerms.length)
                : 0;
            const angle = baseAngle + angleOffset;
            
            const jitterX = (Math.random() - 0.5) * 30;
            const jitterY = (Math.random() - 0.5) * 30;
            
            const node = {
                id: term.id,
                x: this.centerX + Math.cos(angle) * distance + jitterX,
                y: this.centerY + Math.sin(angle) * distance + jitterY,
                vx: 0,
                vy: 0,
                radius: radius,
                term: term,
                color: category ? category.color : '#6b7280',
                highlighted: false,
                visible: true
            };
            
            this.nodes.push(node);
            this.nodeMap.set(term.id, node); // Populate Map
        });
        
        // Create edges
        terms.forEach(term => {
            if (term.related && term.related.length > 0) {
                term.related.forEach(relatedId => {
                    const exists = this.edges.some(e => 
                        (e.source === term.id && e.target === relatedId) ||
                        (e.source === relatedId && e.target === term.id)
                    );
                    if (!exists) {
                        this.edges.push({
                            source: term.id,
                            target: relatedId,
                            strength: 1
                        });
                    }
                });
            }
        });
        
        // Initial physics settle
        for (let i = 0; i < 50; i++) {
            this.simulatePhysics(0.15);
        }
        
        console.log('Graph loaded:', this.nodes.length, 'nodes,', this.edges.length, 'edges');
    }
    
    simulatePhysics(dt = 1) {
        if (!this.physics.enabled) return;
        
        const nodes = this.nodes;
        const eps = 0.001;
        
        nodes.forEach(node => {
            if (!node.visible) return;
            
            // Center gravity
            node.vx += (this.centerX - node.x) * this.physics.centerGravity * dt;
            node.vy += (this.centerY - node.y) * this.physics.centerGravity * dt;
            
            // Repulsion
            nodes.forEach(other => {
                if (node.id === other.id || !other.visible) return;
                
                const dx = node.x - other.x;
                const dy = node.y - other.y;
                const distSq = dx * dx + dy * dy;
                const dist = Math.max(Math.sqrt(distSq), 1);
                
                if (dist < 150) {
                    const force = this.physics.repulsion / (distSq + eps);
                    node.vx += (dx / dist) * force * dt;
                    node.vy += (dy / dist) * force * dt;
                }
            });
        });
        
        // Edge attraction
        this.edges.forEach(edge => {
            const source = this.nodeMap.get(edge.source); // OPTIMIZED
            const target = this.nodeMap.get(edge.target); // OPTIMIZED
            if (!source || !target || !source.visible || !target.visible) return;
            
            const dx = target.x - source.x;
            const dy = target.y - source.y;
            const dist = Math.max(Math.sqrt(dx * dx + dy * dy), 1);
            const targetDist = 150;
            
            const force = (dist - targetDist) * this.physics.attraction;
            source.vx += (dx / dist) * force * dt;
            source.vy += (dy / dist) * force * dt;
            target.vx -= (dx / dist) * force * dt;
            target.vy -= (dy / dist) * force * dt;
        });
        
        // Apply velocity
        nodes.forEach(node => {
            if (!node.visible) return;
            
            node.vx *= this.physics.damping;
            node.vy *= this.physics.damping;
            
            // Clamp velocity
            const maxV = 8;
            const v = Math.sqrt(node.vx * node.vx + node.vy * node.vy);
            if (v > maxV) {
                node.vx = (node.vx / v) * maxV;
                node.vy = (node.vy / v) * maxV;
            }
            
            if (Math.abs(node.vx) > this.physics.minVelocity) node.x += node.vx;
            if (Math.abs(node.vy) > this.physics.minVelocity) node.y += node.vy;
            
            // Bounds
            const padding = this.options.padding;
            node.x = Math.max(padding, Math.min(this.width - padding, node.x));
            node.y = Math.max(padding, Math.min(this.height - padding, node.y));
        });
    }
    
    // Map screen coordinates to logical coordinates
    screenToWorld(sx, sy) {
        // Since canvas CSS size equals logical size, we just adjust for pan/zoom
        return {
            x: (sx - this.panX) / this.zoom,
            y: (sy - this.panY) / this.zoom
        };
    }
    
    findNodeAt(wx, wy) {
        for (let i = this.nodes.length - 1; i >= 0; i--) {
            const node = this.nodes[i];
            if (!node.visible) continue;
            const dx = wx - node.x;
            const dy = wy - node.y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist <= node.radius + 4) return node;
        }
        return null;
    }
    
    // --- Interaction Handlers ---

    handleMouseMove(e) {
        const rect = this.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        const world = this.screenToWorld(x, y);
        
        if (this.isDragging) {
            const dx = e.clientX - this.lastMouse.x;
            const dy = e.clientY - this.lastMouse.y;
            this.panX += dx;
            this.panY += dy;
            this.lastMouse.x = e.clientX;
            this.lastMouse.y = e.clientY;
        } else {
            const hovered = this.findNodeAt(world.x, world.y);
            if (hovered !== this.hoveredNode) {
                this.hoveredNode = hovered;
                this.canvas.style.cursor = hovered ? 'pointer' : 'grab';
                if (this.onHoverChange) this.onHoverChange(hovered, e);
            }
        }
    }
    
    handleTouchStart(e) {
        if (e.touches.length === 1) {
            e.preventDefault();
            const touch = e.touches[0];
            this.isDragging = true;
            this.lastMouse = { x: touch.clientX, y: touch.clientY };
            this.canvas.style.cursor = 'grabbing';
        }
    }

    handleTouchMove(e) {
        if (e.touches.length === 1 && this.isDragging) {
            e.preventDefault();
            const touch = e.touches[0];
            const dx = touch.clientX - this.lastMouse.x;
            const dy = touch.clientY - this.lastMouse.y;
            this.panX += dx;
            this.panY += dy;
            this.lastMouse.x = touch.clientX;
            this.lastMouse.y = touch.clientY;
        }
    }

    handleMouseDown(e) {
        this.isDragging = true;
        this.lastMouse = { x: e.clientX, y: e.clientY };
        this.canvas.style.cursor = 'grabbing';
    }
    
    handleMouseUp() {
        this.isDragging = false;
        this.canvas.style.cursor = this.hoveredNode ? 'pointer' : 'grab';
    }
    
    handleMouseLeave() {
        this.isDragging = false;
        this.hoveredNode = null;
        if (this.onHoverChange) this.onHoverChange(null);
    }
    
    handleWheel(e) {
        e.preventDefault();
        
        const rect = this.canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;
        
        const delta = e.deltaY > 0 ? 0.92 : 1.08;
        const newZoom = Math.max(0.3, Math.min(3, this.zoom * delta));
        
        // Zoom towards mouse position
        this.panX = mx - (mx - this.panX) * (newZoom / this.zoom);
        this.panY = my - (my - this.panY) * (newZoom / this.zoom);
        this.zoom = newZoom;
    }
    
    handleClick(e) {
        // Only select if not dragging (prevents selection on mouseup after drag)
        if (this.hoveredNode) {
            this.selectedNode = this.hoveredNode;
            if (this.onNodeSelect) this.onNodeSelect(this.hoveredNode.term);
        }
    }
    
    // --- Controls ---
    
    zoomIn() {
        const newZoom = Math.min(3, this.zoom * 1.2);
        // Zoom towards center
        this.panX = this.width/2 - (this.width/2 - this.panX) * (newZoom / this.zoom);
        this.panY = this.height/2 - (this.height/2 - this.panY) * (newZoom / this.zoom);
        this.zoom = newZoom;
    }
    
    zoomOut() {
        const newZoom = Math.max(0.3, this.zoom / 1.2);
        this.panX = this.width/2 - (this.width/2 - this.panX) * (newZoom / this.zoom);
        this.panY = this.height/2 - (this.height/2 - this.panY) * (newZoom / this.zoom);
        this.zoom = newZoom;
    }
    
    resetView() {
        this.zoom = 1;
        this.panX = 0;
        this.panY = 0;
    }
    
    highlightNodes(query) {
        const q = query ? query.toLowerCase() : '';
        this.nodes.forEach(node => {
            node.highlighted = q && (
                node.term.name.toLowerCase().includes(q) ||
                node.term.shortDesc.toLowerCase().includes(q) ||
                node.term.tags.some(t => t.toLowerCase().includes(q))
            );
        });
    }
    
    filterByCategory(categoryId) {
        this.nodes.forEach(node => {
            node.visible = !categoryId || node.term.category === categoryId;
        });
    }
    
    startAnimation() {
        const animate = () => {
            this.simulatePhysics();
            this.render();
            this.animationId = requestAnimationFrame(animate);
        };
        animate();
    }
    
    stopAnimation() {
        if (this.animationId) cancelAnimationFrame(this.animationId);
    }
    
    // --- Rendering ---

    render() {
        const ctx = this.ctx;
        
        // Clear logical area (accounting for DPR set in handleResize)
        ctx.save();
        ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0); // Ensure clean slate for DPR
        ctx.fillStyle = '#fafbfc';
        ctx.fillRect(0, 0, this.width, this.height);
        
        // Grid
        ctx.strokeStyle = 'rgba(0,0,0,0.025)';
        ctx.lineWidth = 1;
        const gridSize = 30;
        for (let x = 0; x < this.width; x += gridSize) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, this.height);
            ctx.stroke();
        }
        for (let y = 0; y < this.height; y += gridSize) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(this.width, y);
            ctx.stroke();
        }
        
        // Apply Pan & Zoom
        ctx.translate(this.panX, this.panY);
        ctx.scale(this.zoom, this.zoom);
        
        // Edges
        this.edges.forEach(edge => {
            const source = this.nodeMap.get(edge.source);
            const target = this.nodeMap.get(edge.target);
            if (!source || !target || !source.visible || !target.visible) return;
            
            const isHighlighted = this.selectedNode && 
                (this.selectedNode.id === source.id || this.selectedNode.id === target.id);
            
            ctx.beginPath();
            ctx.moveTo(source.x, source.y);
            ctx.lineTo(target.x, target.y);
            ctx.strokeStyle = isHighlighted ? 'rgba(8,145,178,0.4)' : 'rgba(0,0,0,0.06)';
            ctx.lineWidth = isHighlighted ? 1.5 : 0.8;
            ctx.stroke();
        });
        
        // Nodes
        this.nodes.forEach(node => {
            if (!node.visible) return;
            
            const isSelected = this.selectedNode && this.selectedNode.id === node.id;
            const isHovered = this.hoveredNode && this.hoveredNode.id === node.id;
            const isRelated = this.selectedNode && node.term.related && 
                node.term.related.includes(this.selectedNode.id);
            const r = Math.max(1, node.radius);
            
            // Glow
            if (isSelected || isHovered) {
                const gr = Math.max(1, r + 8);
                const grad = ctx.createRadialGradient(node.x, node.y, 0, node.x, node.y, gr);
                grad.addColorStop(0, node.color + '30');
                grad.addColorStop(1, node.color + '00');
                ctx.fillStyle = grad;
                ctx.beginPath();
                ctx.arc(node.x, node.y, gr, 0, Math.PI * 2);
                ctx.fill();
            }
            
            // Circle
            ctx.beginPath();
            ctx.arc(node.x, node.y, r, 0, Math.PI * 2);
            
            if (isSelected) {
                ctx.fillStyle = node.color;
                ctx.strokeStyle = '#fff';
                ctx.lineWidth = 2.5;
            } else if (isHovered || isRelated || node.highlighted) {
                ctx.fillStyle = node.color + 'cc';
                ctx.strokeStyle = node.color;
                ctx.lineWidth = 2;
            } else {
                ctx.fillStyle = '#fff';
                ctx.strokeStyle = node.color + '60';
                ctx.lineWidth = 1.5;
            }
            ctx.fill();
            ctx.stroke();
            
            // Label
            ctx.font = `500 ${this.options.fontSize}px 'Plus Jakarta Sans', sans-serif`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillStyle = isSelected ? '#fff' : '#374151';
            
            let label = node.term.name;
            if (label.length > 9) label = label.slice(0, 7) + '..';
            ctx.fillText(label, node.x, node.y);
        });
        
        ctx.restore();
    }
}

window.KnowledgeGraph = KnowledgeGraph;
