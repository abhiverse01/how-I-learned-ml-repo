/**
 * AI Knowledge Graph Visualization - GOD MODE EDITION
 * Enhanced rendering, physics, and interaction.
 */

class KnowledgeGraph {
    constructor(canvasId, options = {}) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) {
            console.error('Canvas not found:', canvasId);
            return;
        }
        
        this.ctx = this.canvas.getContext('2d');
        
        // Visual Theme
        this.theme = {
            bg: '#f8fafc', // Base background
            bgGradientCenter: '#f1f5f9',
            gridDot: '#e2e8f0',
            edge: 'rgba(148, 163, 184, 0.25)',
            edgeHighlight: 'rgba(6, 182, 212, 0.6)',
            text: '#334155',
            textHighlight: '#ffffff',
            shadow: 'rgba(0, 0, 0, 0.15)'
        };

        // Options
        this.options = {
            nodeRadius: { core: 26, technique: 18, infrastructure: 14, application: 12 },
            fontSize: 10,
            padding: 80,
            ...options
        };
        
        // State
        this.nodes = [];
        this.edges = [];
        this.nodeMap = new Map();
        this.zoom = 1;
        this.panX = 0;
        this.panY = 0;
        this.hoveredNode = null;
        this.selectedNode = null;
        this.isDragging = false;
        this.lastMouse = { x: 0, y: 0 };
        this.animationId = null;
        
        // Logical dimensions
        this.width = 0;
        this.height = 0;
        this.centerX = 0;
        this.centerY = 0;
        this.dpr = window.devicePixelRatio || 1;
        
        // Time tracker for animations
        this.time = 0;
        
        // Physics config (Smoother constants)
        this.physics = {
            enabled: true,
            repulsion: 800,
            attraction: 0.005,
            centerGravity: 0.01,
            damping: 0.85,
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
        
        // Touch Events
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
        
        // HiDPI Support
        this.dpr = window.devicePixelRatio || 1;
        this.canvas.width = this.width * this.dpr;
        this.canvas.height = this.height * this.dpr;
        this.canvas.style.width = `${this.width}px`;
        this.canvas.style.height = `${this.height}px`;
        
        this.ctx.scale(this.dpr, this.dpr);
    }
    
    loadData() {
        if (!window.KnowledgeBase) return;
        
        this.nodes = [];
        this.edges = [];
        this.nodeMap.clear();
        
        const categories = KnowledgeBase.categories;
        const terms = KnowledgeBase.terms;
        
        if (!categories.length || !terms.length) return;
        
        // Create nodes
        terms.forEach((term) => {
            const category = categories.find(c => c.id === term.category);
            const categoryIndex = categories.indexOf(category);
            const totalCategories = categories.length || 1;
            
            // Circular layout with jitter
            const baseAngle = (categoryIndex / totalCategories) * Math.PI * 2 - Math.PI / 2;
            const categoryTerms = terms.filter(t => t.category === term.category);
            const termIndex = categoryTerms.indexOf(term);
            
            const radius = this.options.nodeRadius[term.type] || 16;
            let distance = 200;
            if (term.type === 'core') distance = 160;
            else if (term.type === 'technique') distance = 220;
            else distance = 280;
            
            const spreadAngle = Math.PI / 4;
            const angleOffset = categoryTerms.length > 1 
                ? (termIndex - (categoryTerms.length - 1) / 2) * (spreadAngle / categoryTerms.length)
                : 0;
            const angle = baseAngle + angleOffset;
            
            const jitterX = (Math.random() - 0.5) * 40;
            const jitterY = (Math.random() - 0.5) * 40;
            
            const node = {
                id: term.id,
                x: this.centerX + Math.cos(angle) * distance + jitterX,
                y: this.centerY + Math.sin(angle) * distance + jitterY,
                vx: 0, vy: 0,
                radius: radius,
                term: term,
                color: category ? category.color : '#94a3b8',
                highlighted: false,
                visible: true,
                // Animation state
                currentRadius: radius, 
                targetRadius: radius
            };
            
            this.nodes.push(node);
            this.nodeMap.set(term.id, node);
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
        
        // Apply forces
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
                
                // Stronger repulsion at close range
                if (dist < 200) {
                    const force = this.physics.repulsion / (distSq + eps);
                    node.vx += (dx / dist) * force * dt;
                    node.vy += (dy / dist) * force * dt;
                }
            });
        });
        
        // Edge attraction
        this.edges.forEach(edge => {
            const source = this.nodeMap.get(edge.source);
            const target = this.nodeMap.get(edge.target);
            if (!source || !target || !source.visible || !target.visible) return;
            
            const dx = target.x - source.x;
            const dy = target.y - source.y;
            const dist = Math.max(Math.sqrt(dx * dx + dy * dy), 1);
            const targetDist = 180; // Ideal spring length
            
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
            
            const maxV = 10;
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
            
            // Animate radius (for selection pulse)
            node.currentRadius += (node.targetRadius - node.currentRadius) * 0.1;
        });
    }
    
    // --- Interaction Logic (Unchanged) ---
    
    screenToWorld(sx, sy) {
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
            if (dist <= node.radius + 5) return node;
        }
        return null;
    }
    
    handleMouseMove(e) {
        const rect = this.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        if (this.isDragging) {
            const dx = e.clientX - this.lastMouse.x;
            const dy = e.clientY - this.lastMouse.y;
            this.panX += dx;
            this.panY += dy;
            this.lastMouse.x = e.clientX;
            this.lastMouse.y = e.clientY;
        } else {
            const world = this.screenToWorld(x, y);
            const hovered = this.findNodeAt(world.x, world.y);
            if (hovered !== this.hoveredNode) {
                this.hoveredNode = hovered;
                this.canvas.style.cursor = hovered ? 'pointer' : 'grab';
                if (this.onHoverChange) this.onHoverChange(hovered, e);
            }
        }
    }
    
    handleMouseDown(e) {
        this.isDragging = true;
        this.lastMouse = { x: e.clientX, y: e.clientY };
        this.canvas.style.cursor = 'grabbing';
    }
    
    handleTouchStart(e) {
        if (e.touches.length === 1) {
            e.preventDefault();
            const touch = e.touches[0];
            this.isDragging = true;
            this.lastMouse = { x: touch.clientX, y: touch.clientY };
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
        
        const delta = e.deltaY > 0 ? 0.9 : 1.1;
        const newZoom = Math.max(0.2, Math.min(4, this.zoom * delta));
        
        this.panX = mx - (mx - this.panX) * (newZoom / this.zoom);
        this.panY = my - (my - this.panY) * (newZoom / this.zoom);
        this.zoom = newZoom;
    }
    
    handleClick(e) {
        if (this.hoveredNode) {
            this.selectedNode = this.hoveredNode;
            // Trigger radius pulse
            this.selectedNode.targetRadius = this.selectedNode.radius * 1.2;
            setTimeout(() => {
                if(this.selectedNode) this.selectedNode.targetRadius = this.selectedNode.radius;
            }, 200);
            
            if (this.onNodeSelect) this.onNodeSelect(this.hoveredNode.term);
        }
    }
    
    zoomIn() {
        const newZoom = Math.min(4, this.zoom * 1.3);
        this.panX = this.width/2 - (this.width/2 - this.panX) * (newZoom / this.zoom);
        this.panY = this.height/2 - (this.height/2 - this.panY) * (newZoom / this.zoom);
        this.zoom = newZoom;
    }
    
    zoomOut() {
        const newZoom = Math.max(0.2, this.zoom / 1.3);
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
            this.time += 0.016; // Approx 60fps delta
            this.simulatePhysics();
            this.render();
            this.animationId = requestAnimationFrame(animate);
        };
        animate();
    }
    
    stopAnimation() {
        if (this.animationId) cancelAnimationFrame(this.animationId);
    }
    
    // --- GOD MODE RENDERING ---

    render() {
        const ctx = this.ctx;
        
        // 1. Clear & Background
        ctx.save();
        ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
        
        // Gradient Background
        const bgGrad = ctx.createRadialGradient(
            this.centerX, this.centerY, 0, 
            this.centerX, this.centerY, Math.max(this.width, this.height) * 0.7
        );
        bgGrad.addColorStop(0, '#ffffff');
        bgGrad.addColorStop(1, this.theme.bg);
        ctx.fillStyle = bgGrad;
        ctx.fillRect(0, 0, this.width, this.height);
        
        // Dot Grid Pattern
        this.drawDotGrid(ctx);
        
        // 2. World Transformations (Pan/Zoom)
        ctx.translate(this.panX, this.panY);
        ctx.scale(this.zoom, this.zoom);
        
        // 3. Edges (Draw first, under nodes)
        this.drawEdges(ctx);
        
        // 4. Nodes
        this.drawNodes(ctx);
        
        ctx.restore();
    }

    drawDotGrid(ctx) {
        ctx.fillStyle = this.theme.gridDot;
        const gap = 40;
        const radius = 1.5;
        
        // Offset grid by pan/zoom to create infinite feel
        // But for performance in this specific app, static is fine, or we can just skip moving it.
        // Let's keep it static for cleaner look during pan.
        
        for (let x = gap; x < this.width; x += gap) {
            for (let y = gap; y < this.height; y += gap) {
                ctx.beginPath();
                ctx.arc(x, y, radius, 0, Math.PI * 2);
                ctx.fill();
            }
        }
    }

    drawEdges(ctx) {
        this.edges.forEach(edge => {
            const source = this.nodeMap.get(edge.source);
            const target = this.nodeMap.get(edge.target);
            
            if (!source || !target || !source.visible || !target.visible) return;
            
            const isSelected = this.selectedNode && 
                (this.selectedNode.id === source.id || this.selectedNode.id === target.id);
            
            // Curved Line Calculation
            const dx = target.x - source.x;
            const dy = target.y - source.y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            
            // Midpoint offset for curve
            const mx = (source.x + target.x) / 2;
            const my = (source.y + target.y) / 2;
            
            // Perpendicular offset
            const offset = dist * 0.1; // Curve intensity
            
            // Alternate curve direction for overlapping edges if needed, 
            // but simple random based on id is consistent
            const direction = (source.id < target.id) ? 1 : -1;
            const cx = mx + (dy / dist) * offset * direction;
            const cy = my - (dx / dist) * offset * direction;

            ctx.beginPath();
            ctx.moveTo(source.x, source.y);
            ctx.quadraticCurveTo(cx, cy, target.x, target.y);
            
            if (isSelected) {
                ctx.strokeStyle = this.theme.edgeHighlight;
                ctx.lineWidth = 2.5;
                ctx.shadowColor = 'rgba(6, 182, 212, 0.5)';
                ctx.shadowBlur = 8;
            } else {
                ctx.strokeStyle = this.theme.edge;
                ctx.lineWidth = 1.2;
                ctx.shadowBlur = 0; // Clear shadow
            }
            
            ctx.stroke();
            ctx.shadowBlur = 0; // Reset shadow
        });
    }

    drawNodes(ctx) {
        // Sort nodes so selected/hovered are drawn on top
        const sortedNodes = [...this.nodes].sort((a, b) => {
            if (a.id === this.selectedNode?.id) return 1;
            if (b.id === this.selectedNode?.id) return -1;
            return 0;
        });

        sortedNodes.forEach(node => {
            if (!node.visible) return;
            
            const isSelected = this.selectedNode && this.selectedNode.id === node.id;
            const isHovered = this.hoveredNode && this.hoveredNode.id === node.id;
            const isRelated = this.selectedNode && node.term.related && 
                node.term.related.includes(this.selectedNode.id);
            
            // Pulse animation for selected node
            let r = node.currentRadius;
            if (isSelected) {
                r += Math.sin(this.time * 5) * 2; // Gentle pulse
            }
            r = Math.max(1, r);
            
            // --- Shadow ---
            if (isSelected || isHovered) {
                ctx.shadowColor = node.color + '80';
                ctx.shadowBlur = 20;
                ctx.shadowOffsetX = 0;
                ctx.shadowOffsetY = 4;
            } else {
                ctx.shadowColor = 'rgba(0,0,0,0.1)';
                ctx.shadowBlur = 10;
                ctx.shadowOffsetX = 0;
                ctx.shadowOffsetY = 2;
            }
            
            // --- Node Body (Gradient) ---
            ctx.beginPath();
            ctx.arc(node.x, node.y, r, 0, Math.PI * 2);
            
            // Create a 3D sphere gradient look
            const grad = ctx.createRadialGradient(
                node.x - r*0.3, node.y - r*0.3, r * 0.1, // Highlight spot
                node.x, node.y, r
            );
            
            if (isSelected || isHovered || node.highlighted) {
                grad.addColorStop(0, '#ffffff');
                grad.addColorStop(0.3, node.color);
                grad.addColorStop(1, this.darkenColor(node.color, 30));
                ctx.fillStyle = grad;
            } else {
                grad.addColorStop(0, '#ffffff');
                grad.addColorStop(0.5, '#ffffff');
                grad.addColorStop(1, '#f1f5f9');
                ctx.fillStyle = grad;
            }
            
            ctx.fill();
            
            // --- Border ---
            ctx.shadowBlur = 0; // Don't apply shadow to border
            ctx.strokeStyle = (isSelected || isHovered || node.highlighted) ? node.color : '#cbd5e1';
            ctx.lineWidth = (isSelected || isHovered) ? 3 : 2;
            ctx.stroke();
            
            // --- Label ---
            ctx.font = `600 ${this.options.fontSize}px 'Plus Jakarta Sans', sans-serif`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            
            // Text Shadow for readability
            ctx.fillStyle = 'rgba(255,255,255,0.8)'; // Fake shadow
            ctx.fillText(node.term.name, node.x + 0.5, node.y + 0.5);
            
            // Actual Text
            ctx.fillStyle = (isSelected || isHovered) ? node.color : '#475569';
            if (isSelected) ctx.fillStyle = '#ffffff'; // White text on solid color
            
            let label = node.term.name;
            if (label.length > 8) label = label.slice(0, 6) + '..';
            ctx.fillText(label, node.x, node.y);
        });
    }

    // Utility to darken colors for gradient depth
    darkenColor(hex, percent) {
        const num = parseInt(hex.replace('#', ''), 16);
        const amt = Math.round(2.55 * percent);
        const R = (num >> 16) - amt;
        const G = (num >> 8 & 0x00FF) - amt;
        const B = (num & 0x0000FF) - amt;
        return '#' + (0x1000000 + 
            (R < 0 ? 0 : R > 255 ? 255 : R) * 0x10000 + 
            (G < 0 ? 0 : G > 255 ? 255 : G) * 0x100 + 
            (B < 0 ? 0 : B > 255 ? 255 : B)
        ).toString(16).slice(1);
    }
}

window.KnowledgeGraph = KnowledgeGraph;
