/**
 * Knowledge Graph Visualization
 * Handles canvas rendering and physics simulation
 */

class KnowledgeGraph {
    constructor(canvasId, options = {}) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) {
            console.error('Canvas not found:', canvasId);
            return;
        }
        
        this.ctx = this.canvas.getContext('2d');
        this.options = {
            nodeRadius: { core: 24, technique: 18, infrastructure: 16, application: 14 },
            edgeWidth: 1,
            fontSize: 11,
            padding: 80,
            ...options
        };
        
        // State
        this.nodes = [];
        this.edges = [];
        this.zoom = 1;
        this.panX = 0;
        this.panY = 0;
        this.hoveredNode = null;
        this.selectedNode = null;
        this.isDragging = false;
        this.lastMouse = { x: 0, y: 0 };
        this.animationFrame = null;
        
        // Physics
        this.physics = {
            enabled: true,
            repulsion: 800,
            attraction: 0.008,
            centerGravity: 0.01,
            damping: 0.85,
            minVelocity: 0.01
        };
        
        // Bind methods
        this.handleResize = this.handleResize.bind(this);
        this.handleMouseMove = this.handleMouseMove.bind(this);
        this.handleMouseDown = this.handleMouseDown.bind(this);
        this.handleMouseUp = this.handleMouseUp.bind(this);
        this.handleWheel = this.handleWheel.bind(this);
        this.handleClick = this.handleClick.bind(this);
        this.handleMouseLeave = this.handleMouseLeave.bind(this);
        
        this.init();
    }
    
    init() {
        this.handleResize();
        this.setupEventListeners();
        this.startAnimation();
    }
    
    setupEventListeners() {
        window.addEventListener('resize', this.handleResize);
        
        this.canvas.addEventListener('mousemove', this.handleMouseMove);
        this.canvas.addEventListener('mousedown', this.handleMouseDown);
        this.canvas.addEventListener('mouseup', this.handleMouseUp);
        this.canvas.addEventListener('wheel', this.handleWheel, { passive: false });
        this.canvas.addEventListener('click', this.handleClick);
        this.canvas.addEventListener('mouseleave', this.handleMouseLeave);
        
        // Touch support
        this.canvas.addEventListener('touchstart', (e) => {
            e.preventDefault();
            const touch = e.touches[0];
            this.handleMouseDown({ clientX: touch.clientX, clientY: touch.clientY });
        });
        this.canvas.addEventListener('touchmove', (e) => {
            e.preventDefault();
            const touch = e.touches[0];
            this.handleMouseMove({ clientX: touch.clientX, clientY: touch.clientY });
        });
        this.canvas.addEventListener('touchend', this.handleMouseUp);
    }
    
    handleResize() {
        const rect = this.canvas.parentElement.getBoundingClientRect();
        this.canvas.width = rect.width;
        this.canvas.height = rect.height;
        this.centerX = this.canvas.width / 2;
        this.centerY = this.canvas.height / 2;
    }
    
    // Load data from KnowledgeBase
    loadData() {
        if (!window.KnowledgeBase) return;
        
        this.nodes = [];
        this.edges = [];
        
        const categories = window.KnowledgeBase.categories;
        const terms = window.KnowledgeBase.terms;
        
        // Create nodes with initial positions
        terms.forEach((term, index) => {
            const category = categories.find(c => c.id === term.category);
            const categoryIndex = categories.indexOf(category);
            const totalCategories = categories.length;
            
            // Position in category cluster
            const baseAngle = (categoryIndex / totalCategories) * Math.PI * 2 - Math.PI / 2;
            const categoryTerms = terms.filter(t => t.category === term.category);
            const termIndex = categoryTerms.indexOf(term);
            
            // Determine radius based on type
            let radius = this.options.nodeRadius[term.type] || 16;
            let distance = 200 + (term.type === 'core' ? 0 : 60);
            
            // Spread within category
            const spreadAngle = Math.PI / 4;
            const angleOffset = (termIndex - categoryTerms.length / 2) * (spreadAngle / Math.max(1, categoryTerms.length));
            const angle = baseAngle + angleOffset;
            
            // Add some randomness
            const jitterX = (Math.random() - 0.5) * 40;
            const jitterY = (Math.random() - 0.5) * 40;
            
            this.nodes.push({
                id: term.id,
                x: this.centerX + Math.cos(angle) * distance + jitterX,
                y: this.centerY + Math.sin(angle) * distance + jitterY,
                vx: 0,
                vy: 0,
                radius: radius,
                term: term,
                color: category ? category.color : '#6b7280',
                highlighted: false
            });
        });
        
        // Create edges from relationships
        terms.forEach(term => {
            if (term.related) {
                term.related.forEach(relatedId => {
                    // Avoid duplicate edges
                    const exists = this.edges.some(e => 
                        (e.source === term.id && e.target === relatedId) ||
                        (e.target === term.id && e.source === relatedId)
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
        
        // Run physics to settle
        for (let i = 0; i < 100; i++) {
            this.simulatePhysics(0.1);
        }
    }
    
    simulatePhysics(dt = 1) {
        if (!this.physics.enabled) return;
        
        const nodes = this.nodes;
        const centerX = this.centerX;
        const centerY = this.centerY;
        
        // Apply forces
        nodes.forEach(node => {
            // Center gravity
            node.vx += (centerX - node.x) * this.physics.centerGravity * dt;
            node.vy += (centerY - node.y) * this.physics.centerGravity * dt;
            
            // Repulsion from other nodes
            nodes.forEach(other => {
                if (node.id === other.id) return;
                
                const dx = node.x - other.x;
                const dy = node.y - other.y;
                const dist = Math.max(1, Math.sqrt(dx * dx + dy * dy));
                const minDist = node.radius + other.radius + 20;
                
                if (dist < minDist * 3) {
                    const force = this.physics.repulsion / (dist * dist);
                    node.vx += (dx / dist) * force * dt;
                    node.vy += (dy / dist) * force * dt;
                }
            });
        });
        
        // Edge attraction
        this.edges.forEach(edge => {
            const source = this.findNode(edge.source);
            const target = this.findNode(edge.target);
            if (!source || !target) return;
            
            const dx = target.x - source.x;
            const dy = target.y - source.y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            const targetDist = 180;
            
            const force = (dist - targetDist) * this.physics.attraction;
            source.vx += (dx / dist) * force * dt;
            source.vy += (dy / dist) * force * dt;
            target.vx -= (dx / dist) * force * dt;
            target.vy -= (dy / dist) * force * dt;
        });
        
        // Apply velocity
        nodes.forEach(node => {
            node.vx *= this.physics.damping;
            node.vy *= this.physics.damping;
            
            // Clamp velocity
            const maxV = 10;
            const v = Math.sqrt(node.vx * node.vx + node.vy * node.vy);
            if (v > maxV) {
                node.vx = (node.vx / v) * maxV;
                node.vy = (node.vy / v) * maxV;
            }
            
            // Apply if above threshold
            if (Math.abs(node.vx) > this.physics.minVelocity) {
                node.x += node.vx;
            }
            if (Math.abs(node.vy) > this.physics.minVelocity) {
                node.y += node.vy;
            }
            
            // Keep in bounds
            const padding = this.options.padding;
            node.x = Math.max(padding, Math.min(this.canvas.width - padding, node.x));
            node.y = Math.max(padding, Math.min(this.canvas.height - padding, node.y));
        });
    }
    
    findNode(id) {
        return this.nodes.find(n => n.id === id);
    }
    
    // Convert screen coords to world coords
    screenToWorld(screenX, screenY) {
        const rect = this.canvas.getBoundingClientRect();
        return {
            x: (screenX - rect.left - this.panX) / this.zoom,
            y: (screenY - rect.top - this.panY) / this.zoom
        };
    }
    
    // Find node at position
    findNodeAt(worldX, worldY) {
        for (let i = this.nodes.length - 1; i >= 0; i--) {
            const node = this.nodes[i];
            const dx = worldX - node.x;
            const dy = worldY - node.y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist <= node.radius + 5) {
                return node;
            }
        }
        return null;
    }
    
    handleMouseMove(e) {
        const world = this.screenToWorld(e.clientX, e.clientY);
        
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
                this.onHoverChange && this.onHoverChange(hovered, e);
            }
        }
    }
    
    handleMouseDown(e) {
        this.isDragging = true;
        this.lastMouse.x = e.clientX;
        this.lastMouse.y = e.clientY;
        this.canvas.style.cursor = 'grabbing';
    }
    
    handleMouseUp() {
        this.isDragging = false;
        this.canvas.style.cursor = this.hoveredNode ? 'pointer' : 'grab';
    }
    
    handleWheel(e) {
        e.preventDefault();
        const delta = e.deltaY > 0 ? 0.9 : 1.1;
        const newZoom = Math.max(0.3, Math.min(3, this.zoom * delta));
        
        // Zoom towards mouse position
        const rect = this.canvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;
        
        this.panX = mouseX - (mouseX - this.panX) * (newZoom / this.zoom);
        this.panY = mouseY - (mouseY - this.panY) * (newZoom / this.zoom);
        this.zoom = newZoom;
    }
    
    handleClick(e) {
        if (this.hoveredNode) {
            this.selectedNode = this.hoveredNode;
            this.onNodeSelect && this.onNodeSelect(this.hoveredNode.term);
        }
    }
    
    handleMouseLeave() {
        this.isDragging = false;
        this.hoveredNode = null;
        this.onHoverChange && this.onHoverChange(null);
    }
    
    // Zoom controls
    zoomIn() {
        this.zoom = Math.min(3, this.zoom * 1.2);
    }
    
    zoomOut() {
        this.zoom = Math.max(0.3, this.zoom / 1.2);
    }
    
    resetView() {
        this.zoom = 1;
        this.panX = 0;
        this.panY = 0;
    }
    
    // Highlight nodes by search
    highlightNodes(query) {
        const q = query.toLowerCase();
        this.nodes.forEach(node => {
            node.highlighted = query && (
                node.term.name.toLowerCase().includes(q) ||
                node.term.shortDesc.toLowerCase().includes(q) ||
                node.term.tags.some(t => t.toLowerCase().includes(q))
            );
        });
    }
    
    // Filter by category
    filterByCategory(categoryId) {
        this.nodes.forEach(node => {
            node.visible = !categoryId || node.term.category === categoryId;
        });
    }
    
    // Animation loop
    startAnimation() {
        const animate = () => {
            this.simulatePhysics();
            this.render();
            this.animationFrame = requestAnimationFrame(animate);
        };
        animate();
    }
    
    stopAnimation() {
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
        }
    }
    
    // Render
    render() {
        const ctx = this.ctx;
        const w = this.canvas.width;
        const h = this.canvas.height;
        
        // Clear
        ctx.fillStyle = '#fafbfc';
        ctx.fillRect(0, 0, w, h);
        
        // Draw subtle grid
        this.drawGrid(ctx, w, h);
        
        // Apply transform
        ctx.save();
        ctx.translate(this.panX, this.panY);
        ctx.scale(this.zoom, this.zoom);
        
        // Draw edges
        this.edges.forEach(edge => {
            const source = this.findNode(edge.source);
            const target = this.findNode(edge.target);
            if (!source || !target) return;
            if (source.visible === false || target.visible === false) return;
            
            const isHighlighted = this.selectedNode && 
                (this.selectedNode.id === source.id || this.selectedNode.id === target.id);
            
            ctx.beginPath();
            ctx.moveTo(source.x, source.y);
            ctx.lineTo(target.x, target.y);
            
            if (isHighlighted) {
                ctx.strokeStyle = 'rgba(8, 145, 178, 0.5)';
                ctx.lineWidth = 2;
            } else {
                ctx.strokeStyle = 'rgba(0, 0, 0, 0.08)';
                ctx.lineWidth = 1;
            }
            ctx.stroke();
        });
        
        // Draw nodes
        this.nodes.forEach(node => {
            if (node.visible === false) return;
            
            const isSelected = this.selectedNode && this.selectedNode.id === node.id;
            const isHovered = this.hoveredNode && this.hoveredNode.id === node.id;
            const isRelated = this.selectedNode && node.term.related?.includes(this.selectedNode.id);
            const radius = Math.max(1, node.radius);
            
            // Glow for selected/hovered
            if (isSelected || isHovered) {
                const glowRadius = Math.max(1, radius + 10);
                const gradient = ctx.createRadialGradient(
                    node.x, node.y, 0,
                    node.x, node.y, glowRadius
                );
                gradient.addColorStop(0, node.color + '40');
                gradient.addColorStop(1, node.color + '00');
                ctx.fillStyle = gradient;
                ctx.beginPath();
                ctx.arc(node.x, node.y, glowRadius, 0, Math.PI * 2);
                ctx.fill();
            }
            
            // Node circle
            ctx.beginPath();
            ctx.arc(node.x, node.y, radius, 0, Math.PI * 2);
            
            if (isSelected) {
                ctx.fillStyle = node.color;
                ctx.strokeStyle = '#ffffff';
                ctx.lineWidth = 3;
            } else if (isHovered || isRelated) {
                ctx.fillStyle = node.color + 'dd';
                ctx.strokeStyle = node.color;
                ctx.lineWidth = 2;
            } else if (node.highlighted) {
                ctx.fillStyle = node.color + 'cc';
                ctx.strokeStyle = node.color;
                ctx.lineWidth = 2;
            } else {
                ctx.fillStyle = '#ffffff';
                ctx.strokeStyle = node.color + '80';
                ctx.lineWidth = 1.5;
            }
            
            ctx.fill();
            ctx.stroke();
            
            // Label
            ctx.font = `${isSelected || isHovered ? '600' : '500'} ${this.options.fontSize}px 'Plus Jakarta Sans', sans-serif`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            
            // Text color based on background
            ctx.fillStyle = isSelected ? '#ffffff' : '#374151';
            
            // Truncate long names
            let label = node.term.name;
            if (label.length > 10) {
                label = label.substring(0, 8) + '..';
            }
            ctx.fillText(label, node.x, node.y);
        });
        
        ctx.restore();
    }
    
    drawGrid(ctx, w, h) {
        const gridSize = 40;
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.03)';
        ctx.lineWidth = 1;
        
        for (let x = 0; x < w; x += gridSize) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, h);
            ctx.stroke();
        }
        
        for (let y = 0; y < h; y += gridSize) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(w, y);
            ctx.stroke();
        }
    }
    
    // Callbacks
    onNodeSelect = null;
    onHoverChange = null;
}

// Export
window.KnowledgeGraph = KnowledgeGraph;
