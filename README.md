# 🧠 TitanML: AI Knowledge Nexus

**An interactive, physics-based knowledge graph for visualising and exploring the complex relationships in Artificial Intelligence.**

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Author](https://img.shields.io/badge/created%20by-Abhishek%20Shah-purple.svg)

<p align="center">
  <img src="https://img.shields.io/badge/Vanilla_JS-✨-yellow" alt="Vanilla JS">
  <img src="https://img.shields.io/badge/Canvas_API-⚡-orange" alt="Canvas API">
  <img src="https://img.shields.io/badge/Physics_Engine-🚀-red" alt="Physics Engine">
</p>

---

### 🚀 Overview
TitanML transforms static documentation into a living, breathing ecosystem. Built with **zero dependencies** (pure HTML5, CSS3, Vanilla JS), it features a custom physics engine, high-DPI canvas rendering, and a modular architecture designed for infinite extensibility.

### ✨ Key Features
*   **Force-Directed Graph**: Real-time physics simulation for organic node placement.
*   **God Mode Visualisation**: High-performance Canvas rendering with directional arrows, category gradients, and flow animations.
*   **Extensible Data Structure**: easily add new AI terms, categories, and relationships via a simple JSON-like object.
*   **Deep Linking**: Share direct links to specific concepts (e.g., `#rag` or `#transformers`).
*   **Navigation History**: Full browser history support (Back/Forward) within the single-page app.
*   **Responsive & Accessible**: Glassmorphism UI, mobile-friendly, and keyboard-navigable.

### 📦 Project Structure
The project uses a clean separation of concerns for easy maintenance:

```text
ai-knowledge-nexus/
├── index.html          # Core UI Shell & Loader
├── css/
│   └── styles.css      # Design System & Layout
└── js/
    ├── data.js         # Extensible Knowledge Base
    ├── graph.js        # Canvas Engine & Physics
    └── app.js          # State Management & UI Logic
```

### ⚡ Quick Start

No build tools or installations required.

1.  **Clone the repo**
    ```bash
    git clone https://github.com/abhiverse01/titanML.git
    ```
2.  **Open `index.html`**
    Simply double-click `index.html` or serve it locally.
3.  **Explore**
    Click nodes, drag the graph, and press `/` to search.

### 🛠 Tech Stack
*   **Engine**: Vanilla JavaScript (ES6+)
*   **Rendering**: HTML5 Canvas (HiDPI/Retina optimised)
*   **Styling**: CSS Variables & Flexbox/Grid
*   **Fonts**: Plus Jakarta Sans

### 📝 Adding New Terms
Expand the knowledge base by editing `js/data.js`. No other code changes needed.

```javascript
// Example: Adding a new concept
{
    id: 'quantum-ml',
    name: 'Quantum ML',
    category: 'training',
    definition: '...',
    related: ['transformers', 'llm']
}
```

### 👤 Creator

**Abhishek Shah**
*   **Portfolio**: [abhiverse01.github.io](https://abhiverse01.github.io)
*   **Email**: abhishek.aimarine@gmail.com
*   **LinkedIn**: [Connect with me](https://www.linkedin.com/in/theabhishekshah/)

---

*Built with precision and passion.*


