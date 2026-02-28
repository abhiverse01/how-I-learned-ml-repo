/**
 * AI Knowledge Base - Extensible Data Structure
 * GOD MODE ENHANCED: Fixed syntax errors and optimized definitions.
 */

// ==========================================
// HELPER FUNCTION
// ==========================================
function createTerm(config) {
    return {
        id: config.id || '',
        name: config.name || '',
        fullName: config.fullName || config.name || '',
        category: config.category || 'general',
        type: config.type || 'technique',
        shortDesc: config.shortDesc || '',
        definition: config.definition || '',
        related: config.related || [],
        tags: config.tags || [],
        codeExample: config.codeExample || '',
        createdAt: config.createdAt || new Date().toISOString(),
        importance: config.importance || 0
    };
}

// ==========================================
// CATEGORY DEFINITIONS
// ==========================================
const CategoryData = [
    { id: 'rag', name: 'RAG', fullName: 'Retrieval Augmented Generation', color: '#0891b2', description: 'Techniques for augmenting LLMs with external knowledge' },
    { id: 'agentic', name: 'Agentic AI', fullName: 'Autonomous Agent Systems', color: '#6366f1', description: 'Autonomous AI systems that plan and execute tasks' },
    { id: 'mcp', name: 'MCP', fullName: 'Model Context Protocol', color: '#8b5cf6', description: 'Protocol for connecting AI assistants to systems' },
    { id: 'architecture', name: 'Architecture', fullName: 'Model Architecture', color: '#f59e0b', description: 'Fundamental neural network architectures' },
    { id: 'training', name: 'Training', fullName: 'Training Methods', color: '#10b981', description: 'Methods for training and fine-tuning models' },
    { id: 'prompting', name: 'Prompting', fullName: 'Prompt Engineering', color: '#ec4899', description: 'Techniques for effective model interaction' },
    { id: 'infrastructure', name: 'Infrastructure', fullName: 'AI Infrastructure', color: '#ef4444', description: 'Systems and tools for AI deployment' },
    { id: 'applications', name: 'Applications', fullName: 'AI Applications', color: '#14b8a6', description: 'Real-world AI implementations' }
];

// ==========================================
// TERM DEFINITIONS
// ==========================================
const TermData = [
    // ========== RAG SECTION ==========
    {
        id: 'rag',
        name: 'RAG',
        fullName: 'Retrieval Augmented Generation',
        category: 'rag',
        type: 'core',
        shortDesc: 'Augmenting LLM responses with retrieved external knowledge',
        definition: `Retrieval Augmented Generation (RAG) is a hybrid architecture that combines the reasoning capabilities of Large Language Models (LLMs) with the factual accuracy of external knowledge bases. 

**Mechanism:**
1. **Retrieval**: A user query is converted into an embedding, and a vector store retrieves the top-k most similar document chunks.
2. **Augmentation**: The retrieved chunks are injected into the LLM prompt as context.
3. **Generation**: The LLM generates a response grounded in the provided context.

**Why it matters:** It solves the "hallucination" problem by grounding answers in facts, allows access to up-to-date information without retraining, and provides source attribution for generated text.

**Advanced Patterns:** 
- **Hybrid Search**: Combining vector search (semantic) with BM25 (keyword) for better recall.
- **Reranking**: Using a Cross-Encoder to re-order retrieved documents for precision.
- **HyDE (Hypothetical Document Embeddings)**: Generating a hypothetical answer to search for similar docs.`,
        related: ['vector-database', 'embeddings', 'hybrid-search', 'chunking', 'reranking'],
        tags: ['core', 'production', 'grounding'],
        codeExample: `# Production RAG Pipeline with Hybrid Search & Reranking
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain_community.retrievers import BM25Retriever

# 1. Prepare Retrievers
# Vector Store (Semantic)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# BM25 (Keyword/Lexical)
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 10

# 2. Hybrid Search (Ensemble)
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6]  # Tune weights based on domain
)

# 3. Reranking Step (Precision)
# Uses a Cross-Encoder to score query+doc pairs simultaneously
compressor = CohereRerank(model="rerank-english-v3.0")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=ensemble_retriever
)

# 4. Final Generation
docs = compression_retriever.invoke(query)
context_text = "\\n".join([d.page_content for d in docs])
prompt = f"Context: {context_text}\\nQuestion: {query}"
response = llm.invoke(prompt)`
    },
    {
        id: 'vector-database',
        name: 'Vector Database',
        category: 'rag',
        type: 'infrastructure',
        shortDesc: 'Specialized DB for high-dimensional similarity search',
        definition: `Vector Databases are optimized for storing, indexing, and querying high-dimensional vectors (embeddings). Unlike relational DBs optimized for exact matches, Vector DBs use **Approximate Nearest Neighbor (ANN)** algorithms to find "close enough" vectors efficiently.

**Core Algorithms:**
- **HNSW (Hierarchical Navigable Small World)**: Graph-based algorithm. Excellent query speed, high memory usage. Best for real-time search.
- **IVF (Inverted File Index)**: Clustering-based. Faster index build, lower memory, but slightly lower recall.
- **Product Quantization (PQ)**: Compresses vectors to save memory at the cost of accuracy.

**Key Metrics:** 
- **Recall**: Percentage of true nearest neighbors found.
- **QPS (Queries Per Second)**: Throughput.
- **Latency**: Time per query (P50, P99).`,
        related: ['embeddings', 'rag'],
        tags: ['infrastructure', 'storage', 'ann'],
        codeExample: `# Production Qdrant Setup with HNSW
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, HnswConfigDiff, OptimizersConfigDiff

client = QdrantClient(":memory:") # Use host/port for production

# Configure HNSW for high performance
hnsw_config = HnswConfigDiff(
    m=16,                # Number of edges per node (higher = better recall, more memory)
    ef_construct=100,    # Search depth during indexing
    full_scan_threshold=10000 # Threshold for brute force search
)

client.create_collection(
    collection_name="production_docs",
    vectors_config=VectorParams(
        size=1536,                 # OpenAI Embedding dimension
        distance=Distance.COSINE,  # Or Euclidean / Dot Product
        hnsw_config=hnsw_config
    ),
    optimizers_config=OptimizersConfigDiff(
        indexing_threshold=2000    # Start indexing after 2k points
    )
)`
    },
    {
        id: 'embeddings',
        name: 'Embeddings',
        category: 'rag',
        type: 'core',
        shortDesc: 'Dense vectors capturing semantic meaning',
        definition: `Embeddings are dense vector representations (lists of floating-point numbers) where the distance between vectors correlates with semantic similarity. They bridge unstructured text and mathematical operations.

**How they are trained:**
- **Contrastive Learning**: Models (like Sentence-BERT) are trained to pull similar sentences closer in vector space and push dissimilar ones apart using contrastive loss (e.g., InfoNCE).

**Trade-offs:**
- **Dimensions**: Higher dims (e.g., 1536) capture more nuance but cost more storage/compute. Lower dims (384) are faster but less precise.
- **Domain Specificity**: Generic models (OpenAI, BGE) vs Domain specific (BioBERT, LegalBERT).`,
        related: ['vector-database', 'rag'],
        tags: ['fundamental', 'semantic', 'vector'],
        codeExample: `# Comparing Embedding Models
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import numpy as np

# 1. Proprietary API (OpenAI)
client = OpenAI()
def get_openai_embedding(text):
    return client.embeddings.create(
        input=[text], model="text-embedding-3-small"
    ).data[0].embedding

# 2. Open Source Local (BGE)
local_model = SentenceTransformer('BAAI/bge-m3') # Supports 100+ languages
def get_local_embedding(text):
    return local_model.encode(text)

# 3. Similarity Calculation
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

vec1 = get_local_embedding("AI is transforming industries")
vec2 = get_local_embedding("Machine learning changes business")
print(f"Semantic Similarity: {cosine_similarity(vec1, vec2):.4f}")`
    },
    {
        id: 'hybrid-search',
        name: 'Hybrid Search',
        category: 'rag',
        type: 'technique',
        shortDesc: 'Combining vector and keyword search for robustness',
        definition: `Hybrid Search fuses two retrieval paradigms:
1. **Sparse Vectors (Keywords)**: BM25/TF-IDF. Great for exact matches (product codes, names). "Sparse" because most dimensions are zero.
2. **Dense Vectors (Embeddings)**: Capture semantic meaning. Great for concepts and synonyms.

**Fusion Algorithm:** 
**Reciprocal Rank Fusion (RRF)** is the standard. It combines ranked lists without needing normalized scores. Formula: RRF(d) = sum( 1 / (k + rank(d)) )`,
        related: ['reranking', 'semantic-search'],
        tags: ['retrieval', 'precision'],
        codeExample: `# Reciprocal Rank Fusion (RRF) Implementation
def reciprocal_rank_fusion(results_dict, k=60):
    fused_scores = {}
    
    for system, docs in results_dict.items():
        for rank, doc in enumerate(docs):
            doc_id = doc.metadata.get('id')
            if doc_id not in fused_scores:
                fused_scores[doc_id] = {"score": 0, "doc": doc}
            
            # RRF Formula
            fused_scores[doc_id]["score"] += 1 / (k + rank + 1)
    
    # Sort by fused score
    ranked = sorted(fused_scores.values(), key=lambda x: x["score"], reverse=True)
    return [item["doc"] for item in ranked[:5]]`
    },
    {
        id: 'graph-rag',
        name: 'GraphRAG',
        category: 'rag',
        type: 'technique',
        shortDesc: 'Leveraging knowledge graphs for multi-hop reasoning',
        definition: `GraphRAG uses Knowledge Graphs (KG) to structure information as entities and relationships, rather than just vector chunks. 

**Why it's needed:** Vector RAG struggles with "multi-hop" queries (e.g., "Who is the CEO of the company that acquired GitHub?"). 
- Vector RAG might retrieve chunks about "CEO" and "Acquisitions" separately.
- GraphRAG traverses: GitHub -> acquired_by -> Microsoft -> CEO -> Satya Nadella.

**Pipeline:**
1. Extract Entities/Relations via LLM.
2. Build Graph (Neo4j / NetworkX).
3. Traverse graph for context.`,
        related: ['rag', 'knowledge-graph'],
        tags: ['advanced', 'reasoning', 'graph'],
        codeExample: `# GraphRAG Concept (Entity Extraction + Traversal)
import networkx as nx

# 1. Entity Extraction (Conceptual)
entities = llm.invoke("Extract entities and relations from: 'Microsoft acquired GitHub.'")
# Output: (Microsoft, acquired, GitHub)

# 2. Build Graph
G = nx.DiGraph()
G.add_edge("Microsoft", "GitHub", relation="acquired")
G.add_edge("Satya Nadella", "Microsoft", relation="CEO")

# 3. Multi-hop Query
def get_context(graph, entity, depth=2):
    context_nodes = nx.single_source_shortest_path_length(graph, entity, cutoff=depth)
    return list(context_nodes.keys())

print(get_context(G, "GitHub")) # ['GitHub', 'Microsoft', 'Satya Nadella']`
    },
    {
        id: 'chunking',
        name: 'Chunking',
        category: 'rag',
        type: 'technique',
        shortDesc: 'Splitting documents for optimal retrieval',
        definition: `Chunking is the process of breaking large documents into smaller pieces for embedding. It is a critical preprocessing step.

**Strategies:**
1. **Fixed Size**: Simple, but cuts sentences in half.
2. **Recursive Character Splitter**: Tries to split on paragraphs, then sentences, then words.
3. **Semantic Chunking**: Splits based on embedding similarity changes (slower but smarter).
4. **Parent Document Retriever**: Indexes small chunks for search, but returns the larger parent document for context (best of both worlds).`,
        related: ['rag', 'embeddings'],
        tags: ['preprocessing', 'retrieval'],
        codeExample: `# Parent Document Retriever (Advanced Chunking)
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Small chunks for precise search
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
# Large chunks for comprehensive context
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,         # Indexes child chunks
    docstore=InMemoryStore(),       # Stores parent docs
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)
# When queried, it retrieves small chunks, maps them to parents, returns full parent docs.`
    },
    {
        id: 'reranking',
        name: 'Reranking',
        category: 'rag',
        type: 'technique',
        shortDesc: 'Re-scoring retrieved documents for precision',
        definition: `Reranking improves retrieval precision by using a heavier, more accurate model on the top-N results.

**Bi-Encoder vs Cross-Encoder:**
- **Bi-Encoder (Embeddings)**: Encodes query and doc separately. Fast, but less accurate.
- **Cross-Encoder**: Encodes (query, doc) together. Slower, but captures fine-grained interactions (e.g., negation, ordering).

**Pipeline:** 
Vector Search (Retrieve 100) -> Cross-Encoder (Rerank to Top 5).`,
        related: ['rag', 'hybrid-search'],
        tags: ['retrieval', 'precision'],
        codeExample: `# Cross-Encoder Reranking
from sentence_transformers import CrossEncoder

model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank(query, docs, top_k=5):
    pairs = [[query, doc.page_content] for doc in docs]
    scores = model.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in ranked[:top_k]]`
    },

    // ========== AGENTIC SECTION ==========
    {
        id: 'agentic-ai',
        name: 'Agentic AI',
        category: 'agentic',
        type: 'core',
        shortDesc: 'Autonomous systems pursuing goals iteratively',
        definition: `Agentic AI shifts LLMs from passive responders to active goal-seekers. An agent operates in a loop: **Observe -> Think -> Act -> Reflect**.

**Core Components:**
1. **Planner**: Decomposes goals into steps.
2. **Tool Executor**: Calls external APIs.
3. **Memory**: Stores past actions and observations.
4. **Critic**: Evaluates if the goal is met.

**Pattern - ReAct (Reasoning + Acting):** The model outputs a "Thought" before every "Action", ensuring reasoning traces are explicit.`,
        related: ['tool-use', 'planning', 'multi-agent'],
        tags: ['core', 'autonomous'],
        codeExample: `# ReAct Agent Loop Implementation
def react_agent(query, llm, tools, max_steps=10):
    history = ""
    for step in range(max_steps):
        prompt = f"""
        Question: {query}
        History: {history}
        Think step-by-step. Use tools if needed.
        Format: 
        Thought: [Your reasoning]
        Action: tool_name[input]
        """
        response = llm.invoke(prompt)
        history += f"\\n{response}"
        
        if "Action:" not in response:
            return response # Final Answer
        
        action = parse_action(response) # e.g., "search['AI news']"
        observation = execute_tool(action, tools)
        history += f"\\nObservation: {observation}"`
    },
    {
        id: 'tool-use',
        name: 'Tool Use',
        category: 'agentic',
        type: 'core',
        shortDesc: 'Extending LLMs with external functions',
        definition: `Tool Use allows an LLM to execute code by generating structured JSON arguments.

**Mechanism:**
1. **Schema Definition**: Define tools with names, descriptions, and JSON schemas for arguments.
2. **Binding**: The tools are "bound" to the LLM request.
3. **Parsing**: The LLM outputs a tool call (JSON).
4. **Execution**: The runtime executes the code.
5. **Loop Back**: The result is fed back to the LLM.`,
        related: ['agentic-ai', 'mcp'],
        tags: ['integration', 'actions'],
        codeExample: `# Structured Tool with Pydantic Schema
from pydantic import BaseModel, Field
from langchain.tools import tool

class CalculatorInput(BaseModel):
    a: int = Field(description="The first number")
    b: int = Field(description="The second number")

@tool("multiply", args_schema=CalculatorInput)
def multiply(a: int, b: int) -> int:
    """Multiplies two integers."""
    return a * b

llm_with_tools = llm.bind_tools([multiply])
response = llm_with_tools.invoke("What is 5 times 7?")
# Model outputs: tool_call(name='multiply', args={'a': 5, 'b': 7})`
    },
    {
        id: 'multi-agent',
        name: 'Multi-Agent Systems',
        category: 'agentic',
        type: 'technique',
        shortDesc: 'Specialized agents collaborating on tasks',
        definition: `Multi-Agent Systems coordinate multiple specialized agents to solve complex problems.

**Patterns:**
- **Hierarchical**: A "Manager" agent assigns tasks to "Worker" agents.
- **Sequential**: Output of Agent A is input to Agent B (Assembly line).
- **Joint Chat**: Agents share a conversation history (Round table).

**Why needed:** Specialization. A single agent cannot be an expert at coding, writing, and finance analysis simultaneously.`,
        related: ['agentic-ai', 'planning'],
        tags: ['collaboration', 'complex'],
        codeExample: `# CrewAI Hierarchical Process
from crewai import Crew, Agent, Task, Process

manager = Agent(role="Manager", goal="Delegate tasks", allow_delegation=True)
researcher = Agent(role="Researcher", goal="Find info", tools=[search_tool])
writer = Agent(role="Writer", goal="Write article")

crew = Crew(
    agents=[researcher, writer],
    manager_agent=manager,
    process=Process.hierarchical,
    tasks=[Task(description="Write about AI")]
)
result = crew.kickoff()`
    },
    {
        id: 'planning',
        name: 'Planning',
        category: 'agentic',
        type: 'technique',
        shortDesc: 'Decomposing goals into actionable steps',
        definition: `Planning agents decompose a vague goal ("Book a flight") into a graph of dependencies.

**Techniques:**
- **Plan-and-Solve**: Generate the full plan upfront, then execute sequentially.
- **ReWOO (Reasoning WithOut Observation)**: Planner generates plan, Worker executes without intermediate LLM calls (cheaper/faster).
- **LATS (Language Agent Tree Search)**: Uses reflection and search (like MCTS) to explore plan paths.`,
        related: ['agentic-ai', 'chain-of-thought'],
        tags: ['reasoning', 'decomposition'],
        codeExample: `# Plan-and-Execute Pattern
def plan_and_execute(goal, llm, executor):
    plan = llm.invoke(f"Break down goal into steps: {goal}")
    steps = parse_plan(plan)
    
    for step in steps:
        for attempt in range(3):
            result = executor(step)
            if result.success: break
            step = llm.invoke(f"Step failed: {step}. Error: {result.error}. Revise.")
        else:
            raise Exception(f"Failed at step: {step}")
    return "Goal Completed"`
    },
    {
        id: 'reflection',
        name: 'Reflection',
        category: 'agentic',
        type: 'technique',
        shortDesc: 'Self-critique for self-improvement',
        definition: `Reflection allows agents to learn from mistakes within a single session. After executing a task, the agent generates a "reflection" on what went wrong, stored in memory to guide future attempts.

**Key Framework:** **Reflexion** (Shinn et al.). Uses a "trial" memory of (Action, Observation, Reflection).`,
        related: ['agentic-ai', 'planning'],
        tags: ['self-improvement', 'reasoning'],
        codeExample: `# Reflexion Pattern
def reflect_and_solve(task, llm, max_retries=3):
    reflections = []
    for i in range(max_retries):
        solution = llm.invoke(f"Task: {task}\\nPast Reflections: {reflections}")
        is_correct, feedback = execute_and_test(solution)
        if is_correct: return solution
        reflection = llm.invoke(f"Solution failed.\\nFeedback: {feedback}\\nReflect on why.")
        reflections.append(reflection)
    return None`
    },

    // ========== MCP SECTION ==========
    {
        id: 'mcp',
        name: 'MCP',
        fullName: 'Model Context Protocol',
        category: 'mcp',
        type: 'core',
        shortDesc: 'Universal standard for AI context integration',
        definition: `MCP (Anthropic) is an open protocol to standardize how AI assistants connect to data sources. It decouples the AI model from integrations.

**Architecture:**
- **MCP Host**: The AI application (e.g., Claude Desktop).
- **MCP Client**: Part of the host, speaks the protocol.
- **MCP Server**: A lightweight program exposing resources/tools (e.g., a Postgres server).

**Core Primitives:**
1. **Resources**: Read-only data (files, DB rows).
2. **Tools**: Functions the model can call.
3. **Prompts**: Templates users can invoke.`,
        related: ['tool-use', 'mcp-resources', 'mcp-tools'],
        tags: ['protocol', 'standard', 'interoperability'],
        codeExample: `# Minimal MCP Server
from mcp.server import Server
from mcp.types import Tool, TextContent

app = Server("example-server")

@app.list_tools()
def list_tools():
    return [Tool(name="echo", description="Echoes input", input_schema={...})]

@app.call_tool()
async def call_tool(name, arguments):
    if name == "echo":
        return [TextContent(type="text", text=arguments.get("message"))]

# Run: npx @anthropic-ai/mcp run server.py`
    },
    {
        id: 'mcp-tools',
        name: 'MCP Tools',
        category: 'mcp',
        type: 'technique',
        shortDesc: 'Functions exposed via MCP',
        definition: `MCP Tools are stateful, discoverable functions. Unlike raw JSON function calling, MCP Tools support:
- **Schema Validation**: Automatic validation of arguments.
- **Discovery**: Clients list available tools dynamically.
- **Annotations**: Human-readable descriptions for the LLM.`,
        related: ['mcp', 'tool-use'],
        tags: ['functions', 'execution'],
        codeExample: `# Defining an MCP Tool
@server.tool()
def query_database(sql: str) -> str:
    """Executes a read-only SQL query."""
    return execute_sql_safely(sql)`
    },
    {
        id: 'mcp-resources',
        name: 'MCP Resources',
        category: 'mcp',
        type: 'technique',
        shortDesc: 'URI-addressable data via MCP',
        definition: `MCP Resources provide a file-like interface to data. Accessible via URIs (e.g., file:///logs/app.log or postgres://db/users). Supports listing and reading.`,
        related: ['mcp', 'context-window'],
        tags: ['data', 'read-only'],
        codeExample: `# MCP Resource Template
@server.resource("logs://{log_id}")
def read_log(log_id: str) -> str:
    return open(f"/logs/{log_id}.log").read()`
    },

    // ========== ARCHITECTURE SECTION ==========
    {
        id: 'transformers',
        name: 'Transformers',
        category: 'architecture',
        type: 'core',
        shortDesc: 'The architecture behind modern AI',
        definition: `Transformers rely on **Self-Attention** to process sequences in parallel, replacing recurrence (RNNs).

**Key Equations:**
- **Attention**: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
- **LayerNorm**: Stabilizes training by normalizing across features.
- **Positional Encoding**: Injects order info since attention is permutation-invariant.`,
        related: ['attention-mechanism', 'llm'],
        tags: ['fundamental', 'deep-learning'],
        codeExample: `# Minimal Transformer Block
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads)
        self.ff = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.ReLU(), nn.Linear(4*d_model, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # Pre-LN architecture (more stable)
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        
        norm_x = self.norm2(x)
        ff_out = self.ff(norm_x)
        x = x + ff_out
        return x`
    },
    {
        id: 'moe',
        name: 'Mixture of Experts',
        category: 'architecture',
        type: 'technique',
        shortDesc: 'Sparse activation for efficient scaling',
        definition: `Mixture of Experts (MoE) scales model capacity without proportional compute cost.

**Mechanism:**
1. **Experts**: Specialized sub-networks (e.g., 8 Feed-Forward layers).
2. **Router/Gating Network**: Decides which expert to use for each token.
3. **Top-k Routing**: Usually k=1 or 2 experts activated per token.

**Trade-off:** High VRAM needed to hold all experts, but low FLOPs per token.`,
        related: ['transformers', 'inference'],
        tags: ['scaling', 'efficiency'],
        codeExample: `# Sparse MoE Layer
class MoELayer(nn.Module):
    def __init__(self, d_model, num_experts=8, top_k=2):
        super().__init__()
        self.experts = nn.ModuleList([FeedForward(d_model) for _ in range(num_experts)])
        self.gate = nn.Linear(d_model, num_experts)
        self.top_k = top_k

    def forward(self, x):
        gate_logits = self.gate(x)
        weights, indices = torch.topk(gate_logits, self.top_k, dim=-1)
        weights = torch.softmax(weights, dim=-1)
        
        output = torch.zeros_like(x)
        for i in range(self.top_k):
            expert_idx = indices[..., i]
            expert_weight = weights[..., i].unsqueeze(-1)
            expert_out = self.experts[expert_idx](x)
            output += expert_out * expert_weight
        return output`
    },
    {
        id: 'flash-attention',
        name: 'Flash Attention',
        category: 'architecture',
        type: 'technique',
        shortDesc: 'Memory-efficient exact attention',
        definition: `Flash Attention is an IO-aware algorithm that computes exact attention in O(N) memory instead of O(N^2).

**Key Insight:** Attention materialization dominates HBM (High Bandwidth Memory) access. Flash Attention computes attention block-by-block on fast SRAM, never writing the huge N x N matrix to HBM.

**Impact:** Enables 4-16x longer context lengths. Now standard in PyTorch 2.0+ and HuggingFace.`,
        related: ['attention-mechanism', 'context-window'],
        tags: ['optimization', 'memory'],
        codeExample: `# Usage in PyTorch 2.0+
import torch.nn.functional as F

# Automatically uses Flash Attention if on CUDA
F.scaled_dot_product_attention(q, k, v, is_causal=True)

# Explicitly control backend
with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False):
    attn_output = F.scaled_dot_product_attention(q, k, v)`
    },
    {
        id: 'llm',
        name: 'LLM',
        fullName: 'Large Language Model',
        category: 'architecture',
        type: 'core',
        shortDesc: 'Foundation models trained on massive text',
        definition: `LLMs are decoder-only Transformers trained on trillions of tokens via Next Token Prediction.

**Scaling Laws:** Performance (L) follows a power law with compute (C), data (D), and parameters (N).

**Architectural Nuances:**
- **RoPE (Rotary Positional Embeddings)**: Modern positional encoding (better length extrapolation).
- **SwiGLU**: Activation function replacing ReLU.
- **GQA (Grouped Query Attention)**: Speeds up inference by sharing KV heads.`,
        related: ['transformers', 'tokenization'],
        tags: ['foundation-model', 'generation'],
        codeExample: `# LLM Generation Config
from transformers import GenerationConfig

config = GenerationConfig(
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.95,         # Nucleus Sampling
    top_k=50,           # Top-K filtering
    repetition_penalty=1.1, # Prevent loops
    do_sample=True
)
output = model.generate(**inputs, generation_config=config)`
    },
    {
        id: 'tokenization',
        name: 'Tokenization',
        category: 'architecture',
        type: 'technique',
        shortDesc: 'Subword encoding for text',
        definition: `Tokenization converts text to integer IDs using subword algorithms.

**Algorithms:**
- **BPE (Byte Pair Encoding)**: Merges frequent byte pairs. Used by GPT, Llama.
- **WordPiece**: Used by BERT. Uses ## prefix for subwords.
- **Unigram**: Used by SentencePiece/T5. Probabilistic.

**Why not words?** Vocabulary size explodes. Why not chars? Context length explodes. Subwords are the sweet spot.`,
        related: ['llm', 'embeddings'],
        tags: ['preprocessing', 'encoding'],
        codeExample: `# Tiktoken (OpenAI BPE)
import tiktoken

enc = tiktoken.get_encoding("cl100k_base") # For GPT-4
ids = enc.encode("Hello World")
text = enc.decode(ids)

# Special token handling
ids = enc.encode("Hello <special>", allowed_special="all")`
    },

    // ========== TRAINING SECTION ==========
    {
        id: 'pre-training',
        name: 'Pre-training',
        category: 'training',
        type: 'core',
        shortDesc: 'Self-supervised learning on massive data',
        definition: `Pre-training builds foundational knowledge via **Next Token Prediction (NTP)**.

**Objective:** Maximize sum(log P(x_t | x_<t)).

**Compute Optimal (Chinchilla):** For a given compute budget, model size and data tokens should scale equally. Chinchilla showed many models were undertrained (too big, too little data).`,
        related: ['llm', 'fine-tuning'],
        tags: ['training', 'foundation'],
        codeExample: `# Distributed Training (Conceptual)
import torch.distributed as dist

# ZeRO Stage 3: Shards params, grads, and optimizer states
optimizer = DeepSpeedZeROOffload(optimizer)
        
def train_step(batch, model, optimizer):
    with torch.autocast("cuda"):
        loss = model(batch).loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()`
    },
    {
        id: 'fine-tuning',
        name: 'Fine-tuning',
        category: 'training',
        type: 'technique',
        shortDesc: 'Adapting models to tasks',
        definition: `Fine-tuning shifts the model distribution from generic pre-training data to specific tasks.

**Types:**
- **SFT (Supervised Fine-Tuning)**: Train on instruction-response pairs.
- **RLHF**: Align with human preferences.
- **PEFT**: Parameter-efficient tuning (LoRA, Adapters).`,
        related: ['lora', 'qlora'],
        tags: ['adaptation', 'specialization'],
        codeExample: `# SFT Loop
def train_sft(model, train_loader, optimizer):
    model.train()
    for batch in train_loader:
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()`
    },
    {
        id: 'lora',
        name: 'LoRA',
        fullName: 'Low-Rank Adaptation',
        category: 'training',
        type: 'technique',
        shortDesc: 'Efficient fine-tuning via low-rank updates',
        definition: `LoRA injects trainable rank decomposition matrices into Transformer layers.
For a weight matrix W, we add Delta_W = BA. B (d x r), A (r x k).
Only B and A are updated.

**Key Hyperparameters:**
- **Rank (r)**: Usually 8-64. Higher = more capacity.
- **Alpha**: Scaling factor. Effective learning rate scales with alpha/r.
- **Target Modules**: Usually query/value projections.`,
        related: ['fine-tuning', 'qlora'],
        tags: ['efficient', 'peft'],
        codeExample: `# LoRA Config
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,
    lora_alpha=32,       # Scale = alpha/r = 2
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none"
)
model = get_peft_model(base_model, config)
# Trainable params: 0.1% of total`
    },
    {
        id: 'rlhf',
        name: 'RLHF',
        fullName: 'Reinforcement Learning from Human Feedback',
        category: 'training',
        type: 'technique',
        shortDesc: 'Aligning models with human values',
        definition: `RLHF aligns LLMs with human intent via 3 steps:
1. **SFT**: Supervised fine-tuning on high-quality data.
2. **Reward Model (RM)**: Train a classifier to predict human preference (A vs B).
3. **PPO**: Optimize the LLM to maximize reward using Reinforcement Learning.`,
        related: ['fine-tuning', 'dpo'],
        tags: ['alignment', 'human-feedback'],
        codeExample: `# PPO Step (Conceptual)
def ppo_step(policy, ref_policy, reward_model, batch):
    gen_ids = policy.generate(batch.input_ids)
    reward = reward_model(gen_ids)
    kl = kl_divergence(policy(gen_ids), ref_policy(gen_ids))
    loss = - (reward - 0.1 * kl)
    loss.backward()`
    },
    {
        id: 'dpo',
        name: 'DPO',
        fullName: 'Direct Preference Optimization',
        category: 'training',
        type: 'technique',
        shortDesc: 'Simpler alignment than RLHF',
        definition: `DPO removes the Reward Model training step. It optimizes the policy directly using preference pairs (x, y_w, y_l) (winner/loser).

**Loss Function:** Directly optimizes the log-likelihood ratio of the chosen vs rejected response relative to a reference model.`,
        related: ['rlhf', 'fine-tuning'],
        tags: ['alignment', 'simplified'],
        codeExample: `# DPO Loss
def dpo_loss(policy, reference, query, chosen, rejected, beta=0.1):
    pi_chosen = policy.log_prob(query, chosen)
    pi_rejected = policy.log_prob(query, rejected)
    ref_chosen = reference.log_prob(query, chosen)
    ref_rejected = reference.log_prob(query, rejected)
    
    pi_logratios = pi_chosen - pi_rejected
    ref_logratios = ref_chosen - ref_rejected
    
    logits = beta * (pi_logratios - ref_logratios)
    loss = -torch.nn.functional.logsigmoid(logits).mean()
    return loss`
    },

    // ========== PROMPTING SECTION ==========
    {
        id: 'chain-of-thought',
        name: 'Chain of Thought',
        category: 'prompting',
        type: 'technique',
        shortDesc: 'Reasoning traces before answers',
        definition: `CoT elicits reasoning by asking the model to "think step by step". It improves performance on arithmetic, logic, and symbolic reasoning.

**Variants:**
- **Zero-Shot CoT**: "Let's think step by step."
- **Few-Shot CoT**: Provide examples of reasoning chains.
- **Self-Consistency**: Sample multiple reasoning paths and vote.`,
        related: ['prompt-engineering', 'planning'],
        tags: ['reasoning', 'technique'],
        codeExample: `# Zero-Shot CoT
prompt = """
Question: If I have 3 apples and eat 1, how many are left?
Let's think step by step.
"""`
    },
    {
        id: 'tree-of-thoughts',
        name: 'Tree of Thoughts',
        category: 'prompting',
        type: 'technique',
        shortDesc: 'Exploring multiple reasoning paths',
        definition: `ToT generalizes CoT by exploring a tree of thoughts. At each step, the model generates multiple possible next thoughts, evaluates them, and searches (BFS/DFS).

**Use Case:** Complex puzzles, creative writing, strategic planning where linear reasoning fails.`,
        related: ['chain-of-thought', 'planning'],
        tags: ['reasoning', 'search'],
        codeExample: `# ToT Search (Conceptual)
def tree_of_thoughts(prompt, llm):
    states = [""]
    for depth in range(3):
        new_states = []
        for state in states:
            thoughts = llm.generate(f"{prompt}\\nCurrent: {state}\\nNext thoughts:")
            for thought in parse(thoughts):
                score = llm.evaluate(thought)
                new_states.append((score, state + thought))
        states = sorted(new_states, reverse=True)[:3]
    return states[0]`
    },
    {
        id: 'prompt-engineering',
        name: 'Prompt Engineering',
        category: 'prompting',
        type: 'core',
        shortDesc: 'Optimizing inputs for outputs',
        definition: `Designing inputs to guide model behavior.
**Techniques:** Role Prompting ("You are an expert..."), Few-Shot (Examples), Structured Output (JSON mode), and Context Management (retrieving relevant context).`,
        related: ['few-shot', 'chain-of-thought'],
        tags: ['fundamental', 'optimization'],
        // FIX: Removed problematic backticks that caused SyntaxError
        codeExample: `# Advanced Prompt
PROMPT = """
Role: You are a Python Expert.
Task: Write a function to sort a list.

Constraints:
- Time Complexity: O(n log n)
- Include type hints
- Handle edge cases (None, empty list)

Output Format:
def function(args):
    # Implementation
    return result
"""`
    },

    // ========== INFRASTRUCTURE SECTION ==========
    {
        id: 'inference',
        name: 'Inference',
        category: 'infrastructure',
        type: 'core',
        shortDesc: 'Serving models at scale',
        definition: `Running models in production requires optimizing for latency, throughput, and cost.

**Key Optimizations:**
1. **KV Caching**: Store previous key/values.
2. **Continuous Batching**: Pack multiple requests.
3. **Speculative Decoding**: Draft model guesses, main model verifies.
4. **Quantization**: FP16 -> INT4.`,
        related: ['quantization', 'kv-cache'],
        tags: ['production', 'serving'],
        codeExample: `# vLLM Serving
from vllm import LLM, SamplingParams

llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.2")
params = SamplingParams(max_tokens=128, temperature=0.7)
outputs = llm.generate(prompts, params)`
    },
    {
        id: 'quantization',
        name: 'Quantization',
        category: 'infrastructure',
        type: 'technique',
        shortDesc: 'Compressing models',
        definition: `Reducing precision to save memory/bandwidth.
**PTQ (Post-Training Quantization):** Fast, slight accuracy drop.
**QAT (Quantization-Aware Training):** Slower, better accuracy.
**GPTQ/AWQ:** Popular LLM quantization methods.`,
        related: ['inference', 'lora'],
        tags: ['compression', 'efficiency'],
        codeExample: `# GPTQ Config
from transformers import GPTQConfig
config = GPTQConfig(bits=4, dataset="c4")
model = AutoModelForCausalLM.from_pretrained("model", quantization_config=config)`
    },
    {
        id: 'kv-cache',
        name: 'KV Cache',
        category: 'infrastructure',
        type: 'technique',
        shortDesc: 'Caching attention states',
        definition: `Stores past Key/Value tensors to avoid recomputation during generation. **PagedAttention (vLLM)** manages KV cache like virtual memory to prevent fragmentation.`,
        related: ['inference', 'attention-mechanism'],
        tags: ['optimization', 'memory'],
        codeExample: `# Paged Attention (vLLM handles internally)
class PagedKVCache:
    def __init__(self, block_size=16):
        self.blocks = {}
        self.block_tables = {}`
    },

    // ========== APPLICATIONS SECTION ==========
    {
        id: 'chatbots',
        name: 'Chatbots',
        category: 'applications',
        type: 'application',
        shortDesc: 'Conversational AI',
        definition: `Chatbots combine RAG, Tools, and Memory.
**State Management:** Handling history truncation/summarization.
**Safety:** Content filtering, guardrails.`,
        related: ['rag', 'agentic-ai'],
        tags: ['conversational', 'interface'],
        codeExample: `# Chatbot
class ChatBot:
    def __init__(self, llm, retriever):
        self.history = []
    def chat(self, msg):
        ctx = retriever.search(msg)
        prompt = build_prompt(self.history, ctx, msg)
        resp = self.llm.invoke(prompt)
        self.history.append((msg, resp))
        return resp`
    },
    {
        id: 'code-generation',
        name: 'Code Generation',
        category: 'applications',
        type: 'application',
        shortDesc: 'AI coding assistants',
        definition: `Models trained on code (Codex, StarCoder).
**FIM (Fill-In-the-Middle)**: Completing code within a file.
**Context:** Needs full file context for imports/dependencies.`,
        related: ['llm', 'tool-use'],
        tags: ['development', 'productivity'],
        codeExample: `# FIM (Fill-In-Middle)
prefix = "def sum(a, b):"
suffix = "return result"
middle = model.generate(f"<PRE>{prefix}<SUF>{suffix}<MID>")
print(prefix + middle + suffix)`
    }
];

// ==========================================
// BUILD & EXPORT
// ==========================================
const KnowledgeBase = {
    meta: {
        version: '2.1.1', // Bumped version
        lastUpdated: '2025-01-15',
        description: 'Interactive AI Knowledge Graph - God Mode'
    },
    categories: CategoryData,
    terms: TermData.map(t => createTerm(t))
};

// ==========================================
// UTILITY FUNCTIONS
// ==========================================
const KnowledgeUtils = {
    addCategory(category) {
        if (!category.id || !category.name) return false;
        if (KnowledgeBase.categories.find(c => c.id === category.id)) return false;
        KnowledgeBase.categories.push({...category});
        return true;
    },

    addTerm(termConfig) {
        if (!termConfig.id || !termConfig.name) return false;
        if (KnowledgeBase.terms.find(t => t.id === termConfig.id)) return false;
        KnowledgeBase.terms.push(createTerm(termConfig));
        return true;
    },

    getTerm(id) {
        return KnowledgeBase.terms.find(t => t.id === id);
    },

    getTermsByCategory(categoryId) {
        return KnowledgeBase.terms.filter(t => t.category === categoryId);
    },

    getRelatedTerms(termId) {
        const term = this.getTerm(termId);
        if (!term || !term.related) return [];
        return term.related.map(r => this.getTerm(r)).filter(Boolean);
    },

    searchTerms(query) {
        if (!query) return KnowledgeBase.terms;
        const q = query.toLowerCase();
        return KnowledgeBase.terms.filter(t => 
            t.name.toLowerCase().includes(q) ||
            t.shortDesc.toLowerCase().includes(q) ||
            t.fullName.toLowerCase().includes(q) ||
            t.tags.some(tag => tag.toLowerCase().includes(q))
        );
    },

    getStats() {
        return {
            categories: KnowledgeBase.categories.length,
            terms: KnowledgeBase.terms.length,
            connections: KnowledgeBase.terms.reduce((sum, t) => sum + (t.related?.length || 0), 0),
            byCategory: KnowledgeBase.categories.reduce((acc, cat) => {
                acc[cat.id] = this.getTermsByCategory(cat.id).length;
                return acc;
            }, {})
        };
    },

    export() {
        return JSON.stringify(KnowledgeBase, null, 2);
    },

    import(jsonString) {
        try {
            const data = JSON.parse(jsonString);
            if (data.categories && data.terms) {
                KnowledgeBase.categories = data.categories;
                KnowledgeBase.terms = data.terms;
                return true;
            }
        } catch (e) {
            console.error('Import failed:', e);
        }
        return false;
    }
};

console.log('KnowledgeBase loaded:', KnowledgeBase.categories.length, 'categories,', KnowledgeBase.terms.length, 'terms');
window.KnowledgeBase = KnowledgeBase;
window.KnowledgeUtils = KnowledgeUtils;
