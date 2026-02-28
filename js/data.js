/**
 * AI Knowledge Base - Extensible Data Structure
 * GOD MODE ENHANCED: Deeper content, more concepts, production code.
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
        type: config.type || 'technique', // core, technique, infrastructure, application
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
    {
        id: 'rag',
        name: 'RAG',
        fullName: 'Retrieval Augmented Generation',
        color: '#0891b2',
        description: 'Techniques for augmenting LLMs with external knowledge'
    },
    {
        id: 'agentic',
        name: 'Agentic AI',
        fullName: 'Autonomous Agent Systems',
        color: '#6366f1',
        description: 'Autonomous AI systems that plan and execute tasks'
    },
    {
        id: 'mcp',
        name: 'MCP',
        fullName: 'Model Context Protocol',
        color: '#8b5cf6',
        description: 'Protocol for connecting AI assistants to systems'
    },
    {
        id: 'architecture',
        name: 'Architecture',
        fullName: 'Model Architecture',
        color: '#f59e0b',
        description: 'Fundamental neural network architectures'
    },
    {
        id: 'training',
        name: 'Training',
        fullName: 'Training Methods',
        color: '#10b981',
        description: 'Methods for training and fine-tuning models'
    },
    {
        id: 'prompting',
        name: 'Prompting',
        fullName: 'Prompt Engineering',
        color: '#ec4899',
        description: 'Techniques for effective model interaction'
    },
    {
        id: 'infrastructure',
        name: 'Infrastructure',
        fullName: 'AI Infrastructure',
        color: '#ef4444',
        description: 'Systems and tools for AI deployment'
    },
    {
        id: 'applications',
        name: 'Applications',
        fullName: 'AI Applications',
        color: '#14b8a6',
        description: 'Real-world AI implementations'
    }
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
        definition: 'RAG (Retrieval Augmented Generation) bridges the gap between parametric knowledge (model weights) and non-parametric knowledge (external databases). It operates in three stages: 1) Retrieval: finding relevant documents using vector similarity or keyword search. 2) Augmentation: injecting context into the prompt. 3) Generation: synthesizing an answer grounded in the provided context. It reduces hallucinations and provides citations.',
        related: ['vector-database', 'embeddings', 'chunking', 'reranking', 'hybrid-search'],
        tags: ['core', 'production', 'grounding'],
        codeExample: `# Production RAG Pipeline with Hybrid Search
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank

# 1. Hybrid Retrieval (Vector + Keyword)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 10

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6]
)

# 2. Reranking for Precision
compressor = CohereRerank()
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=ensemble_retriever
)

# 3. Generate
docs = compression_retriever.get_relevant_documents(query)
response = llm.invoke(f"Context: {docs}\\nQuestion: {query}")`
    },
    {
        id: 'vector-database',
        name: 'Vector Database',
        category: 'rag',
        type: 'infrastructure',
        shortDesc: 'Optimized storage for high-dimensional vector similarity search',
        definition: 'Vector databases are specialized systems for storing embedding vectors and performing Approximate Nearest Neighbor (ANN) search. Unlike traditional DBs, they optimize for cosine/euclidean similarity at scale. Key algorithms include HNSW (Hierarchical Navigable Small World) for speed and IVF (Inverted File Index) for memory efficiency. Examples: Pinecone (managed), Milvus (open-source), Weaviate (hybrid), Qdrant (Rust-based efficiency).',
        related: ['embeddings', 'rag', 'hnsw'],
        tags: ['infrastructure', 'storage', 'similarity-search'],
        codeExample: `# Qdrant Setup with HNSW Config
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, HnswConfigDiff

client = QdrantClient(":memory:")

client.create_collection(
    collection_name="production_docs",
    vectors_config=VectorParams(
        size=1536,
        distance=Distance.COSINE,
        hnsw_config=HnswConfigDiff(
            m=16,             # Connections per node
            ef_construct=100  # Construction search depth
        )
    )
)`
    },
    {
        id: 'embeddings',
        name: 'Embeddings',
        category: 'rag',
        type: 'core',
        shortDesc: 'Dense vector representations capturing semantic meaning',
        definition: 'Embeddings map discrete inputs (text, images) into continuous high-dimensional vectors where distance correlates with semantic similarity. Modern embedding models (OpenAI text-embedding-3, Cohere, Voyage) are trained using contrastive learning (e.g., InfoNCE loss) to pull similar pairs together and push dissimilar ones apart. Dimensionality trade-offs: larger dimensions capture more nuance but increase storage/compute cost.',
        related: ['vector-database', 'semantic-search', 'transformers'],
        tags: ['fundamental', 'representations', 'semantic'],
        codeExample: `# Embedding Generation & Similarity
from openai import OpenAI
import numpy as np

client = OpenAI()

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

vec1 = get_embedding("AI is transforming industries")
vec2 = get_embedding("Machine learning changes business")
print(f"Similarity: {cosine_sim(vec1, vec2):.3f}")`
    },
    {
        id: 'hybrid-search',
        name: 'Hybrid Search',
        category: 'rag',
        type: 'technique',
        shortDesc: 'Combining vector similarity with keyword matching',
        definition: 'Hybrid search merges the semantic understanding of vector search with the precise lexical matching of keyword search (BM25). Essential for domains with specific terminology (medical, legal). Reciprocal Rank Fusion (RRF) is the standard fusion algorithm, combining ranked lists without tuning score normalization.',
        related: ['semantic-search', 'reranking'],
        tags: ['retrieval', 'precision', 'production'],
        codeExample: `# Reciprocal Rank Fusion Implementation
def reciprocal_rank_fusion(results_dict, k=60):
    fused_scores = {}
    
    for system, docs in results_dict.items():
        for rank, doc in enumerate(docs):
            if doc.id not in fused_scores:
                fused_scores[doc.id] = {"score": 0, "doc": doc}
            fused_scores[doc.id]["score"] += 1 / (k + rank + 1)
    
    reranked = sorted(fused_scores.values(), key=lambda x: x["score"], reverse=True)
    return [item["doc"] for item in reranked]`
    },
    {
        id: 'graph-rag',
        name: 'GraphRAG',
        category: 'rag',
        type: 'technique',
        shortDesc: 'Using knowledge graphs for enhanced retrieval',
        definition: 'GraphRAG leverages knowledge graphs instead of simple vector indexes to capture entity relationships. It excels at multi-hop reasoning (connecting facts across documents). By extracting entities and relationships, it builds a graph structure. Queries traverse the graph to find connected information, providing context that vector similarity often misses.',
        related: ['rag', 'knowledge-graph'],
        tags: ['advanced', 'reasoning', 'knowledge-graph'],
        codeExample: `# Conceptual GraphRAG using NetworkX
import networkx as nx

G = nx.Graph()

# Build Graph from Entities
G.add_edge("Elon Musk", "Tesla", relation="CEO")
G.add_edge("Tesla", "Austin", relation="HQ")
G.add_edge("Elon Musk", "SpaceX", relation="Founder")

# Multi-hop Query
def multi_hop_query(graph, start_entity, relationship, hops=2):
    neighbors = list(nx.single_source_shortest_path_length(graph, start_entity, cutoff=hops))
    return [n for n in neighbors if G.has_edge(start_entity, n)]

print(multi_hop_query(G, "Elon Musk", "founder"))`
    },
    {
        id: 'chunking',
        name: 'Chunking',
        category: 'rag',
        type: 'technique',
        shortDesc: 'Splitting documents for retrieval optimization',
        definition: 'Chunking balances context preservation against retrieval precision. Too large = noise; too small = lost context. Strategies include RecursiveCharacterTextSplitter (hierarchical), SemanticChunking (sentence embeddings), and ParentDocumentRetriever (retrieving small chunks but returning parent document). Overlap prevents context loss at boundaries.',
        related: ['rag', 'embeddings'],
        tags: ['preprocessing', 'documents'],
        codeExample: `# Parent Document Retriever
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

# Store small chunks for search, return full parent docs
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=InMemoryStore(),
    child_splitter=RecursiveCharacterTextSplitter(chunk_size=400),
    parent_splitter=RecursiveCharacterTextSplitter(chunk_size=2000)
)`
    },
    {
        id: 'reranking',
        name: 'Reranking',
        category: 'rag',
        type: 'technique',
        shortDesc: 'Re-scoring documents for final relevance',
        definition: 'Reranking uses a Cross-Encoder model that jointly encodes query+document pairs to produce a relevance score. Unlike Bi-Encoders (embeddings) which compute similarity in isolation, Cross-Encoders can attend to interactions between query and document terms. Much slower than vector search but significantly higher precision.',
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
        shortDesc: 'Autonomous systems pursuing goals through iterative actions',
        definition: 'Agentic AI transitions LLMs from passive responders to active goal-seekers. Key loop: Observe -> Think -> Act -> Reflect. Agents utilize tools, maintain memory, and plan using ReAct (Reasoning + Acting). Challenges include loop prevention and reliable error recovery.',
        related: ['tool-use', 'planning', 'memory-agents', 'multi-agent'],
        tags: ['core', 'autonomous'],
        codeExample: `# Core ReAct Loop
def run_agent(query, llm, tools):
    thought_history = []
    for _ in range(10): # Max steps
        response = llm.invoke(f"""
        Query: {query}
        History: {thought_history}
        Think step-by-step. Available tools: {tools}
        Format: Thought: ... Action: tool_name(args)
        """)
        
        if "Final Answer:" in response:
            return parse_final_answer(response)
        
        action = parse_action(response)
        observation = execute_tool(action)
        thought_history.append(f"Thought: {response}\\nObservation: {observation}")`
    },
    {
        id: 'tool-use',
        name: 'Tool Use',
        category: 'agentic',
        type: 'core',
        shortDesc: 'Enabling LLMs to execute external functions',
        definition: 'Tool Use extends LLMs beyond text generation. The model generates structured outputs (JSON) matching a defined schema. The runtime environment parses these outputs and executes actual code (API calls, DB queries). Reliability requires strict schema validation and error handling.',
        related: ['agentic-ai', 'mcp'],
        tags: ['integration', 'actions'],
        codeExample: `# Structured Tool Output
from pydantic import BaseModel
from langchain.tools import tool

class CalculatorInput(BaseModel):
    a: int
    b: int

@tool("multiply", args_schema=CalculatorInput)
def multiply(a: int, b: int) -> int:
    """Multiplies a and b."""
    return a * b

# Binding tools to LLM
llm_with_tools = llm.bind_tools([multiply])`
    },
    {
        id: 'multi-agent',
        name: 'Multi-Agent Systems',
        category: 'agentic',
        type: 'technique',
        shortDesc: 'Collaborative AI agents with specialized roles',
        definition: 'Multi-agent systems distribute tasks among specialized agents (e.g., Researcher, Writer, Critic). Communication patterns: Hierarchical (Manager-Worker), Sequential (Assembly Line), or Joint Chat (Round Table). Frameworks like CrewAI and LangGraph define the communication graph and state management.',
        related: ['agentic-ai', 'planning'],
        tags: ['collaboration', 'complex-systems'],
        codeExample: `# CrewAI Hierarchical Process
from crewai import Crew, Agent, Task, Process

manager = Agent(role="Project Manager", ...)
researcher = Agent(role="Researcher", tools=[search_tool], ...)
writer = Agent(role="Technical Writer", ...)

crew = Crew(
    agents=[researcher, writer],
    manager_agent=manager,
    process=Process.hierarchical,
    tasks=[...]
)`
    },
    {
        id: 'planning',
        name: 'Planning',
        category: 'agentic',
        type: 'technique',
        shortDesc: 'Decomposing complex goals into sub-tasks',
        definition: 'Planning agents break high-level goals into executable steps. Approaches: Plan-and-Solve (generate plan first, then execute), ReWOO (Reasoning WithOut Observation), and LATS (Language Agent Tree Search). Effective planning requires backtracking when steps fail.',
        related: ['agentic-ai', 'chain-of-thought'],
        tags: ['reasoning', 'decomposition'],
        codeExample: `# Plan-and-Execute Pattern
def plan_and_execute(goal, llm, executor):
    # 1. Generate Plan
    plan = llm.invoke(f"Break down goal: {goal} into steps")
    steps = parse_plan(plan)
    
    for step in steps:
        for attempt in range(3):
            result = executor(step)
            if result.success:
                break
            step = llm.invoke(f"Step failed: {step}. Revise step given error: {result.error}")
        else:
            raise Exception("Max retries reached")`
    },
    {
        id: 'reflection',
        name: 'Reflection',
        category: 'agentic',
        type: 'technique',
        shortDesc: 'Self-critique to improve future performance',
        definition: 'Reflection allows agents to analyze their own outputs, identify errors, and generate better strategies. Pattern: Generate -> Critique -> Refine. Reflexion framework stores these critiques in long-term memory to guide future attempts, significantly boosting performance on coding/math tasks.',
        related: ['agentic-ai', 'planning'],
        tags: ['self-improvement', 'reasoning'],
        codeExample: `# Reflexion Pattern
def reflect_and_retry(task, llm, max_retries=3):
    history = []
    
    for i in range(max_retries):
        # Generate
        solution = llm.invoke(f"Task: {task}. History: {history}")
        
        # Evaluate (e.g., unit tests)
        is_correct, feedback = evaluate(solution)
        
        if is_correct: return solution
        
        # Reflect
        reflection = llm.invoke(f"Failed attempt: {solution}\\nFeedback: {feedback}\\nReflect on why it failed.")
        history.append(reflection)
    
    return None`
    },

    // ========== MCP SECTION ==========
    {
        id: 'mcp',
        name: 'MCP',
        fullName: 'Model Context Protocol',
        category: 'mcp',
        type: 'core',
        shortDesc: 'Universal protocol for AI context integration',
        definition: 'MCP (Model Context Protocol) is an open standard (by Anthropic) for connecting AI assistants to systems. It decouples AI models from specific data sources using a Client-Server architecture. Servers expose Resources (read data), Tools (functions), and Prompts (templates). Clients (e.g., Claude Desktop) discover and invoke these capabilities dynamically.',
        related: ['tool-use', 'mcp-resources', 'mcp-tools'],
        tags: ['protocol', 'standard', 'interoperability'],
        codeExample: `# MCP Server Implementation
from mcp.server import Server
from mcp.types import Tool, Resource

app = Server("my-mcp-server")

@app.list_tools()
def list_tools():
    return [
        Tool(name="search_docs", description="Search documentation", input_schema={...})
    ]

@app.call_tool()
def call_tool(name, arguments):
    if name == "search_docs":
        return search(arguments["query"])

# Run via: npx @anthropic-ai/mcp run server.py`
    },
    {
        id: 'mcp-tools',
        name: 'MCP Tools',
        category: 'mcp',
        type: 'technique',
        shortDesc: 'Executable functions exposed via MCP',
        definition: 'MCP Tools are stateful functions exposed by an MCP server. Unlike simple function calling, tools in MCP are discoverable via list_tools and have standardized JSON Schemas. Support for server-to-server composition allows chaining tool calls.',
        related: ['mcp', 'tool-use'],
        tags: ['functions', 'execution'],
        codeExample: `# Defining an MCP Tool
from mcp.server import Server

server = Server("data-tools")

@server.tool()
def query_database(sql: str) -> str:
    \"\"\"Execute SQL safely. Read-only access.\"\"\"
    # Implementation details
    return execute_readonly_sql(sql)

# The schema is auto-generated for the client`
    },
    {
        id: 'mcp-resources',
        name: 'MCP Resources',
        category: 'mcp',
        type: 'technique',
        shortDesc: 'URI-addressable data via MCP',
        definition: 'MCP Resources provide read-only access to data via URI templates. Clients can read specific resources or list available resources. Useful for exposing files, database tables, or API endpoints without full tool complexity.',
        related: ['mcp', 'context-window'],
        tags: ['data', 'read-only'],
        codeExample: `# MCP Resource Template
@server.resource("docs://{doc_id}")
def get_doc(doc_id: str) -> str:
    return db.find_document(doc_id)

# Client reads via URI
content = await session.read_resource("docs/12345")`
    },

    // ========== ARCHITECTURE SECTION ==========
    {
        id: 'transformers',
        name: 'Transformers',
        category: 'architecture',
        type: 'core',
        shortDesc: 'Attention-based neural networks dominating NLP',
        definition: 'Transformers replaced RNNs using self-attention, allowing parallel processing of sequences. Key innovation: Positional Encodings inject sequence order. The architecture consists of Encoder (BERT) and Decoder (GPT) stacks. Scaled Dot-Product Attention computes compatibility between all tokens.',
        related: ['attention-mechanism', 'llm', 'self-attention'],
        tags: ['fundamental', 'deep-learning'],
        codeExample: `# Minimal Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.att = nn.MultiheadAttention(d_model, n_heads)
        self.ff = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.ReLU(), nn.Linear(4*d_model, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        attn_out, _ = self.att(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x`
    },
    {
        id: 'moe',
        name: 'Mixture of Experts',
        category: 'architecture',
        type: 'technique',
        shortDesc: 'Sparse activation for massive scaling',
        definition: 'MoE (Mixture of Experts) scales parameter count without linear compute cost. A Gating Network routes input tokens to a subset of Expert layers. E.g., GPT-4 and Mixtral use 8 experts, routing to 2 per token. This achieves 8x parameters at ~2x compute cost.',
        related: ['transformers', 'inference'],
        tags: ['scaling', 'efficiency'],
        codeExample: `# Sparse MoE Layer
class MoELayer(nn.Module):
    def __init__(self, d_model, num_experts=8, top_k=2):
        super().__init__()
        self.experts = nn.ModuleList([Expert(d_model) for _ in range(num_experts)])
        self.gate = nn.Linear(d_model, num_experts)
        self.top_k = top_k

    def forward(self, x):
        gate_logits = self.gate(x)
        weights, indices = torch.topk(gate_logits, self.top_k, dim=-1)
        weights = torch.softmax(weights, dim=-1)
        
        # Sparse computation
        output = torch.zeros_like(x)
        for i in range(self.top_k):
            expert_idx = indices[:, :, i]
            expert_weight = weights[:, :, i].unsqueeze(-1)
            expert_out = self.experts[expert_idx](x)
            output += expert_out * expert_weight
        return output`
    },
    {
        id: 'flash-attention',
        name: 'Flash Attention',
        category: 'architecture',
        type: 'technique',
        shortDesc: 'IO-aware exact attention algorithm',
        definition: 'Flash Attention optimizes attention by minimizing HBM (High Bandwidth Memory) access. It computes attention block-by-block on SRAM, avoiding the massive O(N^2) intermediate matrix. It is exact (not approximate) and enables 4x longer context training.',
        related: ['attention-mechanism', 'context-window'],
        tags: ['optimization', 'performance'],
        codeExample: `# Flash Attention usage in PyTorch 2.0+
# Enabled by default with scaled_dot_product_attention
import torch.nn.functional as F

F.scaled_dot_product_attention(q, k, v, is_causal=True)

# Or explicitly via xformers/sdpa backends
with torch.backends.cuda.sdp_kernel(enable_flash=True):
    out = F.scaled_dot_product_attention(q, k, v)`
    },
    {
        id: 'llm',
        name: 'LLM',
        fullName: 'Large Language Model',
        category: 'architecture',
        type: 'core',
        shortDesc: 'Foundation models trained on massive text corpora',
        definition: 'LLMs are transformer-based models pre-trained on trillions of tokens to predict the next token. Scaling laws dictate performance improves predictably with parameters, data, and compute. Architecture choice (Decoder-only) favors generation. Key sizes: 7B (edge), 70B (enterprise), 1T+ (frontier).',
        related: ['transformers', 'tokenization', 'inference'],
        tags: ['foundation-model', 'generation'],
        codeExample: `# LLM Generation Config
from transformers import GenerationConfig

gen_config = GenerationConfig(
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.95,
    top_k=50,
    repetition_penalty=1.1,
    do_sample=True
)

output = model.generate(**inputs, generation_config=gen_config)`
    },
    {
        id: 'tokenization',
        name: 'Tokenization',
        category: 'architecture',
        type: 'technique',
        shortDesc: 'Subword encoding for text representation',
        definition: 'Tokenization converts raw text to integer IDs. BPE (Byte Pair Encoding) merges frequent character pairs iteratively. It balances vocabulary size against sequence length. Special tokens (<BOS>, <EOS>, <PAD>) handle sequence boundaries. Tokenizer choice affects model behavior (e.g., code efficiency).',
        related: ['llm', 'embeddings'],
        tags: ['preprocessing', 'encoding'],
        codeExample: `# Tiktoken usage
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

text = "Hello world"
ids = enc.encode(text)
decoded = enc.decode(ids)

# Handling special tokens
ids_with_special = enc.encode(text, allowed_special="all")`
    },

    // ========== TRAINING SECTION ==========
    {
        id: 'pre-training',
        name: 'Pre-training',
        category: 'training',
        type: 'core',
        shortDesc: 'Self-supervised learning on massive datasets',
        definition: 'Pre-training teaches general knowledge via Next Token Prediction (NTP). Compute-optimal training (Chinchilla scaling) balances model size and data. Requires distributed training frameworks (Megatron-LM, DeepSpeed). Gradients are accumulated across thousands of GPUs with ZeRO optimization.',
        related: ['llm', 'fine-tuning'],
        tags: ['training', 'foundation'],
        codeExample: `# Distributed Training Loop (Conceptual)
import torch.distributed as dist

def train_step(batch, model, optimizer):
    with torch.autocast("cuda"):
        loss = model(batch).loss
    
    loss.backward()
    
    # Gradient clipping to prevent explosion
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    optimizer.step()
    optimizer.zero_grad()
    
    return loss.item()`
    },
    {
        id: 'fine-tuning',
        name: 'Fine-tuning',
        category: 'training',
        type: 'technique',
        shortDesc: 'Adapting pre-trained weights to tasks',
        definition: 'Fine-tuning shifts model distribution from general to specific. Full fine-tuning updates all weights (compute-intensive). Parameter-Efficient Fine-Tuning (PEFT) updates <1% weights via Adapters, LoRA, or Prefix Tuning, preserving base knowledge.',
        related: ['lora', 'qlora'],
        tags: ['adaptation', 'specialization'],
        codeExample: `# Fine-tuning Loop with Evaluation
def train_epoch(model, train_loader, val_loader, optimizer, epochs):
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = sum(model(**b).loss for b in val_loader)
        
        print(f"Epoch {epoch}: Val Loss {val_loss}")`
    },
    {
        id: 'lora',
        name: 'LoRA',
        fullName: 'Low-Rank Adaptation',
        category: 'training',
        type: 'technique',
        shortDesc: 'Efficient fine-tuning via low-rank updates',
        definition: 'LoRA freezes pre-trained weights and injects trainable rank decomposition matrices into each layer. For a weight matrix W, LoRA adds BA where B and A are low-rank. This allows training multiple lightweight adapters per base model, sharing the massive base weights.',
        related: ['fine-tuning', 'qlora'],
        tags: ['efficient', 'peft'],
        codeExample: `# LoRA Fine-tuning
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,               # Rank
    lora_alpha=32,      # Scaling factor
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none"
)

model = get_peft_model(base_model, config)

# Trainable params: 4M || all params: 7B || trainable%: 0.06%`
    },
    {
        id: 'rlhf',
        name: 'RLHF',
        fullName: 'Reinforcement Learning from Human Feedback',
        category: 'training',
        type: 'technique',
        shortDesc: 'Aligning models with human preferences',
        definition: 'RLHF optimizes models for helpfulness/harmlessness. Steps: 1) Collect preference comparisons (A vs B). 2) Train Reward Model (RM) to predict preferences. 3) Optimize policy (LLM) to maximize RM reward using PPO. Crucial for aligning behavior with intent.',
        related: ['fine-tuning', 'dpo'],
        tags: ['alignment', 'human-feedback'],
        codeExample: `# PPO Step (Conceptual)
def ppo_step(policy, ref_policy, reward_model, batch):
    # Generate
    gen_tokens = policy.generate(batch.input_ids)
    
    # Score with Reward Model
    rewards = reward_model(gen_tokens)
    
    # PPO Objective
    ratio = policy.log_prob(gen_tokens) / ref_policy.log_prob(gen_tokens)
    clipped_ratio = torch.clamp(ratio, 0.8, 1.2)
    
    loss = -torch.min(ratio * rewards, clipped_ratio * rewards).mean()
    
    loss.backward()`
    },
    {
        id: 'dpo',
        name: 'DPO',
        fullName: 'Direct Preference Optimization',
        category: 'training',
        type: 'technique',
        shortDesc: 'Simpler alternative to RLHF',
        definition: 'DPO eliminates the Reward Model training step of RLHF. It directly optimizes the policy using preference pairs (chosen vs rejected). The loss function increases likelihood of chosen outputs while decreasing rejected ones, relative to a reference model.',
        related: ['rlhf', 'fine-tuning'],
        tags: ['alignment', 'simplified'],
        codeExample: `# DPO Loss Function
def dpo_loss(policy, reference, input_ids, chosen_ids, rejected_ids, beta=0.1):
    policy_chosen = policy.log_prob(input_ids, chosen_ids)
    policy_rejected = policy.log_prob(input_ids, rejected_ids)
    
    with torch.no_grad():
        ref_chosen = reference.log_prob(input_ids, chosen_ids)
        ref_rejected = reference.log_prob(input_ids, rejected_ids)
    
    pi_logratios = policy_chosen - policy_rejected
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
        shortDesc: 'Step-by-step reasoning traces',
        definition: 'CoT forces the model to generate intermediate reasoning steps before the final answer. Zero-Shot CoT ("Let\'s think step by step") works surprisingly well. Auto-CoT generates reasoning chains automatically. Crucial for arithmetic, logic, and complex reasoning tasks.',
        related: ['prompt-engineering', 'planning'],
        tags: ['reasoning', 'technique'],
        codeExample: `# Zero-Shot CoT
prompt = """
Question: A juggler can juggle 6 balls. Half the balls are golf balls.
Half of the golf balls are blue. How many blue golf balls?
Let's think step by step.
"""`

    },
    {
        id: 'tree-of-thoughts',
        name: 'Tree of Thoughts',
        category: 'prompting',
        type: 'technique',
        shortDesc: 'Exploring multiple reasoning paths',
        definition: 'ToT generalizes CoT by exploring multiple reasoning possibilities as a tree search. It generates intermediate thoughts, evaluates them, and backtracks from dead ends. Useful for puzzles, creative writing, and strategic decisions where a single linear path might fail.',
        related: ['chain-of-thought', 'planning'],
        tags: ['reasoning', 'search'],
        codeExample: `# Tree of Thoughts Pattern
def tree_of_thoughts(prompt, llm, depth=3, breadth=3):
    # BFS over thought states
    current_states = [""]
    
    for d in range(depth):
        new_states = []
        for state in current_states:
            # Generate possible next thoughts
            thoughts = llm.generate(f"{prompt}\\nCurrent state: {state}\\nGenerate {breadth} possible next thoughts:")
            
            for thought in parse_thoughts(thoughts):
                new_state = state + thought
                # Evaluate promise of this state
                score = llm.evaluate(new_state)
                new_states.append((score, new_state))
        
        # Keep best branches
        current_states = sorted(new_states, reverse=True)[:breadth]
    
    return current_states[0][1]`
    },
    {
        id: 'prompt-engineering',
        name: 'Prompt Engineering',
        category: 'prompting',
        type: 'core',
        shortDesc: 'Optimizing inputs for model outputs',
        definition: 'Prompt Engineering designs input templates to guide model behavior. Techniques: Few-Shot (examples), Role Prompting (persona), Structured Output Requests (JSON/XML). Advanced methods use generated knowledge (generate facts before answering) or Self-Consistency (sample multiple reasoning paths).',
        related: ['few-shot', 'chain-of-thought'],
        tags: ['fundamental', 'optimization'],
        codeExample: `# Advanced Prompt Template
PROMPT = """
Role: You are a senior engineer.
Task: Write a production-grade function.

Requirements:
- Use type hints
- Include docstrings
- Handle edge cases

Format:
\`\`\`python
# Code here
\`\`\`

Question: {query}
"""`

    },

    // ========== INFRASTRUCTURE SECTION ==========
    {
        id: 'inference',
        name: 'Inference',
        category: 'infrastructure',
        type: 'core',
        shortDesc: 'Serving models for predictions',
        definition: 'Inference optimization balances latency, throughput, and cost. Key techniques: KV Caching (memoization), Continuous Batching (packing requests), Speculative Decoding (draft model guesses). Tools: vLLM (PagedAttention), TensorRT-LLM (NVIDIA), TGI (HuggingFace).',
        related: ['quantization', 'kv-cache'],
        tags: ['production', 'serving'],
        codeExample: `# vLLM Continuous Batching
from vllm import LLM, SamplingParams

llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.2")

params = SamplingParams(max_tokens=128, temperature=0.7)

# Handles continuous batching internally
outputs = llm.generate(prompts, params)`
    },
    {
        id: 'quantization',
        name: 'Quantization',
        category: 'infrastructure',
        type: 'technique',
        shortDesc: 'Compressing models via reduced precision',
        definition: 'Quantization reduces weights from FP16/FP32 to INT8/INT4. PTQ (Post-Training Quantization) is fast but lower quality. QAT (Quantization-Aware Training) preserves accuracy better. GPTQ/AWQ/SpQR are popular algorithms for LLM compression. 4-bit is the sweet spot for efficiency/quality.',
        related: ['inference', 'lora'],
        tags: ['compression', 'efficiency'],
        codeExample: `# GPTQ Quantization
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

quantization_config = GPTQConfig(
    bits=4,
    dataset="c4",
    group_size=128,
    desc_act=False
)

model = AutoModelForCausalLM.from_quantized(
    "TheBloke/Llama-2-7B-GPTQ",
    quantization_config=quantization_config
)`
    },
    {
        id: 'kv-cache',
        name: 'KV Cache',
        category: 'infrastructure',
        type: 'technique',
        shortDesc: 'Caching attention states for speed',
        definition: 'KV Cache stores previous Key/Value projections to avoid recomputation during token generation. Requires O(N) memory. PagedAttention (vLLM) manages KV cache like virtual memory pages, solving fragmentation. Enables 10x+ higher throughput.',
        related: ['inference', 'attention-mechanism'],
        tags: ['optimization', 'memory'],
        codeExample: `# PagedAttention Concept
class PagedKVCache:
    def __init__(self, block_size=16):
        self.blocks = {}  # Virtual memory blocks
        self.block_tables = {}  # Mapping seq_id -> blocks
    
    def allocate(self, seq_id):
        # Allocate blocks on demand
        self.block_tables[seq_id] = []
        
    def append(self, seq_id, new_k, new_v):
        # Write to paged memory
        pass`
    },

    // ========== APPLICATIONS SECTION ==========
    {
        id: 'chatbots',
        name: 'Chatbots',
        category: 'applications',
        type: 'application',
        shortDesc: 'Conversational AI systems',
        definition: 'Chatbots use RAG for knowledge, Tools for actions, and Memory for context. Key challenge: State Management across turns. Long-context models reduce retrieval frequency. Systems must handle intent recognition, safety filters, and graceful failure.',
        related: ['rag', 'agentic-ai'],
        tags: ['conversational', 'interface'],
        codeExample: `# Chatbot State Machine
class ChatBot:
    def __init__(self, llm, retriever):
        self.history = []
    
    def chat(self, message):
        # 1. Retrieve Context
        context = retriever.search(message)
        
        # 2. Build Prompt
        prompt = build_prompt(history, context, message)
        
        # 3. Generate
        response = llm.invoke(prompt)
        
        # 4. Update History
        self.history.append((message, response))
        return response`
    },
    {
        id: 'code-generation',
        name: 'Code Generation',
        category: 'applications',
        type: 'application',
        shortDesc: 'AI-assisted software development',
        definition: 'Code Generation models (Codex, StarCoder, CodeLlama) are trained on code corpora. Context Window is critical (fitting large files). FIM (Fill-In-the-Middle) enables code completion. Agents can iteratively fix compiler errors. RAG with documentation ensures library usage accuracy.',
        related: ['llm', 'tool-use'],
        tags: ['development', 'productivity'],
        codeExample: `# FIM (Fill-In-Middle)
prefix = "def calculate_sum("
suffix = "    return total"
middle = model.generate(f"<PRE> {prefix} <SUF>{suffix} <MID>")
print(prefix + middle + suffix)
# Output: def calculate_sum(numbers):
#             total = 0
#             for n in numbers:
#                 total += n
#         return total`
    }
];

// ==========================================
// BUILD & EXPORT KNOWLEDGE BASE
// ==========================================
const KnowledgeBase = {
    meta: {
        version: '2.0.0',
        lastUpdated: '2025-01-15',
        description: 'Interactive AI Knowledge Graph - Enhanced'
    },
    categories: CategoryData,
    terms: TermData.map(t => createTerm(t))
};

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
        return term.related.map(rId => this.getTerm(rId)).filter(Boolean);
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
    }
};

console.log('KnowledgeBase loaded:', KnowledgeBase.categories.length, 'categories,', KnowledgeBase.terms.length, 'terms');
window.KnowledgeBase = KnowledgeBase;
window.KnowledgeUtils = KnowledgeUtils;
