/**
 * AI Knowledge Base - Extensible Data Structure
 * 
 * To extend:
 * 1. Add new categories to the categories array
 * 2. Add new terms to the terms array with proper relations
 * 3. Use consistent IDs for relationships
 */

const KnowledgeBase = {
    // ==========================================
    // VERSION & METADATA
    // ==========================================
    meta: {
        version: '1.0.0',
        lastUpdated: '2025-01-15',
        description: 'Interactive AI Knowledge Graph'
    },

    // ==========================================
    // CATEGORY DEFINITIONS
    // ==========================================
    categories: [
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
    ],

    // ==========================================
    // TERM DEFINITIONS
    // ==========================================
    terms: [
        // ========== RAG SECTION ==========
        createTerm({
            id: 'rag',
            name: 'RAG',
            fullName: 'Retrieval Augmented Generation',
            category: 'rag',
            type: 'core',
            shortDesc: 'Augmenting LLM responses with retrieved external knowledge',
            definition: `RAG is a technique that combines generative language models with external knowledge retrieval. Instead of relying solely on parametric knowledge stored in model weights, RAG systems retrieve relevant documents from a knowledge base and use them to ground responses in factual, up-to-date information. This significantly reduces hallucinations and enables access to information beyond the training cutoff date.`,
            related: ['vector-database', 'embeddings', 'chunking', 'reranking', 'semantic-search'],
            tags: ['core', 'production', 'knowledge-retrieval'],
            codeExample: `# Basic RAG Pipeline
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# 1. Index documents
vectorstore = Chroma.from_documents(
    documents=chunked_docs,
    embedding=OpenAIEmbeddings()
)

# 2. Retrieve relevant context
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)
docs = retriever.get_relevant_documents(query)

# 3. Generate with context
response = llm.generate(
    prompt=f"Context: {docs}\\nQuestion: {query}"
)`
        }),

        createTerm({
            id: 'vector-database',
            name: 'Vector Database',
            category: 'rag',
            type: 'infrastructure',
            shortDesc: 'Specialized databases for storing and querying vector embeddings',
            definition: `Vector databases are specialized storage systems designed for efficient similarity search over high-dimensional vector embeddings. They use approximate nearest neighbor (ANN) algorithms like HNSW, IVF, or LSH to enable fast retrieval even at scale. Popular options include Pinecone, Weaviate, Chroma, Milvus, Qdrant, and pgvector for PostgreSQL.`,
            related: ['embeddings', 'rag', 'semantic-search', 'hnsw'],
            tags: ['infrastructure', 'storage', 'similarity-search'],
            codeExample: `# Vector Database Operations with Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient(":memory:")

# Create collection
client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(
        size=1536,
        distance=Distance.COSINE
    )
)

# Upsert vectors
client.upsert(
    collection_name="documents",
    points=[{
        "id": "doc1",
        "vector": embedding,
        "payload": {"text": "...", "source": "..."}
    }]
)

# Search
results = client.search(
    collection_name="documents",
    query_vector=query_embedding,
    limit=5
)`
        }),

        createTerm({
            id: 'embeddings',
            name: 'Embeddings',
            category: 'rag',
            type: 'core',
            shortDesc: 'Dense vector representations of text or data',
            definition: `Embeddings are dense vector representations that capture semantic meaning in a continuous space. Text embeddings map words, sentences, or documents to vectors where semantically similar items are close in the vector space. Modern embedding models like OpenAI's text-embedding-3, Cohere embeddings, Voyage AI, or open-source models like Sentence Transformers enable semantic search, clustering, and retrieval.`,
            related: ['vector-database', 'semantic-search', 'rag', 'transformers'],
            tags: ['fundamental', 'representations', 'semantic'],
            codeExample: `# Generate Embeddings
from openai import OpenAI
client = OpenAI()

response = client.embeddings.create(
    model="text-embedding-3-small",
    input="Your text here"
)
embedding = response.data[0].embedding  # 1536 dimensions

# Calculate similarity
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

similarity = cosine_similarity(embedding1, embedding2)`
        }),

        createTerm({
            id: 'chunking',
            name: 'Chunking',
            category: 'rag',
            type: 'technique',
            shortDesc: 'Splitting documents into pieces for retrieval',
            definition: `Chunking is the process of splitting documents into smaller, semantically meaningful pieces for embedding and retrieval. Good chunking strategies balance maintaining context with keeping chunks focused. Methods include fixed-size chunking, recursive character splitting, semantic chunking based on meaning boundaries, and parent-document retrieval that indexes small chunks but retrieves larger context.`,
            related: ['rag', 'embeddings', 'document-processing'],
            tags: ['preprocessing', 'retrieval', 'documents'],
            codeExample: `# Chunking Strategies
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SemanticChunker
)

# Recursive splitting (recommended)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\\n\\n", "\\n", ". ", " ", ""]
)
chunks = splitter.split_text(document)

# Semantic chunking
from langchain.embeddings import OpenAIEmbeddings
semantic = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile"
)
semantic_chunks = semantic.split_text(document)`
        }),

        createTerm({
            id: 'reranking',
            name: 'Reranking',
            category: 'rag',
            type: 'technique',
            shortDesc: 'Re-scoring retrieved documents for relevance',
            definition: `Reranking is a two-stage retrieval technique where initial fast retrieval using vector similarity is followed by a sophisticated cross-encoder model that scores document-query pairs. This significantly improves precision at the cost of latency. Models like Cohere Rerank, BGE Reranker, Voyage Reranker, or ColBERT provide better relevance scoring than pure vector similarity.`,
            related: ['rag', 'semantic-search', 'retrieval-evaluation'],
            tags: ['retrieval', 'optimization', 'precision'],
            codeExample: `# Reranking with Cohere
import cohere
co = cohere.Client("api_key")

# Initial retrieval
initial_docs = vectorstore.similarity_search(query, k=20)

# Rerank
results = co.rerank(
    query=query,
    documents=[doc.page_content for doc in initial_docs],
    top_n=5,
    model="rerank-english-v3.0"
)

# Get top reranked docs
top_docs = [initial_docs[r.index] for r in results.results]`
        }),

        createTerm({
            id: 'semantic-search',
            name: 'Semantic Search',
            category: 'rag',
            type: 'technique',
            shortDesc: 'Finding documents by meaning not keywords',
            definition: `Semantic search uses embeddings to find documents based on meaning rather than exact keyword matches. It enables finding relevant content even when users use different vocabulary. Combined with keyword search in hybrid approaches, it provides robust retrieval handling both vocabulary mismatches and precise term matching.`,
            related: ['embeddings', 'vector-database', 'rag', 'hybrid-search'],
            tags: ['search', 'retrieval', 'semantic'],
            codeExample: `# Hybrid Search Implementation
def hybrid_search(query, vectorstore, bm25_index, alpha=0.5):
    """
    alpha: weight for semantic (1-alpha for keyword)
    """
    # Semantic search
    semantic_results = vectorstore.similarity_search(query, k=20)
    
    # Keyword search
    keyword_results = bm25_index.search(query, k=20)
    
    # Reciprocal Rank Fusion
    def rrf(rank, k=60):
        return 1 / (k + rank)
    
    scores = {}
    for rank, doc in enumerate(semantic_results):
        scores[doc.id] = scores.get(doc.id, 0) + alpha * rrf(rank)
    for rank, doc in enumerate(keyword_results):
        scores[doc.id] = scores.get(doc.id, 0) + (1-alpha) * rrf(rank)
    
    return sorted(scores.items(), key=lambda x: -x[1])`
        }),

        createTerm({
            id: 'hybrid-search',
            name: 'Hybrid Search',
            category: 'rag',
            type: 'technique',
            shortDesc: 'Combining semantic and keyword search',
            definition: `Hybrid search combines semantic vector search with traditional keyword-based search (like BM25) to get the best of both worlds. Semantic search handles vocabulary mismatch and conceptual similarity, while keyword search ensures exact term matching. Fusion methods like Reciprocal Rank Fusion (RRF) combine the rankings effectively.`,
            related: ['semantic-search', 'reranking', 'rag'],
            tags: ['search', 'retrieval', 'fusion'],
            codeExample: `# Hybrid Search with RRF Fusion
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# Create retrievers
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 10

# Ensemble with RRF
ensemble = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6],  # Keyword, Semantic
    c=60  # RRF constant
)

results = ensemble.get_relevant_documents(query)`
        }),

        // ========== AGENTIC SECTION ==========
        createTerm({
            id: 'agentic-ai',
            name: 'Agentic AI',
            category: 'agentic',
            type: 'core',
            shortDesc: 'AI systems that autonomously plan and execute tasks',
            definition: `Agentic AI refers to AI systems capable of autonomous goal-directed behavior. Unlike single-prompt interactions, agentic systems can plan multi-step actions, use external tools, maintain memory across interactions, and self-correct. Key frameworks include LangChain Agents, AutoGen, CrewAI, and the ReAct (Reasoning + Acting) paradigm.`,
            related: ['tool-use', 'planning', 'memory-agents', 'multi-agent', 'react-pattern'],
            tags: ['core', 'autonomous', 'framework'],
            codeExample: `# Agentic Loop with ReAct Pattern
def agent_loop(query, llm, tools, max_iterations=10):
    history = []
    
    for i in range(max_iterations):
        # Generate thought and action
        response = llm.invoke(f"""
        Question: {query}
        History: {history}
        
        Think step by step. Available tools: {tools}
        Output format: Thought: ... Action: tool_name(args)
        """)
        
        action = parse_action(response)
        
        if action.type == "FINISH":
            return action.answer
        
        # Execute tool
        observation = execute_tool(action.name, action.args)
        history.append(f"Thought: {action.thought}\\nAction: {action}\\nObservation: {observation}")
    
    return "Max iterations reached"`
        }),

        createTerm({
            id: 'tool-use',
            name: 'Tool Use',
            category: 'agentic',
            type: 'core',
            shortDesc: 'Enabling LLMs to interact with external systems',
            definition: `Tool use (or function calling) allows LLMs to interact with external systems by generating structured outputs that trigger predefined functions. The model outputs parameters in a specified format, which are then executed by the runtime environment. This enables searching the web, querying databases, calling APIs, and any programmatic action.`,
            related: ['agentic-ai', 'function-calling', 'mcp', 'tool-definition'],
            tags: ['agent', 'integration', 'external'],
            codeExample: `# Tool Definition & Use with OpenAI
tools = [{
    "type": "function",
    "function": {
        "name": "search_web",
        "description": "Search the web for current information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    }
}]

# Model call with tools
response = client.chat.completions.create(
    model="gpt-4-turbo",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=tools
)

# Execute tool if called
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    result = search_web(json.loads(tool_call.function.arguments)["query"])`
        }),

        createTerm({
            id: 'planning',
            name: 'Planning',
            category: 'agentic',
            type: 'technique',
            shortDesc: 'Breaking complex goals into steps',
            definition: `Planning in agentic AI involves decomposing complex goals into actionable steps. Techniques include chain-of-thought for implicit planning, explicit plan generation followed by execution, and dynamic replanning when execution fails. Advanced approaches like Tree of Thoughts explore multiple plan branches before committing.`,
            related: ['agentic-ai', 'tool-use', 'reasoning', 'chain-of-thought'],
            tags: ['agent', 'reasoning', 'decomposition'],
            codeExample: `# Planning Agent Pattern
def plan_and_execute(goal, llm, tools):
    # Generate plan
    plan = llm.invoke(f"""
    Goal: {goal}
    Create a step-by-step plan to achieve this goal.
    Each step should be a specific action.
    
    Format:
    1. [action_type] description
    2. [action_type] description
    ...
    """)
    
    steps = parse_plan(plan)
    results = []
    
    for step in steps:
        result = execute_step(step, tools)
        results.append(result)
        
        # Replan on failure
        if result.failed:
            new_plan = llm.invoke(f"""
            Original plan failed at: {step}
            Error: {result.error}
            Previous results: {results}
            Generate a new plan to achieve: {goal}
            """)
            steps = parse_plan(new_plan)
    
    return synthesize(results)`
        }),

        createTerm({
            id: 'memory-agents',
            name: 'Agent Memory',
            category: 'agentic',
            type: 'technique',
            shortDesc: 'Persisting context across agent interactions',
            definition: `Agent memory systems enable AI to retain information across interactions. Types include: short-term memory (conversation history), long-term memory (persistent facts), episodic memory (past interaction sequences), and working memory (current task context). Vector databases often power semantic memory retrieval.`,
            related: ['agentic-ai', 'vector-database', 'context-window', 'rag'],
            tags: ['agent', 'persistence', 'context'],
            codeExample: `# Agent Memory System
class AgentMemory:
    def __init__(self, llm):
        self.short_term = []  # Recent messages
        self.long_term = VectorStore()  # Semantic memory
        self.working = {}  # Task state
        self.llm = llm
    
    def add(self, message):
        self.short_term.append(message)
        
        # Summarize when too long
        if len(self.short_term) > 10:
            summary = self.llm.invoke(
                f"Summarize: {self.short_term[:5]}"
            )
            self.long_term.add(summary, metadata={"type": "summary"})
            self.short_term = self.short_term[5:]
    
    def recall(self, query, k=5):
        # Semantic search over memories
        return self.long_term.search(query, k=k)
    
    def get_context(self):
        return {
            "recent": self.short_term,
            "relevant": self.recall(self.short_term[-1] if self.short_term else ""),
            "working": self.working
        }`
        }),

        createTerm({
            id: 'multi-agent',
            name: 'Multi-Agent Systems',
            category: 'agentic',
            type: 'technique',
            shortDesc: 'Coordinating multiple AI agents',
            definition: `Multi-agent systems coordinate multiple specialized AI agents to solve complex problems. Agents can have different roles (researcher, writer, reviewer), share common memory, and communicate through structured messages. Frameworks like CrewAI, AutoGen, and LangGraph enable orchestrating agent teams with defined workflows.`,
            related: ['agentic-ai', 'planning', 'tool-use', 'communication-protocols'],
            tags: ['agent', 'coordination', 'teams'],
            codeExample: `# Multi-Agent with CrewAI
from crewai import Agent, Task, Crew

researcher = Agent(
    role="Senior Researcher",
    goal="Find comprehensive, accurate information",
    backstory="Expert researcher with attention to detail",
    tools=[search_tool, scrape_tool],
    llm="gpt-4"
)

writer = Agent(
    role="Content Writer",
    goal="Create engaging, accurate content",
    backstory="Professional writer with technical expertise",
    tools=[write_tool],
    llm="gpt-4"
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[
        Task(description="Research AI trends", agent=researcher),
        Task(description="Write article on findings", agent=writer)
    ],
    verbose=True
)

result = crew.kickoff()`
        }),

        createTerm({
            id: 'react-pattern',
            name: 'ReAct Pattern',
            category: 'agentic',
            type: 'technique',
            shortDesc: 'Reasoning and Acting interleaving',
            definition: `ReAct (Reasoning + Acting) is a prompting paradigm that interleaves reasoning traces with actions. The model generates thoughts, decides on actions, observes results, and continues. This creates interpretable reasoning paths and enables dynamic decision-making. It's the foundation of most modern agent frameworks.`,
            related: ['agentic-ai', 'chain-of-thought', 'tool-use', 'planning'],
            tags: ['agent', 'reasoning', 'paradigm'],
            codeExample: `# ReAct Prompt Template
REACT_PROMPT = """
Answer the question as best you can. You have access to these tools:

{tools}

Use the following format:

Question: the input question
Thought: think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (repeat Thought/Action/Action Input/Observation)
Thought: I now know the final answer
Final Answer: the final answer to the question

Begin!

Question: {input}
Thought: {agent_scratchpad}
"""`
        }),

        // ========== MCP SECTION ==========
        createTerm({
            id: 'mcp',
            name: 'MCP',
            fullName: 'Model Context Protocol',
            category: 'mcp',
            type: 'core',
            shortDesc: 'Open protocol for connecting AI to systems',
            definition: `The Model Context Protocol (MCP) is an open standard that enables AI assistants to connect to external systems uniformly. It provides a standardized way to expose resources (files, data), prompts (reusable templates), and tools (functions) to AI models. MCP servers can be built for any data source, and any MCP-compatible client can use them.`,
            related: ['tool-use', 'function-calling', 'mcp-resources', 'mcp-tools'],
            tags: ['protocol', 'standard', 'integration'],
            codeExample: `# MCP Server Implementation
from mcp.server import Server
from mcp.types import Tool, Resource

server = Server("my-server")

@server.tool()
def search_database(query: str) -> str:
    """Search the internal database for information."""
    return db.search(query)

@server.resource("docs://{path}")
def get_document(path: str) -> str:
    """Access document files by path."""
    return open(f"documents/{path}").read()

# Client usage
async with Client() as client:
    await client.connect_to_server("my-server")
    tools = await client.list_tools()
    result = await client.call_tool("search_database", {"query": "AI"})`
        }),

        createTerm({
            id: 'mcp-resources',
            name: 'MCP Resources',
            category: 'mcp',
            type: 'technique',
            shortDesc: 'URI-based data access through MCP',
            definition: `MCP Resources provide a URI-based way to expose read-only data to AI models. Resources can represent files, database records, API responses, or any data. They support templates for dynamic URIs and subscriptions for real-time updates. Resources are distinct from tools as they are read-only.`,
            related: ['mcp', 'mcp-tools', 'mcp-prompts'],
            tags: ['mcp', 'data', 'read-only'],
            codeExample: `# MCP Resource Definition
from mcp.server import Server
from mcp.types import Resource, ResourceTemplate

server = Server("database-server")

# Static resource
@server.resource("config://settings")
def get_config() -> dict:
    return load_config()

# Dynamic template
@server.resource("users://{user_id}/profile")
def get_user_profile(user_id: str) -> dict:
    return db.users.find(user_id)

# List all resources
@server.list_resources()
def list_resources():
    return [
        Resource(uri="config://settings", name="Settings"),
        ResourceTemplate(
            uriTemplate="users://{user_id}/profile",
            name="User Profile"
        )
    ]`
        }),

        createTerm({
            id: 'mcp-tools',
            name: 'MCP Tools',
            category: 'mcp',
            type: 'technique',
            shortDesc: 'Functions exposed through MCP',
            definition: `MCP Tools are functions that AI models can call to perform actions. Unlike resources, tools can modify state, make API calls, or execute any operation. Tools include JSON Schema for parameters enabling type-safe invocation. The model decides when to call tools based on user intent and tool descriptions.`,
            related: ['mcp', 'mcp-resources', 'tool-use'],
            tags: ['mcp', 'functions', 'actions'],
            codeExample: `# MCP Tool with Schema
from mcp.server import Server
from mcp.types import Tool

server = Server("email-server")

@server.tool()
def send_email(to: str, subject: str, body: str) -> dict:
    """
    Send an email to a recipient.
    
    Args:
        to: Recipient email address
        subject: Email subject line
        body: Email body text (markdown supported)
    
    Returns:
        Confirmation with message ID
    """
    return email_service.send(to, subject, body)

# Tool schema is auto-generated
# {
#   "name": "send_email",
#   "inputSchema": {
#     "type": "object",
#     "properties": {
#       "to": {"type": "string"},
#       "subject": {"type": "string"},
#       "body": {"type": "string"}
#     },
#     "required": ["to", "subject", "body"]
#   }
# }`
        }),

        // ========== ARCHITECTURE SECTION ==========
        createTerm({
            id: 'transformers',
            name: 'Transformers',
            category: 'architecture',
            type: 'core',
            shortDesc: 'Foundational architecture behind modern LLMs',
            definition: `Transformers are neural network architectures using self-attention mechanisms to process sequences in parallel. Introduced in "Attention is All You Need" (2017), they efficiently capture long-range dependencies. All major LLMs (GPT, Claude, Llama) are transformer-based, using encoder-only (BERT), decoder-only (GPT), or encoder-decoder (T5) variants.`,
            related: ['attention-mechanism', 'llm', 'positional-encoding', 'self-attention'],
            tags: ['fundamental', 'architecture', 'deep-learning'],
            codeExample: `# Transformer Block (PyTorch)
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        # Self-attention with pre-norm
        attn_out, _ = self.attention(
            self.norm1(x), self.norm1(x), self.norm1(x),
            attn_mask=mask, need_weights=False
        )
        x = x + attn_out
        
        # Feed-forward with pre-norm
        x = x + self.ff(self.norm2(x))
        return x`
        }),

        createTerm({
            id: 'attention-mechanism',
            name: 'Attention Mechanism',
            category: 'architecture',
            type: 'core',
            shortDesc: 'Allowing models to focus on relevant inputs',
            definition: `Attention mechanisms allow neural networks to dynamically focus on relevant parts of input when producing each output. Self-attention computes relationships between all sequence positions. Multi-head attention runs multiple attention operations in parallel, capturing different relationship types. Scaled dot-product attention is the standard implementation.`,
            related: ['transformers', 'self-attention', 'multi-head-attention'],
            tags: ['fundamental', 'architecture', 'mechanism'],
            codeExample: `# Scaled Dot-Product Attention
import torch
import math

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    
    # Compute attention scores
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    # Apply mask (for padding or causal attention)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Softmax
    attn_weights = torch.softmax(scores, dim=-1)
    
    if dropout is not None:
        attn_weights = dropout(attn_weights)
    
    # Weighted sum of values
    return torch.matmul(attn_weights, value), attn_weights`
        }),

        createTerm({
            id: 'llm',
            name: 'LLM',
            fullName: 'Large Language Model',
            category: 'architecture',
            type: 'core',
            shortDesc: 'Foundation models trained on massive text',
            definition: `Large Language Models are neural networks trained on vast text corpora to predict the next token. Through this objective, they learn language understanding, world knowledge, reasoning, and generation. Modern LLMs like GPT-4, Claude, Llama, and Mistral range from 7B to trillions of parameters and can be adapted via fine-tuning or prompting.`,
            related: ['transformers', 'tokenization', 'pre-training', 'emergent-abilities'],
            tags: ['fundamental', 'foundation-model', 'generation'],
            codeExample: `# LLM Text Generation
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

def generate(prompt, max_new_tokens=100, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.9,
        do_sample=True
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)`
        }),

        createTerm({
            id: 'tokenization',
            name: 'Tokenization',
            category: 'architecture',
            type: 'technique',
            shortDesc: 'Converting text to model tokens',
            definition: `Tokenization converts raw text into tokens that models process. Modern LLMs use subword tokenization (BPE, WordPiece, SentencePiece) balancing vocabulary size with sequence length. Tokens are mapped to embedding vectors. Different tokenizers produce different counts for the same text, affecting cost and context usage.`,
            related: ['llm', 'embeddings', 'context-window'],
            tags: ['preprocessing', 'fundamentals', 'encoding'],
            codeExample: `# Tokenization with tiktoken
import tiktoken

# Model-specific tokenizers
enc = tiktoken.encoding_for_model("gpt-4")

text = "Hello, world!"
tokens = enc.encode(text)  # [9906, 11, 1917, 0]
decoded = enc.decode(tokens)  # "Hello, world!"

# Count tokens for API usage
def count_tokens(text, model="gpt-4"):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

# Token-aware truncation
def truncate(text, max_tokens, model="gpt-4"):
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)[:max_tokens]
    return enc.decode(tokens)`
        }),

        createTerm({
            id: 'context-window',
            name: 'Context Window',
            category: 'architecture',
            type: 'technique',
            shortDesc: 'Maximum sequence length for processing',
            definition: `The context window is the maximum number of tokens a model can process in a single forward pass, including both input and output. Larger windows enable longer documents but increase computational cost. Techniques like RoPE scaling, ALiBi, and Ring Attention help extend effective context length.`,
            related: ['llm', 'tokenization', 'attention-mechanism', 'kv-cache'],
            tags: ['limitations', 'architecture', 'performance'],
            codeExample: `# Context Window Management
class ContextManager:
    def __init__(self, max_tokens=4096, reserve_for_output=500):
        self.max_tokens = max_tokens
        self.reserve = reserve_for_output
        self.messages = []
    
    def add_message(self, role, content, tokenizer):
        new_tokens = len(tokenizer.encode(content))
        current = sum(len(tokenizer.encode(m["content"])) for m in self.messages)
        
        # Truncate old messages if needed
        while current + new_tokens > self.max_tokens - self.reserve:
            removed = self.messages.pop(0)
            current -= len(tokenizer.encode(removed["content"]))
        
        self.messages.append({"role": role, "content": content})
        return len(self.messages)`
        }),

        // ========== TRAINING SECTION ==========
        createTerm({
            id: 'pre-training',
            name: 'Pre-training',
            category: 'training',
            type: 'core',
            shortDesc: 'Initial training on massive datasets',
            definition: `Pre-training is the foundational phase where models learn from massive unlabeled text corpora. The objective is typically next-token prediction, teaching language patterns, world knowledge, and reasoning. Pre-training requires enormous compute (thousands of GPUs for months) and produces base models that can be fine-tuned.`,
            related: ['llm', 'fine-tuning', 'next-token-prediction', 'scaling-laws'],
            tags: ['training', 'foundation', 'compute'],
            codeExample: `# Pre-training Loop (Conceptual)
def pretrain(model, dataloader, optimizer, steps, grad_accum=4):
    model.train()
    optimizer.zero_grad()
    
    for step, batch in enumerate(dataloader):
        # Next token prediction
        input_ids = batch["input_ids"]
        labels = batch["labels"]  # Shifted input_ids
        
        # Forward pass
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss / grad_accum
        
        # Backward pass
        loss.backward()
        
        # Gradient accumulation
        if (step + 1) % grad_accum == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item() * grad_accum:.4f}")
        
        if step >= steps:
            break`
        }),

        createTerm({
            id: 'fine-tuning',
            name: 'Fine-tuning',
            category: 'training',
            type: 'technique',
            shortDesc: 'Adapting pre-trained models for tasks',
            definition: `Fine-tuning adapts a pre-trained model to specific tasks by continuing training on labeled data. Methods include full fine-tuning (updating all weights), parameter-efficient fine-tuning (PEFT) like LoRA that only updates small adapter modules, and instruction tuning. Fine-tuning is much cheaper than pre-training.`,
            related: ['pre-training', 'lora', 'instruction-tuning', 'rlhf'],
            tags: ['training', 'adaptation', 'specialization'],
            codeExample: `# Fine-tuning with LoRA (PEFT)
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16
)

# LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,  # Low rank
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 4M || all params: 7B || trainable%: 0.06%`
        }),

        createTerm({
            id: 'rlhf',
            name: 'RLHF',
            fullName: 'Reinforcement Learning from Human Feedback',
            category: 'training',
            type: 'technique',
            shortDesc: 'Aligning models with human preferences',
            definition: `RLHF aligns language models with human preferences by training a reward model on human comparisons, then optimizing the LLM to maximize this reward. Process: (1) collect human rankings of outputs, (2) train a reward model to predict preferences, (3) use PPO to optimize the LLM. This makes models more helpful and safe.`,
            related: ['fine-tuning', 'dpo', 'alignment', 'reward-model'],
            tags: ['alignment', 'training', 'human-feedback'],
            codeExample: `# RLHF Pipeline (Conceptual)

# Step 1: Train Reward Model
def train_reward_model(reward_model, comparison_data):
    for prompt, chosen, rejected in comparison_data:
        chosen_score = reward_model(prompt, chosen)
        rejected_score = reward_model(prompt, rejected)
        
        # Bradley-Terry loss
        loss = -torch.log(torch.sigmoid(chosen_score - rejected_score))
        loss.backward()

# Step 2: PPO Optimization
def ppo_step(policy, ref_policy, reward_model, prompt, ppo_config):
    response = policy.generate(prompt)
    reward = reward_model(prompt, response)
    
    # PPO clipped objective
    ratio = policy.log_prob(response) / ref_policy.log_prob(response)
    clipped = torch.clamp(ratio, 1-ppo_config.epsilon, 1+ppo_config.epsilon)
    
    loss = -torch.min(ratio * reward, clipped * reward)
    loss.backward()`
        }),

        createTerm({
            id: 'dpo',
            name: 'DPO',
            fullName: 'Direct Preference Optimization',
            category: 'training',
            type: 'technique',
            shortDesc: 'Simpler alternative to RLHF',
            definition: `DPO is a simpler alternative to RLHF that directly optimizes the policy using preference data without training a separate reward model. It reparameterizes the reward function in terms of the policy, enabling direct optimization with a simple classification loss on preferred vs rejected responses.`,
            related: ['rlhf', 'fine-tuning', 'alignment'],
            tags: ['alignment', 'training', 'simplified'],
            codeExample: `# DPO Loss Implementation
def dpo_loss(policy, ref_policy, prompt, chosen, rejected, beta=0.1):
    """
    Direct Preference Optimization loss
    beta: controls KL constraint strength
    """
    # Get log probabilities from policy
    chosen_logp = policy.log_prob(prompt, chosen)
    rejected_logp = policy.log_prob(prompt, rejected)
    
    # Reference model log probs (frozen)
    with torch.no_grad():
        ref_chosen_logp = ref_policy.log_prob(prompt, chosen)
        ref_rejected_logp = ref_policy.log_prob(prompt, rejected)
    
    # Implicit reward
    chosen_reward = beta * (chosen_logp - ref_chosen_logp)
    rejected_reward = beta * (rejected_logp - ref_rejected_logp)
    
    # DPO loss
    loss = -torch.log(torch.sigmoid(chosen_reward - rejected_reward)).mean()
    
    return loss`
        }),

        createTerm({
            id: 'lora',
            name: 'LoRA',
            fullName: 'Low-Rank Adaptation',
            category: 'training',
            type: 'technique',
            shortDesc: 'Parameter-efficient fine-tuning',
            definition: `LoRA enables efficient fine-tuning by adding small trainable rank-decomposition matrices to existing weights. Instead of updating full weight matrix W, LoRA learns smaller matrices A and B such that W' = W + BA. This reduces trainable parameters by 10,000x while maintaining comparable performance.`,
            related: ['fine-tuning', 'qlora', 'peft'],
            tags: ['efficient', 'fine-tuning', 'adaptation'],
            codeExample: `# LoRA Implementation
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=16, alpha=32):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Original weight (frozen)
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) * 0.01,
            requires_grad=False
        )
        
        # LoRA matrices (trainable)
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
    def forward(self, x):
        # W'x = Wx + (B @ A) @ x * scaling
        result = x @ self.weight.T
        result += (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return result`
        }),

        // ========== PROMPTING SECTION ==========
        createTerm({
            id: 'prompt-engineering',
            name: 'Prompt Engineering',
            category: 'prompting',
            type: 'core',
            shortDesc: 'Crafting effective LLM inputs',
            definition: `Prompt engineering is designing inputs to elicit desired outputs from language models. Techniques include clear instruction specification, providing examples (few-shot), breaking down tasks (chain-of-thought), and iterative refinement. Good prompting dramatically improves model performance without training.`,
            related: ['chain-of-thought', 'few-shot', 'system-prompt', 'prompt-templates'],
            tags: ['fundamental', 'techniques', 'optimization'],
            codeExample: `# Prompt Engineering Best Practices

# 1. Be Specific and Clear
vague = "Write about AI"
specific = """
Write a 500-word blog post about AI in healthcare.
Target audience: Healthcare professionals
Tone: Professional but accessible
Include: 3 specific hospital AI applications
Structure: Introduction, 3 sections, conclusion
"""

# 2. Role-Based Prompting
role_prompt = """
You are an expert Python developer specializing in data pipelines.
Provide clean, well-documented code with type hints.
Include error handling and logging.
Explain any trade-offs in your approach.
"""

# 3. Structured Output
structured_prompt = """
Extract key information in JSON format:
{
  "entities": [...],
  "relationships": [...],
  "summary": "..."
}

Text: {input}
"""`
        }),

        createTerm({
            id: 'chain-of-thought',
            name: 'Chain of Thought',
            category: 'prompting',
            type: 'technique',
            shortDesc: 'Step-by-step reasoning decomposition',
            definition: `Chain-of-Thought (CoT) prompting improves reasoning by asking models to think step-by-step before giving final answers. This dramatically improves performance on math, logic, and complex reasoning. Zero-shot CoT adds "Let's think step by step" while few-shot CoT provides worked examples with reasoning.`,
            related: ['prompt-engineering', 'reasoning', 'tree-of-thoughts', 'planning'],
            tags: ['reasoning', 'technique', 'prompting'],
            codeExample: `# Chain-of-Thought Prompting

# Zero-shot CoT
zero_shot_cot = """
Question: A store has 23 apples. They sell 15 and receive 8 more.
How many apples do they have now?

Let's think step by step.
"""

# Few-shot CoT
few_shot_cot = """
Question: Roger has 5 tennis balls. He buys 2 cans of 3 balls each. How many?
Answer: Roger started with 5 balls. 2 cans of 3 balls each is 6 balls.
5 + 6 = 11. The answer is 11.

Question: A store has 23 apples. They sell 15 and receive 8 more. How many?
Answer:
"""`
        }),

        createTerm({
            id: 'few-shot',
            name: 'Few-Shot Prompting',
            category: 'prompting',
            type: 'technique',
            shortDesc: 'Providing examples to guide behavior',
            definition: `Few-shot prompting provides input-output examples before the actual query. This helps models understand desired format, style, and reasoning without weight updates. Research shows 2-6 examples often work best, and example selection and ordering significantly impact performance.`,
            related: ['prompt-engineering', 'in-context-learning', 'prompt-templates'],
            tags: ['technique', 'examples', 'in-context'],
            codeExample: `# Few-Shot Prompting Patterns

# Classification
classification = """
Classify sentiment:

Text: "This product exceeded my expectations!"
Sentiment: Positive

Text: "Worst purchase ever made."
Sentiment: Negative

Text: "It's okay, nothing special."
Sentiment: Neutral

Text: "Absolutely love it!"
Sentiment:
"""

# Format Following
format_following = """
Extract entities as JSON:

Text: "John works at Google in NYC."
Entities: {"person": "John", "company": "Google", "location": "NYC"}

Text: "Sarah is CEO of Tesla, based in Austin."
Entities: {"person": "Sarah", "company": "Tesla", "location": "Austin"}

Text: "Elon announced SpaceX launch from Florida."
Entities:
"""`
        }),

        createTerm({
            id: 'system-prompt',
            name: 'System Prompt',
            category: 'prompting',
            type: 'technique',
            shortDesc: 'Defining model behavior and constraints',
            definition: `System prompts are instructions that define model behavior, personality, capabilities, and constraints. They're processed before user messages and persist through conversations. Good system prompts specify role, tone, output format, what to do/avoid, and safety guidelines. They're crucial for consistent, controlled AI behavior.`,
            related: ['prompt-engineering', 'prompt-templates', 'prompt-injection'],
            tags: ['configuration', 'behavior', 'instructions'],
            codeExample: `# System Prompt Examples

# Role Definition
system_assistant = """
You are a helpful AI assistant specializing in Python.
- Provide clear, well-commented code examples
- Explain concepts at an intermediate level
- Suggest best practices and common pitfalls
- If unsure, acknowledge limitations
"""

# Structured Output
system_json = """
You are a data analyst. Always respond in valid JSON:
{
  "analysis": "brief summary",
  "insights": ["insight1", "insight2"],
  "recommendations": ["rec1", "rec2"],
  "confidence": "high|medium|low"
}
No text outside JSON.
"""

# API Usage
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": system_assistant},
        {"role": "user", "content": user_query}
    ]
)`
        }),

        // ========== INFRASTRUCTURE SECTION ==========
        createTerm({
            id: 'inference',
            name: 'Inference',
            category: 'infrastructure',
            type: 'core',
            shortDesc: 'Running trained models for predictions',
            definition: `Inference is running a trained model to generate outputs. Unlike training, inference doesn't update weights. Key considerations: latency (time per request), throughput (requests/second), cost, and accuracy. Optimization techniques include quantization, batching, caching, and specialized hardware (GPUs, TPUs).`,
            related: ['quantization', 'kv-cache', 'model-serving', 'latency'],
            tags: ['deployment', 'production', 'optimization'],
            codeExample: `# Inference Optimization

# 1. Batching
def batch_inference(model, inputs, batch_size=32):
    results = []
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i:i+batch_size]
        outputs = model.generate(batch)
        results.extend(outputs)
    return results

# 2. Continuous Batching (vLLM style)
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-2-7b-hf")
params = SamplingParams(max_tokens=100, temperature=0.7)

# Handles batching automatically
outputs = llm.generate(prompts, params)

# 3. Quantized Inference
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b",
    quantization_config=bnb_config
)`
        }),

        createTerm({
            id: 'quantization',
            name: 'Quantization',
            category: 'infrastructure',
            type: 'technique',
            shortDesc: 'Reducing model precision for efficiency',
            definition: `Quantization reduces memory and compute requirements by lowering numerical precision (FP16 to INT8 or INT4). Can be done post-training (PTQ) or during training (QAT). Modern methods like GPTQ, AWQ, and GGUF maintain quality while reducing size 2-4x and enabling faster inference.`,
            related: ['inference', 'model-compression', 'qlora'],
            tags: ['optimization', 'efficiency', 'deployment'],
            codeExample: `# Quantization Methods

# 1. bitsandbytes for LLMs
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b",
    quantization_config=bnb_config,
    device_map="auto"
)

# 2. GPTQ for efficient serving
from auto_gptq import AutoGPTQForCausalLM

model = AutoGPTQForCausalLM.from_quantized(
    "TheBloke/Llama-2-7B-GPTQ",
    device_map="auto"
)

# 3. GGUF for CPU/GPU hybrid
# Use llama.cpp CLI:
# ./main -m model.gguf -p "prompt" -n 100`
        }),

        createTerm({
            id: 'kv-cache',
            name: 'KV Cache',
            fullName: 'Key-Value Cache',
            category: 'infrastructure',
            type: 'technique',
            shortDesc: 'Caching attention states for faster generation',
            definition: `The KV Cache stores computed key and value tensors during autoregressive generation. Instead of recomputing attention for all previous tokens each step, the model reuses cached values and only computes for new tokens. This reduces generation from O(n²) to O(n) complexity, dramatically speeding up inference at memory cost.`,
            related: ['inference', 'attention-mechanism', 'context-window'],
            tags: ['optimization', 'memory', 'generation'],
            codeExample: `# KV Cache Implementation

class KVCache:
    def __init__(self, n_layers, n_heads, head_dim, max_seq_len, dtype=torch.float16):
        self.cache = [
            torch.zeros(1, n_heads, max_seq_len, head_dim, dtype=dtype)
            for _ in range(2 * n_layers)  # K and V per layer
        ]
        self.seq_len = 0
    
    def update(self, layer_idx, new_k, new_v):
        # Append new keys and values
        k_cache = self.cache[2 * layer_idx]
        v_cache = self.cache[2 * layer_idx + 1]
        
        k_cache[:, :, self.seq_len:self.seq_len+1] = new_k
        v_cache[:, :, self.seq_len:self.seq_len+1] = new_v
        
        # Return full K, V up to current position
        return (
            k_cache[:, :, :self.seq_len+1],
            v_cache[:, :, :self.seq_len+1]
        )
    
    def increment(self):
        self.seq_len += 1
    
    def clear(self):
        self.seq_len = 0
        for tensor in self.cache:
            tensor.zero_()`
        }),

        // ========== APPLICATIONS SECTION ==========
        createTerm({
            id: 'chatbots',
            name: 'Chatbots',
            category: 'applications',
            type: 'application',
            shortDesc: 'Conversational AI interfaces',
            definition: `Chatbots are conversational AI systems that interact through natural language dialogue. Modern chatbots use LLMs for conversation combined with RAG for knowledge, tools for actions, and memory for context. They range from simple Q&A to sophisticated agents completing complex multi-turn tasks.`,
            related: ['rag', 'agentic-ai', 'system-prompt', 'memory-agents'],
            tags: ['application', 'conversation', 'interface'],
            codeExample: `# Chatbot Architecture
class Chatbot:
    def __init__(self, llm, rag, tools, memory):
        self.llm = llm
        self.rag = rag
        self.tools = tools
        self.memory = memory
    
    async def respond(self, user_input):
        # Store user message
        self.memory.add({"role": "user", "content": user_input})
        
        # Retrieve context
        context = await self.rag.retrieve(user_input)
        
        # Check for tool needs
        if self.needs_tool(user_input):
            tool_result = await self.tools.execute(user_input)
            context += f"\\nTool Result: {tool_result}"
        
        # Build messages
        messages = [
            {"role": "system", "content": self.system_prompt},
            *self.memory.get_context(),
            {"role": "user", "content": f"Context: {context}\\nQuery: {user_input}"}
        ]
        
        # Generate response
        response = await self.llm.generate(messages)
        
        # Store and return
        self.memory.add({"role": "assistant", "content": response})
        return response`
        }),

        createTerm({
            id: 'code-generation',
            name: 'Code Generation',
            category: 'applications',
            type: 'application',
            shortDesc: 'AI-powered code writing assistance',
            definition: `Code generation uses LLMs trained on code to write, complete, explain, and debug software. Models like GitHub Copilot, CodeLlama, and StarCoder can generate code from natural language, complete partial code, write tests, and translate between languages. They integrate into IDEs for real-time assistance.`,
            related: ['llm', 'fine-tuning', 'tool-use'],
            tags: ['application', 'development', 'productivity'],
            codeExample: `# Code Generation System
class CodeAssistant:
    def __init__(self, model, language="python"):
        self.model = model
        self.language = language
    
    def generate(self, prompt, context=""):
        full_prompt = f"""
        Language: {self.language}
        Context: {context}
        
        Task: {prompt}
        
        Generate clean, well-commented code with error handling.
        """
        return self.model.generate(full_prompt)
    
    def complete(self, code_prefix):
        return self.model.generate(code_prefix, max_tokens=200)
    
    def explain(self, code):
        return self.model.generate(f"Explain this code:\\n```\\n{code}\\n```")
    
    def test(self, code):
        return self.model.generate(f"Write unit tests for:\\n```\\n{code}\\n```")
    
    def review(self, code):
        return self.model.generate(f"Review this code for issues:\\n```\\n{code}\\n```")`
        })
    ]
};

// ==========================================
// HELPER FUNCTION FOR TERM CREATION
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
        importance: config.importance || 0 // For sorting/weighting
    };
}

// ==========================================
// EXTENSION UTILITIES
// ==========================================
const KnowledgeUtils = {
    // Add a new category
    addCategory(category) {
        if (!category.id || !category.name) {
            console.error('Category must have id and name');
            return false;
        }
        if (KnowledgeBase.categories.find(c => c.id === category.id)) {
            console.error('Category already exists:', category.id);
            return false;
        }
        KnowledgeBase.categories.push({
            id: category.id,
            name: category.name,
            fullName: category.fullName || category.name,
            color: category.color || '#6b7280',
            description: category.description || ''
        });
        return true;
    },

    // Add a new term
    addTerm(termConfig) {
        if (!termConfig.id || !termConfig.name) {
            console.error('Term must have id and name');
            return false;
        }
        if (KnowledgeBase.terms.find(t => t.id === termConfig.id)) {
            console.error('Term already exists:', termConfig.id);
            return false;
        }
        KnowledgeBase.terms.push(createTerm(termConfig));
        return true;
    },

    // Get term by ID
    getTerm(id) {
        return KnowledgeBase.terms.find(t => t.id === id);
    },

    // Get terms by category
    getTermsByCategory(categoryId) {
        return KnowledgeBase.terms.filter(t => t.category === categoryId);
    },

    // Get related terms
    getRelatedTerms(termId) {
        const term = this.getTerm(termId);
        if (!term || !term.related) return [];
        return term.related.map(rId => this.getTerm(rId)).filter(Boolean);
    },

    // Search terms
    searchTerms(query) {
        const q = query.toLowerCase();
        return KnowledgeBase.terms.filter(t => 
            t.name.toLowerCase().includes(q) ||
            t.shortDesc.toLowerCase().includes(q) ||
            t.fullName.toLowerCase().includes(q) ||
            t.tags.some(tag => tag.toLowerCase().includes(q))
        );
    },

    // Get statistics
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

    // Export for backup
    export() {
        return JSON.stringify(KnowledgeBase, null, 2);
    },

    // Import from backup
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

// Export for use in other modules
window.KnowledgeBase = KnowledgeBase;
window.KnowledgeUtils = KnowledgeUtils;
