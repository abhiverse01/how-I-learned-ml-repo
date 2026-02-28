/**
 * AI Knowledge Base - Extensible Data Structure
 * 
 * FIXED: createTerm function must be defined BEFORE it's called
 */

// ==========================================
// HELPER FUNCTION - DEFINED FIRST
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
        definition: 'RAG is a technique that combines generative language models with external knowledge retrieval. Instead of relying solely on parametric knowledge stored in model weights, RAG systems retrieve relevant documents from a knowledge base and use them to ground responses in factual, up-to-date information. This significantly reduces hallucinations and enables access to information beyond the training cutoff date.',
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
    },
    {
        id: 'vector-database',
        name: 'Vector Database',
        category: 'rag',
        type: 'infrastructure',
        shortDesc: 'Specialized databases for storing and querying vector embeddings',
        definition: 'Vector databases are specialized storage systems designed for efficient similarity search over high-dimensional vector embeddings. They use approximate nearest neighbor (ANN) algorithms like HNSW, IVF, or LSH to enable fast retrieval even at scale. Popular options include Pinecone, Weaviate, Chroma, Milvus, Qdrant, and pgvector for PostgreSQL.',
        related: ['embeddings', 'rag', 'semantic-search'],
        tags: ['infrastructure', 'storage', 'similarity-search'],
        codeExample: `# Vector Database with Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient(":memory:")

# Create collection
client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
)

# Upsert vectors
client.upsert(
    collection_name="documents",
    points=[{"id": "doc1", "vector": embedding, "payload": {"text": "..."}}]
)

# Search
results = client.search(
    collection_name="documents",
    query_vector=query_embedding,
    limit=5
)`
    },
    {
        id: 'embeddings',
        name: 'Embeddings',
        category: 'rag',
        type: 'core',
        shortDesc: 'Dense vector representations of text or data',
        definition: 'Embeddings are dense vector representations that capture semantic meaning in a continuous space. Text embeddings map words, sentences, or documents to vectors where semantically similar items are close in the vector space. Modern embedding models like OpenAI text-embedding-3, Cohere embeddings, and Sentence Transformers enable semantic search, clustering, and retrieval.',
        related: ['vector-database', 'semantic-search', 'rag'],
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
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))`
    },
    {
        id: 'chunking',
        name: 'Chunking',
        category: 'rag',
        type: 'technique',
        shortDesc: 'Splitting documents into pieces for retrieval',
        definition: 'Chunking is the process of splitting documents into smaller, semantically meaningful pieces for embedding and retrieval. Good chunking strategies balance maintaining context with keeping chunks focused. Methods include fixed-size chunking, recursive character splitting, and semantic chunking based on meaning boundaries.',
        related: ['rag', 'embeddings'],
        tags: ['preprocessing', 'retrieval', 'documents'],
        codeExample: `# Chunking Strategies
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\\n\\n", "\\n", ". ", " ", ""]
)
chunks = splitter.split_text(document)`
    },
    {
        id: 'reranking',
        name: 'Reranking',
        category: 'rag',
        type: 'technique',
        shortDesc: 'Re-scoring retrieved documents for relevance',
        definition: 'Reranking is a two-stage retrieval technique where initial fast retrieval using vector similarity is followed by a sophisticated cross-encoder model that scores document-query pairs. This significantly improves precision at the cost of latency. Models like Cohere Rerank, BGE Reranker provide better relevance scoring.',
        related: ['rag', 'semantic-search'],
        tags: ['retrieval', 'optimization', 'precision'],
        codeExample: `# Reranking with Cohere
import cohere
co = cohere.Client("api_key")

results = co.rerank(
    query=query,
    documents=[doc.page_content for doc in initial_docs],
    top_n=5,
    model="rerank-english-v3.0"
)`
    },
    {
        id: 'semantic-search',
        name: 'Semantic Search',
        category: 'rag',
        type: 'technique',
        shortDesc: 'Finding documents by meaning not keywords',
        definition: 'Semantic search uses embeddings to find documents based on meaning rather than exact keyword matches. It enables finding relevant content even when users use different vocabulary. Combined with keyword search in hybrid approaches, it provides robust retrieval.',
        related: ['embeddings', 'vector-database', 'rag'],
        tags: ['search', 'retrieval', 'semantic'],
        codeExample: `# Hybrid Search
def hybrid_search(query, vectorstore, bm25_index, alpha=0.5):
    semantic_results = vectorstore.similarity_search(query, k=20)
    keyword_results = bm25_index.search(query, k=20)
    # Reciprocal Rank Fusion
    return rrf_fusion(semantic_results, keyword_results, alpha)`
    },

    // ========== AGENTIC SECTION ==========
    {
        id: 'agentic-ai',
        name: 'Agentic AI',
        category: 'agentic',
        type: 'core',
        shortDesc: 'AI systems that autonomously plan and execute tasks',
        definition: 'Agentic AI refers to AI systems capable of autonomous goal-directed behavior. Unlike single-prompt interactions, agentic systems can plan multi-step actions, use external tools, maintain memory across interactions, and self-correct. Key frameworks include LangChain Agents, AutoGen, CrewAI, and the ReAct paradigm.',
        related: ['tool-use', 'planning', 'memory-agents', 'multi-agent'],
        tags: ['core', 'autonomous', 'framework'],
        codeExample: `# Agentic Loop with ReAct
def agent_loop(query, llm, tools, max_iterations=10):
    history = []
    for i in range(max_iterations):
        response = llm.invoke(f"Question: {query}\\nHistory: {history}")
        action = parse_action(response)
        if action.type == "FINISH":
            return action.answer
        observation = execute_tool(action.name, action.args)
        history.append(f"Action: {action}\\nObservation: {observation}")`
    },
    {
        id: 'tool-use',
        name: 'Tool Use',
        category: 'agentic',
        type: 'core',
        shortDesc: 'Enabling LLMs to interact with external systems',
        definition: 'Tool use (or function calling) allows LLMs to interact with external systems by generating structured outputs that trigger predefined functions. The model outputs parameters in a specified format, which are then executed by the runtime environment.',
        related: ['agentic-ai', 'mcp'],
        tags: ['agent', 'integration', 'external'],
        codeExample: `# Tool Definition with OpenAI
tools = [{
    "type": "function",
    "function": {
        "name": "search_web",
        "description": "Search the web for information",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"]
        }
    }
}]

response = client.chat.completions.create(
    model="gpt-4-turbo",
    messages=[{"role": "user", "content": "What's the weather?"}],
    tools=tools
)`
    },
    {
        id: 'planning',
        name: 'Planning',
        category: 'agentic',
        type: 'technique',
        shortDesc: 'Breaking complex goals into steps',
        definition: 'Planning in agentic AI involves decomposing complex goals into actionable steps. Techniques include chain-of-thought for implicit planning, explicit plan generation followed by execution, and dynamic replanning when execution fails.',
        related: ['agentic-ai', 'tool-use'],
        tags: ['agent', 'reasoning', 'decomposition'],
        codeExample: `# Planning Agent
def plan_and_execute(goal, llm, tools):
    plan = llm.invoke(f"Goal: {goal}\\nCreate a step-by-step plan.")
    steps = parse_plan(plan)
    for step in steps:
        result = execute_step(step, tools)
        if result.failed:
            plan = llm.invoke(f"Replan for: {goal}")
            steps = parse_plan(plan)`
    },
    {
        id: 'memory-agents',
        name: 'Agent Memory',
        category: 'agentic',
        type: 'technique',
        shortDesc: 'Persisting context across agent interactions',
        definition: 'Agent memory systems enable AI to retain information across interactions. Types include: short-term memory (conversation history), long-term memory (persistent facts), and working memory (current task context). Vector databases often power semantic memory retrieval.',
        related: ['agentic-ai', 'vector-database'],
        tags: ['agent', 'persistence', 'context'],
        codeExample: `# Agent Memory System
class AgentMemory:
    def __init__(self):
        self.short_term = []
        self.long_term = VectorStore()
    
    def add(self, message):
        self.short_term.append(message)
        if len(self.short_term) > 10:
            summary = summarize(self.short_term[:5])
            self.long_term.add(summary)
            self.short_term = self.short_term[5:]`
    },
    {
        id: 'multi-agent',
        name: 'Multi-Agent Systems',
        category: 'agentic',
        type: 'technique',
        shortDesc: 'Coordinating multiple AI agents',
        definition: 'Multi-agent systems coordinate multiple specialized AI agents to solve complex problems. Agents can have different roles (researcher, writer, reviewer), share common memory, and communicate through structured messages.',
        related: ['agentic-ai', 'planning'],
        tags: ['agent', 'coordination', 'teams'],
        codeExample: `# Multi-Agent with CrewAI
from crewai import Agent, Task, Crew

researcher = Agent(
    role="Researcher",
    goal="Find accurate information",
    tools=[search_tool]
)

writer = Agent(
    role="Writer",
    goal="Create engaging content"
)

crew = Crew(agents=[researcher, writer], tasks=[...])
result = crew.kickoff()`
    },

    // ========== MCP SECTION ==========
    {
        id: 'mcp',
        name: 'MCP',
        fullName: 'Model Context Protocol',
        category: 'mcp',
        type: 'core',
        shortDesc: 'Open protocol for connecting AI to systems',
        definition: 'The Model Context Protocol (MCP) is an open standard that enables AI assistants to connect to external systems uniformly. It provides a standardized way to expose resources (files, data), prompts (templates), and tools (functions) to AI models.',
        related: ['tool-use', 'mcp-resources', 'mcp-tools'],
        tags: ['protocol', 'standard', 'integration'],
        codeExample: `# MCP Server Implementation
from mcp.server import Server

server = Server("my-server")

@server.tool()
def search_database(query: str) -> str:
    """Search the database."""
    return db.search(query)

# Client usage
client.connect_to_server("my-server")
result = client.call_tool("search_database", {"query": "AI"})`
    },
    {
        id: 'mcp-resources',
        name: 'MCP Resources',
        category: 'mcp',
        type: 'technique',
        shortDesc: 'URI-based data access through MCP',
        definition: 'MCP Resources provide a URI-based way to expose read-only data to AI models. Resources can represent files, database records, API responses, or any data. They support templates for dynamic URIs.',
        related: ['mcp', 'mcp-tools'],
        tags: ['mcp', 'data', 'read-only'],
        codeExample: `# MCP Resource Definition
@server.resource("users://{user_id}/profile")
def get_user_profile(user_id: str) -> dict:
    return db.users.find(user_id)

# List resources
resources = client.list_resources()
data = client.read_resource("users/123/profile")`
    },
    {
        id: 'mcp-tools',
        name: 'MCP Tools',
        category: 'mcp',
        type: 'technique',
        shortDesc: 'Functions exposed through MCP',
        definition: 'MCP Tools are functions that AI models can call to perform actions. Unlike resources, tools can modify state, make API calls, or execute any operation. Tools include JSON Schema for parameters.',
        related: ['mcp', 'mcp-resources', 'tool-use'],
        tags: ['mcp', 'functions', 'actions'],
        codeExample: `# MCP Tool with Schema
@server.tool()
def send_email(to: str, subject: str, body: str) -> dict:
    \"\"\"Send an email.\"\"\"
    return email_service.send(to, subject, body)`
    },

    // ========== ARCHITECTURE SECTION ==========
    {
        id: 'transformers',
        name: 'Transformers',
        category: 'architecture',
        type: 'core',
        shortDesc: 'Foundational architecture behind modern LLMs',
        definition: 'Transformers are neural network architectures using self-attention mechanisms to process sequences in parallel. Introduced in "Attention is All You Need" (2017), they efficiently capture long-range dependencies. All major LLMs (GPT, Claude, Llama) are transformer-based.',
        related: ['attention-mechanism', 'llm', 'self-attention'],
        tags: ['fundamental', 'architecture', 'deep-learning'],
        codeExample: `# Transformer Block
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, n_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 2048),
            nn.GELU(),
            nn.Linear(2048, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        attn_out, _ = self.attention(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x`
    },
    {
        id: 'attention-mechanism',
        name: 'Attention',
        category: 'architecture',
        type: 'core',
        shortDesc: 'Allowing models to focus on relevant inputs',
        definition: 'Attention mechanisms allow neural networks to dynamically focus on relevant parts of input when producing each output. Self-attention computes relationships between all sequence positions.',
        related: ['transformers', 'self-attention', 'llm'],
        tags: ['fundamental', 'architecture', 'mechanism'],
        codeExample: `# Scaled Dot-Product Attention
import torch
import math

def attention(query, key, value, mask=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    return torch.matmul(torch.softmax(scores, dim=-1), value)`
    },
    {
        id: 'llm',
        name: 'LLM',
        fullName: 'Large Language Model',
        category: 'architecture',
        type: 'core',
        shortDesc: 'Foundation models trained on massive text',
        definition: 'Large Language Models are neural networks trained on vast text corpora to predict the next token. Through this objective, they learn language understanding, world knowledge, reasoning, and generation. Models like GPT-4, Claude, Llama range from 7B to trillions of parameters.',
        related: ['transformers', 'tokenization', 'pre-training'],
        tags: ['fundamental', 'foundation-model', 'generation'],
        codeExample: `# LLM Text Generation
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b")

inputs = tokenizer("Hello, world!", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))`
    },
    {
        id: 'tokenization',
        name: 'Tokenization',
        category: 'architecture',
        type: 'technique',
        shortDesc: 'Converting text to model tokens',
        definition: 'Tokenization converts raw text into tokens that models process. Modern LLMs use subword tokenization (BPE, WordPiece, SentencePiece) balancing vocabulary size with sequence length.',
        related: ['llm', 'embeddings', 'context-window'],
        tags: ['preprocessing', 'fundamentals', 'encoding'],
        codeExample: `# Tokenization with tiktoken
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4")
tokens = enc.encode("Hello, world!")  # [9906, 11, 1917, 0]
text = enc.decode(tokens)  # "Hello, world!"`
    },
    {
        id: 'context-window',
        name: 'Context Window',
        category: 'architecture',
        type: 'technique',
        shortDesc: 'Maximum sequence length for processing',
        definition: 'The context window is the maximum number of tokens a model can process in a single forward pass, including both input and output. Larger windows enable longer documents but increase computational cost.',
        related: ['llm', 'tokenization', 'kv-cache'],
        tags: ['limitations', 'architecture', 'performance'],
        codeExample: `# Context Management
class ContextManager:
    def __init__(self, max_tokens=4096):
        self.max_tokens = max_tokens
        self.messages = []
    
    def add(self, role, content, tokenizer):
        if count_tokens(content) + total_tokens() > self.max_tokens - 500:
            self.truncate()
        self.messages.append({"role": role, "content": content})`
    },

    // ========== TRAINING SECTION ==========
    {
        id: 'pre-training',
        name: 'Pre-training',
        category: 'training',
        type: 'core',
        shortDesc: 'Initial training on massive datasets',
        definition: 'Pre-training is the foundational phase where models learn from massive unlabeled text corpora. The objective is typically next-token prediction, teaching language patterns, world knowledge, and reasoning.',
        related: ['llm', 'fine-tuning'],
        tags: ['training', 'foundation', 'compute'],
        codeExample: `# Pre-training Loop
def pretrain(model, dataloader, optimizer):
    model.train()
    for batch in dataloader:
        outputs = model(batch["input_ids"], labels=batch["labels"])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()`
    },
    {
        id: 'fine-tuning',
        name: 'Fine-tuning',
        category: 'training',
        type: 'technique',
        shortDesc: 'Adapting pre-trained models for tasks',
        definition: 'Fine-tuning adapts a pre-trained model to specific tasks by continuing training on labeled data. Methods include full fine-tuning and parameter-efficient methods like LoRA.',
        related: ['pre-training', 'lora', 'rlhf'],
        tags: ['training', 'adaptation', 'specialization'],
        codeExample: `# Fine-tuning with LoRA
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()`
    },
    {
        id: 'rlhf',
        name: 'RLHF',
        fullName: 'Reinforcement Learning from Human Feedback',
        category: 'training',
        type: 'technique',
        shortDesc: 'Aligning models with human preferences',
        definition: 'RLHF aligns language models with human preferences by training a reward model on human comparisons, then optimizing the LLM to maximize this reward using PPO.',
        related: ['fine-tuning', 'dpo'],
        tags: ['alignment', 'training', 'human-feedback'],
        codeExample: `# RLHF Loss
def rlhf_loss(policy, ref_policy, prompt, chosen, rejected, beta=0.1):
    chosen_reward = beta * (policy.log_prob(prompt, chosen) - ref_policy.log_prob(prompt, chosen))
    rejected_reward = beta * (policy.log_prob(prompt, rejected) - ref_policy.log_prob(prompt, rejected))
    return -torch.log(torch.sigmoid(chosen_reward - rejected_reward))`
    },
    {
        id: 'dpo',
        name: 'DPO',
        fullName: 'Direct Preference Optimization',
        category: 'training',
        type: 'technique',
        shortDesc: 'Simpler alternative to RLHF',
        definition: 'DPO is a simpler alternative to RLHF that directly optimizes the policy using preference data without training a separate reward model.',
        related: ['rlhf', 'fine-tuning'],
        tags: ['alignment', 'training', 'simplified'],
        codeExample: `# DPO Loss
def dpo_loss(policy, ref_policy, prompt, chosen, rejected, beta=0.1):
    chosen_reward = beta * (policy.log_prob(prompt, chosen) - ref_policy.log_prob(prompt, chosen))
    rejected_reward = beta * (policy.log_prob(prompt, rejected) - ref_policy.log_prob(prompt, rejected))
    return -torch.log(torch.sigmoid(chosen_reward - rejected_reward)).mean()`
    },
    {
        id: 'lora',
        name: 'LoRA',
        fullName: 'Low-Rank Adaptation',
        category: 'training',
        type: 'technique',
        shortDesc: 'Parameter-efficient fine-tuning',
        definition: 'LoRA enables efficient fine-tuning by adding small trainable rank-decomposition matrices to existing weights. This reduces trainable parameters by 10,000x while maintaining comparable performance.',
        related: ['fine-tuning'],
        tags: ['efficient', 'fine-tuning', 'adaptation'],
        codeExample: `# LoRA Layer
class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=16, alpha=32):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features), requires_grad=False)
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank
    
    def forward(self, x):
        return x @ self.weight.T + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling`
    },

    // ========== PROMPTING SECTION ==========
    {
        id: 'prompt-engineering',
        name: 'Prompt Engineering',
        category: 'prompting',
        type: 'core',
        shortDesc: 'Crafting effective LLM inputs',
        definition: 'Prompt engineering is designing inputs to elicit desired outputs from language models. Techniques include clear instruction specification, providing examples (few-shot), and breaking down tasks (chain-of-thought).',
        related: ['chain-of-thought', 'few-shot', 'system-prompt'],
        tags: ['fundamental', 'techniques', 'optimization'],
        codeExample: `# Prompt Best Practices
specific_prompt = """
Write a 500-word blog post about AI in healthcare.
Target audience: Healthcare professionals
Include: 3 specific hospital AI applications
Structure: Introduction, 3 sections, conclusion
"""`
    },
    {
        id: 'chain-of-thought',
        name: 'Chain of Thought',
        category: 'prompting',
        type: 'technique',
        shortDesc: 'Step-by-step reasoning decomposition',
        definition: 'Chain-of-Thought (CoT) prompting improves reasoning by asking models to think step-by-step before giving final answers. Zero-shot CoT adds "Let\'s think step by step".',
        related: ['prompt-engineering', 'planning'],
        tags: ['reasoning', 'technique', 'prompting'],
        codeExample: `# Zero-shot CoT
prompt = """
Question: A store has 23 apples. They sell 15 and receive 8 more. How many?
Let's think step by step.
"""`
    },
    {
        id: 'few-shot',
        name: 'Few-Shot Prompting',
        category: 'prompting',
        type: 'technique',
        shortDesc: 'Providing examples to guide behavior',
        definition: 'Few-shot prompting provides input-output examples before the actual query. This helps models understand desired format, style, and reasoning without weight updates.',
        related: ['prompt-engineering'],
        tags: ['technique', 'examples', 'in-context'],
        codeExample: `# Few-Shot Classification
prompt = """
Classify sentiment:

Text: "This product exceeded my expectations!"
Sentiment: Positive

Text: "Worst purchase ever."
Sentiment: Negative

Text: "It's okay, nothing special."
Sentiment:
"""`
    },
    {
        id: 'system-prompt',
        name: 'System Prompt',
        category: 'prompting',
        type: 'technique',
        shortDesc: 'Defining model behavior and constraints',
        definition: 'System prompts are instructions that define model behavior, personality, capabilities, and constraints. They\'re processed before user messages and persist through conversations.',
        related: ['prompt-engineering'],
        tags: ['configuration', 'behavior', 'instructions'],
        codeExample: `# System Prompt
system_prompt = """
You are a helpful AI assistant specializing in Python.
- Provide clear, well-commented code
- Explain at an intermediate level
- Suggest best practices
"""

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]
)`
    },

    // ========== INFRASTRUCTURE SECTION ==========
    {
        id: 'inference',
        name: 'Inference',
        category: 'infrastructure',
        type: 'core',
        shortDesc: 'Running trained models for predictions',
        definition: 'Inference is running a trained model to generate outputs. Key considerations: latency, throughput, cost, and accuracy. Optimization techniques include quantization, batching, and caching.',
        related: ['quantization', 'kv-cache'],
        tags: ['deployment', 'production', 'optimization'],
        codeExample: `# Batched Inference
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-2-7b-hf")
params = SamplingParams(max_tokens=100, temperature=0.7)
outputs = llm.generate(prompts, params)`
    },
    {
        id: 'quantization',
        name: 'Quantization',
        category: 'infrastructure',
        type: 'technique',
        shortDesc: 'Reducing model precision for efficiency',
        definition: 'Quantization reduces memory and compute requirements by lowering numerical precision (FP16 to INT8 or INT4). Methods like GPTQ, AWQ maintain quality while reducing size 2-4x.',
        related: ['inference', 'lora'],
        tags: ['optimization', 'efficiency', 'deployment'],
        codeExample: `# Quantized Loading
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b",
    quantization_config=bnb_config
)`
    },
    {
        id: 'kv-cache',
        name: 'KV Cache',
        fullName: 'Key-Value Cache',
        category: 'infrastructure',
        type: 'technique',
        shortDesc: 'Caching attention states for faster generation',
        definition: 'The KV Cache stores computed key and value tensors during autoregressive generation. This reduces generation from O(n2) to O(n) complexity, dramatically speeding up inference.',
        related: ['inference', 'attention-mechanism'],
        tags: ['optimization', 'memory', 'generation'],
        codeExample: `# KV Cache
class KVCache:
    def __init__(self, n_layers, n_heads, head_dim, max_seq_len):
        self.cache = [torch.zeros(1, n_heads, max_seq_len, head_dim) 
                      for _ in range(2 * n_layers)]
        self.seq_len = 0
    
    def update(self, layer_idx, new_k, new_v):
        self.cache[2*layer_idx][:, :, self.seq_len] = new_k
        self.cache[2*layer_idx+1][:, :, self.seq_len] = new_v
        self.seq_len += 1`
    },

    // ========== APPLICATIONS SECTION ==========
    {
        id: 'chatbots',
        name: 'Chatbots',
        category: 'applications',
        type: 'application',
        shortDesc: 'Conversational AI interfaces',
        definition: 'Chatbots are conversational AI systems that interact through natural language dialogue. Modern chatbots use LLMs for conversation combined with RAG for knowledge, tools for actions, and memory for context.',
        related: ['rag', 'agentic-ai', 'memory-agents'],
        tags: ['application', 'conversation', 'interface'],
        codeExample: `# Chatbot
class Chatbot:
    def __init__(self, llm, rag, memory):
        self.llm = llm
        self.rag = rag
        self.memory = memory
    
    async def respond(self, user_input):
        self.memory.add({"role": "user", "content": user_input})
        context = await self.rag.retrieve(user_input)
        response = await self.llm.generate(self.memory.get_context(), context)
        self.memory.add({"role": "assistant", "content": response})
        return response`
    },
    {
        id: 'code-generation',
        name: 'Code Generation',
        category: 'applications',
        type: 'application',
        shortDesc: 'AI-powered code writing assistance',
        definition: 'Code generation uses LLMs trained on code to write, complete, explain, and debug software. Models like GitHub Copilot, CodeLlama can generate code from natural language.',
        related: ['llm', 'tool-use'],
        tags: ['application', 'development', 'productivity'],
        codeExample: `# Code Assistant
class CodeAssistant:
    def generate(self, prompt, language="python"):
        return self.model.generate(f"Language: {language}\\nTask: {prompt}")
    
    def explain(self, code):
        return self.model.generate(f"Explain this code:\\n{code}")
    
    def test(self, code):
        return self.model.generate(f"Write tests for:\\n{code}")`
    }
];

// ==========================================
// BUILD KNOWLEDGE BASE OBJECT
// ==========================================
const KnowledgeBase = {
    meta: {
        version: '1.0.0',
        lastUpdated: '2025-01-15',
        description: 'Interactive AI Knowledge Graph'
    },
    categories: CategoryData,
    terms: TermData.map(t => createTerm(t))
};

// ==========================================
// UTILITY FUNCTIONS
// ==========================================
const KnowledgeUtils = {
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

// ==========================================
// CONSOLE LOG FOR DEBUGGING
// ==========================================
console.log('KnowledgeBase loaded:', KnowledgeBase.categories.length, 'categories,', KnowledgeBase.terms.length, 'terms');

// Export to window
window.KnowledgeBase = KnowledgeBase;
window.KnowledgeUtils = KnowledgeUtils;
