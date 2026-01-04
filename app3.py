# EduMate - Intelligent Learning Assistant with LangGraph Agents
import os
import tempfile
import streamlit as st
from typing import List, Annotated, TypedDict, Literal
import docx
from pypdf import PdfReader

# Embeddings
EMBED_MODEL = None
USE_HF_EMBED = False

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np
    EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    EMBED_MODEL = SentenceTransformer(EMBED_MODEL_NAME)
    USE_HF_EMBED = True
except Exception:
    EMBED_MODEL = None
    USE_HF_EMBED = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# LangGraph and LangChain imports
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool

# ------------------ UTILS ------------------
def split_text_into_chunks(text: str, chunk_size=1000, chunk_overlap=200):
    """Split text into overlapping chunks"""
    if not text:
        return []
    chunks, start, N = [], 0, len(text)
    while start < N:
        end = min(start + chunk_size, N)
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

def load_pdf(path: str) -> str:
    """Extract text from PDF"""
    text = []
    reader = PdfReader(path)
    for p in reader.pages:
        t = p.extract_text()
        if t:
            text.append(t)
    return "\n".join(text)

def load_docx(path: str) -> str:
    """Extract text from DOCX"""
    d = docx.Document(path)
    return "\n".join([p.text for p in d.paragraphs if p.text.strip()])

# ------------------ INDEX BUILD ------------------
def build_hf_index(chunks: List[str]):
    """Build FAISS index with sentence transformers"""
    vectors = EMBED_MODEL.encode(chunks, convert_to_numpy=True)
    faiss.normalize_L2(vectors)
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index, vectors, chunks

def query_hf_index(index, query: str, top_k=4):
    """Query FAISS index"""
    q_emb = EMBED_MODEL.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, top_k)
    return I[0].tolist()

def build_tfidf_index(chunks: List[str]):
    """Build TF-IDF index"""
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(chunks)
    return tfidf, X, chunks

def query_tfidf_index(tfidf, X, query: str, top_k=4):
    """Query TF-IDF index"""
    qv = tfidf.transform([query])
    sims = cosine_similarity(qv, X).flatten()
    idxs = sims.argsort()[::-1][:top_k]
    return idxs.tolist()

# ------------------ LLM ------------------
os.environ["GROQ_API_KEY"] = "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"
LLM = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.2)

def ask_groq(prompt: str) -> str:
    """Query Groq LLM"""
    try:
        return LLM.invoke(prompt).content
    except Exception as e:
        return f"Error: {str(e)}"

# ------------------ HELPER FUNCTIONS ------------------
def fetch_context_helper(query: str, top_k=6) -> str:
    """Fetch relevant context from the document"""
    try:
        if "chunks" not in st.session_state or not st.session_state.chunks:
            return "No document loaded. Please upload a document first."
        
        chunks = st.session_state.chunks
        mode = st.session_state.get("index_mode", "tfidf")
        
        if mode == "hf" and st.session_state.get("hf_index") is not None:
            idxs = query_hf_index(st.session_state.hf_index, query, top_k)
        elif st.session_state.get("tfidf") is not None:
            idxs = query_tfidf_index(st.session_state.tfidf, st.session_state.tfidf_X, query, top_k)
        else:
            return "Document index not ready. Please upload a document."
        
        return "\n\n".join([chunks[i] for i in idxs])
    except Exception as e:
        return f"Error fetching context: {str(e)}"

# ------------------ LANGGRAPH TOOLS ------------------
@tool
def fetch_document_context(query: str) -> str:
    """
    Fetch relevant passages from the uploaded document based on the query.
    Use this tool when you need to retrieve specific information from the document.
    
    Args:
        query: The search query to find relevant document passages
    
    Returns:
        Relevant text passages from the document
    """
    return fetch_context_helper(query, top_k=6)

@tool
def generate_summary(topic: str = "general") -> str:
    """
    Generate a comprehensive summary of the document content.
    
    Args:
        topic: Optional specific topic to focus the summary on
    
    Returns:
        A detailed summary of the document
    """
    context = fetch_context_helper(f"summary overview {topic}")
    prompt = f"""Provide a clear and comprehensive summary of this content.
Use paragraphs and structure it well.

Content:
{context}

Summary:"""
    return ask_groq(prompt)

@tool
def generate_analysis(focus: str = "general") -> str:
    """
    Provide academic analysis of the document with key concepts and themes.
    
    Args:
        focus: Optional specific aspect to focus the analysis on
    
    Returns:
        Detailed academic analysis with sections
    """
    context = fetch_context_helper(f"analysis key concepts {focus}")
    prompt = f"""Provide detailed academic analysis with the following structure:

1. INTRODUCTION: Brief overview
2. KEY CONCEPTS: Main ideas and definitions
3. THEMES & PATTERNS: Important themes identified
4. CRITICAL ANALYSIS: Deep insights
5. CONCLUSION: Summary of findings

Content:
{context}

Analysis:"""
    return ask_groq(prompt)

@tool
def create_quiz_questions(topic: str, num_questions: int = 8) -> str:
    """
    Generate MCQ quiz questions based on the document content.
    
    Args:
        topic: The topic or area to focus the quiz on
        num_questions: Number of questions to generate (default 8)
    
    Returns:
        Quiz questions in MCQ format with answers
    """
    context = fetch_context_helper(topic)
    prompt = f"""Generate {num_questions} Multiple Choice Questions (MCQ) based on this content.

FORMAT FOR EACH QUESTION:
Q[number]. [Question text]
A) [Option A]
B) [Option B]
C) [Option C]
D) [Option D]

✓ Correct Answer: [Letter]
📝 Explanation: [Brief explanation why this is correct]

---

Make questions test understanding, not just memorization.
Include a mix of difficulty levels.

Content:
{context}

QUIZ:"""
    return ask_groq(prompt)

@tool
def create_assignment(topic: str) -> str:
    """
    Generate a BCA assignment based on the document content.
    
    Args:
        topic: The topic or area to base the assignment on
    
    Returns:
        A structured assignment with tasks and evaluation criteria
    """
    context = fetch_context_helper(topic)
    prompt = f"""Generate a detailed BCA (Bachelor of Computer Applications) assignment.

Use this EXACT FORMAT:

═══════════════════════════════════════════
📋 ASSIGNMENT TITLE: [Creative Title Based on Topic]
═══════════════════════════════════════════

🎯 MAIN OBJECTIVE:
[Clear statement of what students will learn/achieve]

───────────────────────────────────────────
📝 TASK 1: [Task Name]
───────────────────────────────────────────
Description: [What needs to be done]
Requirements: 
  • [Requirement 1]
  • [Requirement 2]
  • [Requirement 3]
Deliverables: [What to submit]

───────────────────────────────────────────
📝 TASK 2: [Task Name]
───────────────────────────────────────────
Description: [What needs to be done]
Requirements:
  • [Requirement 1]
  • [Requirement 2]
  • [Requirement 3]
Deliverables: [What to submit]

───────────────────────────────────────────
📝 TASK 3: [Task Name]
───────────────────────────────────────────
Description: [What needs to be done]
Requirements:
  • [Requirement 1]
  • [Requirement 2]
  • [Requirement 3]
Deliverables: [What to submit]

───────────────────────────────────────────
📊 EVALUATION CRITERIA:
───────────────────────────────────────────
• [Criterion 1] - [XX]%
• [Criterion 2] - [XX]%
• [Criterion 3] - [XX]%
• [Criterion 4] - [XX]%
Total: 100%

───────────────────────────────────────────
📤 SUBMISSION GUIDELINES:
───────────────────────────────────────────
• Deadline: [Reasonable timeframe]
• Format: [File format requirements]
• Submission Method: [How to submit]
• Additional Notes: [Any other important info]

═══════════════════════════════════════════

Content:
{context}

ASSIGNMENT:"""
    return ask_groq(prompt)

@tool
def answer_question(question: str) -> str:
    """
    Answer a specific question based on the document content.
    
    Args:
        question: The question to answer
    
    Returns:
        Detailed answer based strictly on document content
    """
    context = fetch_context_helper(question)
    prompt = f"""Answer this question based on the provided context.

FORMAT:
🔍 QUESTION: {question}

📖 ANSWER:
[Provide a detailed, well-structured answer]

💡 KEY POINTS:
• [Key point 1]
• [Key point 2]
• [Key point 3]

If the answer is not in the context, clearly state: "This information is not available in the provided document."

Context:
{context}

Answer:"""
    return ask_groq(prompt)

# Create tools list
TOOLS = [
    fetch_document_context,
    generate_summary,
    generate_analysis,
    create_quiz_questions,
    create_assignment,
    answer_question
]

# Bind tools to LLM
llm_with_tools = LLM.bind_tools(TOOLS)

# ------------------ LANGGRAPH STATE ------------------
class AgentState(TypedDict):
    """State for the agent graph"""
    messages: Annotated[list, "The messages in the conversation"]

# ------------------ LANGGRAPH NODES ------------------
def agent_node(state: AgentState) -> AgentState:
    """Main agent reasoning node"""
    messages = state["messages"]
    
    # Add system message if first message
    if not any(isinstance(m, SystemMessage) for m in messages):
        system_msg = SystemMessage(content="""You are EduMate, an intelligent learning assistant specialized in education.

CRITICAL INSTRUCTIONS:
1. ALWAYS use tools to get information from the document
2. AFTER using tools, you MUST synthesize the results and provide a complete response
3. Format your responses based on the request type:
   - Quiz: Present the MCQ format clearly
   - Assignment: Show the structured format
   - Summary: Use clear paragraphs
   - Analysis: Use sections with headers
   - Q&A: Provide detailed answers

Available tools:
- fetch_document_context: Get relevant text from document
- generate_summary: Create summaries
- generate_analysis: Provide academic analysis
- create_quiz_questions: Generate MCQ quizzes
- create_assignment: Create structured assignments
- answer_question: Answer specific questions

WORKFLOW:
1. Understand user's request
2. Call appropriate tool(s)
3. MUST respond with the tool results in proper format
4. Be helpful and educational

NEVER leave responses empty after tool calls!""")
        messages = [system_msg] + messages
    
    # Get LLM response
    response = llm_with_tools.invoke(messages)
    
    return {
        "messages": messages + [response]
    }

def tool_node_func(state: AgentState) -> AgentState:
    """Execute tool calls"""
    messages = state["messages"]
    last_message = messages[-1]
    
    # Create tool node and execute
    tool_executor = ToolNode(TOOLS)
    tool_results = tool_executor.invoke({"messages": [last_message]})
    
    return {
        "messages": messages + tool_results["messages"]
    }

# ------------------ ROUTING FUNCTION ------------------
def should_continue(state: AgentState) -> Literal["tools", "synthesize", "end"]:
    """Determine next step based on last message"""
    messages = state["messages"]
    last_message = messages[-1]
    
    # If last message has tool calls, go to tools
    if isinstance(last_message, AIMessage) and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    
    # If last message is a tool result, synthesize
    if isinstance(last_message, ToolMessage):
        return "synthesize"
    
    # Otherwise end
    return "end"

def synthesize_node(state: AgentState) -> AgentState:
    """Synthesize tool results into final response"""
    messages = state["messages"]
    
    # Create a prompt to synthesize the tool results
    synthesis_prompt = """Based on the tool results above, provide a complete, well-formatted response to the user.

IMPORTANT:
- Present the information clearly and professionally
- Maintain the format requested (quiz, assignment, etc.)
- Do NOT add extra commentary like "Here's the quiz..." - just present the content
- Be comprehensive and helpful

Now provide the final formatted response:"""
    
    # Add synthesis instruction
    messages_with_prompt = messages + [HumanMessage(content=synthesis_prompt)]
    
    # Get final response
    final_response = LLM.invoke(messages_with_prompt)
    
    return {
        "messages": messages + [final_response]
    }

# ------------------ BUILD LANGGRAPH ------------------
def create_agent_graph():
    """Create the LangGraph agent workflow"""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node_func)
    workflow.add_node("synthesize", synthesize_node)
    
    # Set entry point
    workflow.set_entry_point("agent")
    
    # Add edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "synthesize": "synthesize",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "tools",
        lambda state: "synthesize",
        {
            "synthesize": "synthesize"
        }
    )
    
    workflow.add_conditional_edges(
        "synthesize",
        lambda state: "end",
        {
            "end": END
        }
    )
    
    # Compile
    return workflow.compile()

# ------------------ UI ------------------
st.set_page_config(page_title="EduMate – Intelligent Learning Assistant", layout="wide")
st.title("📚 EduMate – AI Agent Learning Assistant")
st.markdown("*Powered by LangGraph Agents - Upload documents and interact with an intelligent AI agent*")

# Sidebar
with st.sidebar:
    st.header("🤖 Agent Info")
    st.markdown("""
    This app uses **LangGraph agents** with:
    - Tool calling capabilities
    - State management
    - Multi-step reasoning
    - Response synthesis
    
    **Available Tools:**
    - Document search
    - Summary generation
    - Analysis creation
    - Quiz generation (MCQ)
    - Assignment creation
    - Q&A answering
    """)
    st.markdown("---")
    st.markdown("**Example queries:**")
    st.code("Generate a quiz on loops")
    st.code("Create an assignment on databases")
    st.code("What are the main concepts?")
    st.code("Analyze the key themes")

# Initialize session state FIRST (before any usage)
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "index_mode" not in st.session_state:
    st.session_state.index_mode = None
if "hf_index" not in st.session_state:
    st.session_state.hf_index = None
if "tfidf" not in st.session_state:
    st.session_state.tfidf = None
if "tfidf_X" not in st.session_state:
    st.session_state.tfidf_X = None
if "summary" not in st.session_state:
    st.session_state.summary = None
if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "agent_graph" not in st.session_state:
    st.session_state.agent_graph = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# File uploader
uploaded = st.file_uploader("📁 Upload PDF or DOCX", type=["pdf", "docx"])

# Process uploaded file
if uploaded:
    with st.spinner("⏳ Processing document..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded.name) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = tmp.name
        
        try:
            # Extract text
            full_text = load_pdf(tmp_path) if uploaded.name.endswith(".pdf") else load_docx(tmp_path)
            
            # Display preview
            st.markdown("### 📄 Document Preview")
            with st.expander("Click to view"):
                st.text_area("Preview", full_text[:800] + ("..." if len(full_text) > 800 else ""), height=200, disabled=True)
            
            # Split into chunks
            chunks = split_text_into_chunks(full_text)
            st.session_state.chunks = chunks
            st.success(f"✅ Processed: {len(chunks)} chunks created")
            
            # Build index
            with st.spinner("🔨 Building index..."):
                if USE_HF_EMBED and EMBED_MODEL:
                    try:
                        idx, embs, meta = build_hf_index(chunks)
                        st.session_state.index_mode = "hf"
                        st.session_state.hf_index = idx
                        st.info("✨ Using sentence transformers")
                    except:
                        tfidf, X, meta = build_tfidf_index(chunks)
                        st.session_state.index_mode = "tfidf"
                        st.session_state.tfidf = tfidf
                        st.session_state.tfidf_X = X
                        st.info("📊 Using TF-IDF")
                else:
                    tfidf, X, meta = build_tfidf_index(chunks)
                    st.session_state.index_mode = "tfidf"
                    st.session_state.tfidf = tfidf
                    st.session_state.tfidf_X = X
                    st.info("📊 Using TF-IDF")
            
            # Create agent graph
            st.session_state.agent_graph = create_agent_graph()
            st.success("🤖 LangGraph Agent initialized!")
            
            # Generate initial summary
            with st.spinner("🤖 Agent generating summary..."):
                ctx = fetch_context_helper("summary overview")
                summary = ask_groq(f"Summarize:\n{ctx}\nSummary:")
                st.session_state.summary = summary
            
            os.unlink(tmp_path)
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Main interface
if st.session_state.get("chunks") and st.session_state.get("agent_graph"):
    st.markdown("---")
    
    # Display summary
    if st.session_state.get("summary"):
        with st.expander("📝 Document Summary", expanded=True):
            st.markdown(st.session_state.summary)
    
    st.markdown("---")
    st.subheader("💬 Chat with AI Agent")
    st.success("✅ Document loaded! Agent is ready to help.")
    
    # Display message history
    for msg in st.session_state.messages:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.markdown(msg.content)
        elif isinstance(msg, AIMessage) and msg.content:
            # Check if this is a final response (not just a tool-calling message)
            has_tool_calls = hasattr(msg, 'tool_calls') and msg.tool_calls and len(msg.tool_calls) > 0
            
            # Display if it has substantial content (final response)
            if not has_tool_calls or len(msg.content) > 50:
                with st.chat_message("assistant"):
                    st.markdown(msg.content)
                    # Show tools used if any
                    if has_tool_calls:
                        with st.expander("🔧 Tools Used"):
                            for tool_call in msg.tool_calls:
                                st.code(f"Tool: {tool_call['name']}\nArgs: {tool_call['args']}")
    
    # Chat input
    user_input = st.chat_input("Ask me anything about the document...")
    
    if user_input:
        # Check if document is loaded
        if not st.session_state.chunks:
            st.error("⚠️ Please upload a document first before asking questions!")
        else:
            # Add user message
            st.session_state.messages.append(HumanMessage(content=user_input))
            
            with st.chat_message("user"):
                st.markdown(user_input)
            
            # Run agent
            with st.chat_message("assistant"):
                with st.spinner("🤖 Agent thinking..."):
                    try:
                        # Invoke agent graph
                        result = st.session_state.agent_graph.invoke({
                            "messages": st.session_state.messages
                        })
                        
                        # Get final messages
                        final_messages = result["messages"]
                        
                        # Update session state
                        st.session_state.messages = final_messages
                        
                        # Find and display the last AI message with content
                        response_found = False
                        for msg in reversed(final_messages):
                            if isinstance(msg, AIMessage) and msg.content:
                                # Check if this message has tool calls AND content
                                # or if it's just a final response
                                has_tool_calls = hasattr(msg, 'tool_calls') and msg.tool_calls and len(msg.tool_calls) > 0
                                
                                # Display if it's a final response (no tool calls) or has substantial content
                                if not has_tool_calls or len(msg.content) > 50:
                                    st.markdown(msg.content)
                                    response_found = True
                                    break
                        
                        if not response_found:
                            st.warning("⚠️ Agent didn't generate a response. Please try again.")
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        st.exception(e)
                        st.info("💡 Try rephrasing your question or reload the page")
            
            st.rerun()

else:
    st.info("👆 Upload a document to start chatting with the AI agent!")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### ✨ Features:
        - 🤖 **LangGraph Agent** architecture
        - 🔧 **Tool calling** for specialized tasks
        - 🧠 **Multi-step reasoning**
        - 💾 **Stateful conversations**
        - 📚 **Document understanding**
        - 📝 **Formatted outputs** (MCQ, Assignments, etc.)
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 Use Cases:
        - Study assistance
        - Quiz generation (MCQ format)
        - Assignment creation (Structured)
        - Content analysis
        - Q&A sessions
        """)

st.markdown("---")
st.markdown("*Powered by LangGraph 🚀 | Groq AI ⚡ | Built with ❤️*")