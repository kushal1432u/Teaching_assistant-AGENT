# EduMate – AI Agent Learning Assistant 📚🤖

**EduMate** is an intelligent learning companion powered by **LangGraph** agents. It allows users to upload educational documents and interact with an intelligent agent capable of multi-step reasoning and automated content generation.

---

## 🚀 Key Features

* **Agentic Orchestration:** Built with **LangGraph** for advanced state management and multi-step reasoning.
* **Document Intelligence:** * Supports **PDF & DOCX** (up to 200MB).
    * Automatic document chunking and processing.
    * Uses **Sentence Transformers** for high-accuracy semantic search.
* **Built-in Learning Tools:**
    * **Quiz Generation:** Instantly create MCQs from your reading material.
    * **Assignment Creation:** Generate structured homework or practice tasks.
    * **Summary & Analysis:** Get concise overviews of complex documents.
    * **Q&A Answering:** Chat directly with your documents to clarify doubts.

---

## 🛠️ Technical Stack

| Component | Technology |
| :--- | :--- |
| **Agent Framework** | LangGraph |
| **Embeddings** | Sentence Transformers |
| **Interface** | Streamlit |
| **Document Handling** | PyPDF / Docx2txt |
| **Reasoning** | Multi-step Agentic Logic |

---

## 💻 Getting Started

### Prerequisites
* Python 3.9+
* API Key for your LLM provider (OpenAI, Anthropic, etc.)

### Installation
1. **Clone the repository**
   ```bash
   git clone [https://github.com/yourusername/edumate.git](https://github.com/yourusername/edumate.git)
   cd edumate Install dependencies

Bash

pip install -r requirements.txt
Configure Environment Create a .env file in the root directory:

Code snippet

LLM_API_KEY=your_api_key_here
Launch the App

Bash

streamlit run app.py
💡 Usage Examples
Once the agent is initialized (as shown by the @LangGraph Agent initialized! status), you can try:

"Generate a quiz on the main concepts of this file."

"Create an assignment based on the introduction section."

"Summarize the key findings in 5 bullet points."

📝 License
Distributed under the MIT License. See LICENSE for more information.

Built with ❤️ for the future of AI-driven education.


Would you like me to help you write a **Requirements.txt** file to go along with this R 
