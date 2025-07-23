# 📰 VNExpress RAG Chatbot

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Vector DB](https://img.shields.io/badge/VectorDB-Qdrant-purple)](https://qdrant.tech/)
[![LLMs](https://img.shields.io/badge/LLMs-Gemini%20%2F%20OpenAI%20%2F%20Ollama-lightgrey)](#tech-stack)
[![Frontend](https://img.shields.io/badge/Frontend-Streamlit-orange)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/RAG%20Stack-LangChain-green)](https://python.langchain.com/)

---

## 🧠 What is this?

**VNExpress RAG Chatbot** is an intelligent Q&A system powered by **LangChain**, **Qdrant**, and **Gemini/OpenAI/Ollama**, designed to retrieve and summarize Vietnamese news in real time from [VNExpress](https://vnexpress.net/).

> 🧩 Retrieval-Augmented Generation (RAG) ensures accurate, up-to-date answers by combining live vector search with LLM generation.

---

## 🚀 Why This Matters

📌 **News evolves fast. Static LLMs can’t keep up.**  
This RAG chatbot dynamically pulls **fresh news** from VNExpress and answers your questions on the fly using powerful LLMs.

🔍 **Perfect for**:
- 📈 **Finance & Securities** firms needing instant summaries of breaking news
- 📰 **Journalists** monitoring policy or tech updates
- 🧑‍💻 **Developers** building news intelligence or alert systems

---

## 🛠️ Features at a Glance

| ✅ Feature                         | 🔍 Description                                               |
|----------------------------------|--------------------------------------------------------------|
| 📰 **VNExpress Crawler**         | Scrapes and extracts latest news from [vnexpress.net](https://vnexpress.net) |
| 🔎 **Qdrant Vector Search**      | Embeds articles and enables fast semantic search             |
| 🧠 **Gemini / OpenAI Summarizer**| Generates concise answers from retrieved documents           |
| 🖥️ **Streamlit UI**             | Friendly web interface for exploration and Q&A               |
| ⚙️ **.env Config + Logs**       | Simple setup, secure API usage, and detailed logging         |

---

## 🎯 Use Cases

- 💹 **Securities Firms** – Instant updates on market regulations or government policy
- 🧾 **Research Teams** – Auto-summary of political or business articles
- 🤖 **Chatbots** – Embed into client-facing news or alert platforms

---

## 🧬 Tech Stack

| Layer            | Tool/Library                     |
|------------------|----------------------------------|
| Language Model   | `Gemini`, `OpenAI`, `Ollama`     |
| RAG Framework    | `LangChain`                      |
| Vector DB        | `Qdrant`                         |
| Crawler          | `BeautifulSoup`, `Requests`      |
| Web Interface    | `Streamlit`                      |
| Embedding        | `GoogleGenerativeAIEmbeddings`   |

---

## 🏢 Why RAG for News? (Enterprise Perspective)

VNExpress is Vietnam’s leading online news portal, trusted by millions for timely, accurate updates on politics, finance, technology, and more.

**Why build a RAG chatbot for news?**
- **Real-Time Intelligence:** Companies (e.g., securities firms, banks, research teams) need instant, reliable news summaries to make fast decisions and keep clients informed.
- **Automated Summarization:** RAG chatbots can distill complex news into actionable insights, saving analysts and customers hours of manual reading.
- **Scalable Service:** Integrate into client portals, dashboards, or alert systems to deliver personalized news Q&A and summaries at scale.
- **Competitive Edge:** Stay ahead of market moves, regulatory changes, and breaking events with AI-powered news intelligence.

---

## ⚡ Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/mhuy26/VNExpress-RAG-ChatBot.git
cd VNExpress-RAG-ChatBot
```

### 2. Set Up Python Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure .env
Create a `.env` file in the project root:
```env
QDRANT_URL=your-cloud-qdrant-url
QDRANT_API_KEY=your-qdrant-api-key
QDRANT_COLLECTION=vnexpress_articles
GOOGLE_API_KEY=your-google-api-key
GOOGLE_EMBEDDING_MODEL_NAME=models/embedding-001
```

### 4. Crawl & Embed News Articles
```bash
python source/crawl_vnexpress.py
```

### 5. Launch Chatbot Web App
```bash
streamlit run source/app.py
```

---

## 🧩 Advanced Usage
- **Run manual RAG pipeline for debugging:**
  ```bash
  python source/agent.py
  ```
- **Customize LLM or embeddings:** Edit `.env` and source files to switch between Gemini, OpenAI, or Ollama.
- **Scale to Qdrant Cloud:** Use Qdrant Cloud for production-grade performance and security.

---

## 🏆 Enterprise Benefits
- **For Securities/Finance:** Get instant summaries of market-moving news, regulatory updates, and economic events.
- **For Media Monitoring:** Track breaking news and generate concise reports for clients or internal teams.
- **For Customer Service:** Provide automated, conversational news updates to users.

---

> Built for scalable, enterprise-grade news intelligence. Inspired by best practices from leading tech companies.
