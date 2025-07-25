# 📰 VNExpress RAG Chatbot

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Vector DB](https://img.shields.io/badge/VectorDB-Qdrant-purple)](https://qdrant.tech/)
[![LLMs](https://img.shields.io/badge/LLMs-Gemini-yellow)](#tech-stack)
[![Frontend](https://img.shields.io/badge/Frontend-Streamlit-orange)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/RAG%20Stack-LangChain-darkgreen)](https://python.langchain.com/)

---
# [CLICK FOR DEMO](https://chat-vnexpress.streamlit.app/)
---

## 🧠 What is this?

**VNExpress RAG Chatbot** is an intelligent Q&A system powered by **LangChain**, **Qdrant**, and **Gemini**, designed to retrieve and summarize Vietnamese news in real time from [VNExpress](https://vnexpress.net/).

> 🚻 Retrieval-Augmented Generation (RAG) ensures accurate, up-to-date answers by combining vector search with LLM generation.

---

## 🚀 Why This Matters

📌 **News evolves fast. Static LLMs can’t keep up.**  
This chatbot dynamically pulls **fresh news** from VNExpress and answers your questions on the fly using powerful LLMs.

🔍 **Perfect for**:
- 📈 **Finance & Securities** teams: Track real-time news that impacts markets, regulations, and stocks.
- 📰 **Media Monitoring**: Monitor trends, summarize updates, and streamline content curation.
- 🤖 **Chatbot Integration**: Plug into intelligent RAG pipelines for answering news-related queries.

---

## 🛠️ Features at a Glance

| ✅ Feature                         | 🔍 Description                                               |
|----------------------------------|--------------------------------------------------------------|
| 📰 **VNExpress Crawler**         | Scrapes and extracts latest news from [vnexpress.net](https://vnexpress.net) |
| 🔎 **Qdrant Vector Search**      | Embeds articles and enables fast semantic search             |
| 🧠 **Gemini Summarizer**         | Generates concise answers from retrieved documents           |
| 🖥️ **Streamlit UI**             | Friendly web interface for exploration and Q&A               |
| ⚙️ **.env Config + Logging**     | Easy setup, safe API handling, and error visibility          |

---

## 📱 Tech Stack

| Layer            | Tool/Library                     |
|------------------|----------------------------------|
| Language Model   | `Gemini`                         |
| RAG Framework    | `LangChain`                      |
| Vector DB        | `Qdrant`                         |
| Web Interface    | `Streamlit`                      |
| Crawler          | `BeautifulSoup`, `Requests`      |
| Embedding        | `GoogleGenerativeAIEmbeddings`   |

---

## Project Structure
```bash
source/
├── app.py               # Streamlit app: UI + RAG response logic
├── config.py            # Load & validate environment variables
├── model.py             # Initialize Gemini LLM and embedding model
├── main.py              # Entry point for running full pipeline or app
├── crawl.py             # Main crawler orchestration entry point
│
├── crawler/             # VNExpress crawler module
│   ├── links.py         # Generate/capture article URLs
│   ├── content.py       # Parse & extract article content + metadata
│   └── utils.py         # Retry + helpers (e.g., `retry_on_failure`, validation)
│
├── storage/             # Vector DB storage and upload logic
│   ├── embed.py         # Embedding + chunking logic for documents
│   └── qdrant_store.py  # Qdrant connector + uploader

```

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
QDRANT_URL=<your-cloud-qdrant-url>
QDRANT_API_KEY=<your-qdrant-api-key>
QDRANT_COLLECTION=<your-collection-name>
GOOGLE_API_KEY=<your-google-api-key>
GOOGLE_EMBEDDING_MODEL_NAME=models/embedding-001
GEMINI_MODEL_NAME=gemini-2.0-flash
```

### 4. Crawl & Embed News Articles
```bash
python source/crawl.py
```

### 5. Launch Chatbot Web App
```bash
streamlit run source/app.py
```

---

## 🏆 Enterprise Benefits
- **For Securities/Finance:** Get instant summaries of market-moving news, regulatory updates, and economic events.
- **For Media Monitoring:** Track breaking news and generate concise reports for clients or internal teams.
- **For Customer Service:** Provide automated, conversational news updates to users.

---

## 🔆 Coming Soon: 
- Scheduled automatic news crawler.
- Outdated news deletion and continuous updates.
> Minh Huy 
