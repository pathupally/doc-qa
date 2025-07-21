# 🤖 LangChain Local RAG Chatbot

A fully self-contained Retrieval-Augmented Generation (RAG) chatbot built with LangChain, OpenAI, and FAISS. This application allows you to load documents (PDF or TXT files), index them using vector embeddings, and ask questions about their content using natural language.

---

## 🚀 Features

- **Document Loading**: Supports PDF and TXT files with automatic chunking.
- **Vector Search**: FAISS-based similarity search for efficient document retrieval.
- **RAG Pipeline**: Combines retrieval with OpenAI's language models for accurate answers.
- **Dual Interface**: Command-line and Streamlit web interfaces.
- **Source Tracking**: See which documents were used to generate each answer.
- **Conversation Logging**: Automatic logging of all Q&A interactions.
- **Modular Design**: Clean, well-documented, and easily extensible codebase.
- **Persistent Index**: FAISS index is saved and can be reused across sessions.
- **Error Handling**: Comprehensive error handling and graceful degradation.

---

## 📋 Requirements

- **Python**: 3.10+ (tested with Python 3.13)
- **OpenAI API Key**: Required for embeddings and language model
- **Memory**: 2GB+ RAM (for FAISS indexing)
- **Storage**: ~100MB for dependencies + space for your documents

---

## 🛠️ Installation

### 1. Clone and Setup

```bash
git clone https://github.com/pathupally/langhcain-rag-chatbot
cd langhcain-rag-chatbot

python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Environment Configuration

```bash
cp env.example .env
nano .env  # or use your preferred editor
```

Add your OpenAI API key to the `.env` file:
```
OPENAI_API_KEY=sk-your-openai-api-key-here
OPENAI_MODEL_NAME=gpt-4.1
```

### 3. Add Your Documents

Place your documents in the `data/` directory:
```
data/
├── document1.pdf
├── document2.txt
├── research_paper.pdf
└── manual.txt
```

---

## 🎯 Quick Start

### Command Line Interface

```bash
python src/app.py
# With custom options
python src/app.py --model gpt-4 --temperature 0.2 --rebuild
python src/app.py --help
```

### Streamlit Web Interface

```bash
streamlit run src/app.py
```
The web interface will open at `http://localhost:8501`

---

## 📁 Project Structure

```
langhcain-rag-chatbot/
├── data/                # Source documents (PDF/TXT)
├── docs/                # Documentation
│   └── API_REFERENCE.md
├── outputs/             # Logs, FAISS index, conversation logs
│   ├── faiss_index
│   ├── documents.pkl
│   └── conversation_log.jsonl
├── src/                 # Source code
│   ├── loaders.py
│   ├── embedder.py
│   ├── rag_chain.py
│   └── app.py
├── venv/                # Virtual environment
├── .env.example         # Example environment file
├── requirements.txt     # Required packages
├── test_basic.py        # Basic functionality tests
├── README.md            # This file
└── LICENSE              # MIT License
```

---

## 🏗️ How It Works

### Document Processing Pipeline

```
Documents → Chunking → Embedding → FAISS Index → Retrieval → RAG Chain → Response
```

**Step-by-step process:**
1. **Document Loading**: Documents are loaded using LangChain's document loaders.
2. **Text Chunking**: Large documents are split into manageable chunks.
3. **Embedding Generation**: Text chunks are converted to vectors using OpenAI's embedding model.
4. **Vector Storage**: Embeddings are stored in a FAISS index for fast similarity search.
5. **Query Processing**: User questions are embedded and similar document chunks are retrieved.
6. **Answer Generation**: Retrieved context is combined with the question to generate accurate answers.
7. **Source Attribution**: The system tracks which documents were used for each answer.

---

## 🖥️ Usage Details

### CLI Mode Features

- `quit`, `exit`, `q` - Exit the application
- `stats` - View system statistics
- `rebuild` - Rebuild the FAISS index

**Command Line Options:**
- `--rebuild` - Force rebuild the FAISS index
- `--model` - Choose OpenAI model (gpt-3.5-turbo, gpt-4, gpt-4-turbo)
- `--temperature` - Control response creativity (0.0-1.0)
- `--max-tokens` - Maximum response length
- `--data-dir` - Custom data directory path

### Streamlit Mode Features

- Chat-like interaction with message history
- Real-time system statistics in sidebar
- Source document viewing with expandable sections
- Configuration options for model and parameters
- Index rebuild functionality
- Sample questions for testing

---

## 🧪 Sample Questions

- **Q:** "What is machine learning?"
  - **A:** Machine Learning is a subset of AI that provides systems the ability to automatically learn and improve from experience without being explicitly programmed.
- **Q:** "How does RAG work?"
  - **A:** RAG combines document retrieval with language model generation. The process involves: 1) Loading and chunking documents, 2) Generating embeddings, 3) Storing in vector database, 4) Retrieving relevant chunks for queries, 5) Using retrieved context to generate accurate answers.

---

## ⚙️ Configuration

### Environment Variables

```
# Required
OPENAI_API_KEY=sk-your-api-key

# Optional
OPENAI_MODEL_NAME=gpt-3.5-turbo
```

### Advanced Configuration

- **Document Chunking** (`src/loaders.py`):  
  `chunk_size=1000`, `chunk_overlap=200`
- **Embedding Model** (`src/embedder.py`):  
  `model_name="text-embedding-ada-002"`
- **RAG Chain** (`src/rag_chain.py`):  
  `model_name="gpt-3.5-turbo"`, `temperature=0.1`, `max_tokens=1000`

---

## 🐞 Troubleshooting

- **OPENAI_API_KEY not found**: Check your `.env` file.
- **No documents found**: Add documents to `data/` directory.
- **Error loading index**: Rebuild the index (`python src/app.py --rebuild`).
- **Out of memory**: Reduce chunk size in `src/loaders.py`.
- **Module not found**: Activate virtual environment and install requirements.

---

## 🧩 Extending the System

- **Custom Document Loaders**: Add support for new file types in `src/loaders.py`.
- **Custom Embedding Models**: Modify `FAISSEmbedder` in `src/embedder.py`.
- **Batch Processing**: For large document collections, process in batches and periodically rebuild the index.

---

## 📝 Logging and Outputs

- **outputs/faiss_index**: FAISS vector database
- **outputs/documents.pkl**: Document metadata and content
- **outputs/conversation_log.jsonl**: All Q&A interactions

---

## 🧪 Testing

Run the test suite to verify installation and basic functionality:

```bash
python test_basic.py
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Add tests if applicable
5. Run tests: `python test_basic.py`
6. Submit a pull request

**Code Style:**  
- Follow PEP 8 guidelines  
- Add docstrings and type hints  
- Add logging for important operations

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain)
- [OpenAI](https://openai.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Streamlit](https://streamlit.io/)
- [PyPDF](https://pypdf.readthedocs.io/)

---

**Happy Chatting! 🤖✨**
