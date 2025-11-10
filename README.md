# RAG-Based Document Q&A System

A Streamlit application that enables question-answering over PDF research papers using Retrieval-Augmented Generation (RAG) with LangChain, Groq LLM, and FAISS vector search.

## Overview

This application allows users to upload PDF research papers, create vector embeddings from the documents, and ask questions that are answered based on the content of those papers. It uses a RAG architecture to retrieve relevant document chunks and generate accurate responses.

## Features

- **PDF Document Loading**: Automatically loads all PDF files from a specified directory
- **Vector Embeddings**: Creates embeddings using Ollama for semantic search
- **FAISS Vector Store**: Efficient similarity search over document embeddings
- **Groq LLM Integration**: Uses Llama3-8b-8192 model for response generation
- **Interactive UI**: Clean Streamlit interface for querying documents
- **Document Similarity View**: Displays relevant document chunks used for answering
- **Response Time Tracking**: Monitors query processing performance

## Architecture

The application follows a standard RAG pipeline:

1. **Document Ingestion**: PDFs are loaded from the `research_papers` directory
2. **Text Splitting**: Documents are chunked into 1000-character segments with 50-character overlap
3. **Embedding Generation**: Text chunks are converted to vector embeddings using Ollama
4. **Vector Storage**: Embeddings are stored in FAISS for efficient retrieval
5. **Query Processing**: User questions are embedded and similar chunks are retrieved
6. **Answer Generation**: Retrieved context is passed to Groq's Llama3 model to generate answers

## Prerequisites

- Python 3.8+
- Ollama installed and running locally
- Groq API key
- PDF research papers in a `research_papers` directory

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <project-directory>
```

2. Install required packages:
```bash
pip install streamlit langchain-groq langchain-openai langchain-community langchain-ollama faiss-cpu python-dotenv pypdf
```

3. Set up environment variables:
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```

4. Create the research papers directory:
```bash
mkdir research_papers
```

5. Add your PDF files to the `research_papers` directory

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Click the **"Document Embedding"** button to initialize the vector database (first-time setup)

3. Enter your question in the text input field

4. View the generated answer and expand the "Document similarity search" section to see relevant source passages

## Code Structure

### Key Components

**Environment Setup**
- Loads environment variables from `.env` file
- Initializes Groq API key and LLM instance

**LLM Configuration**
```python
llm = ChatGroq(groq_api_key=groq_api_key, model="Llama3-8b-8192")
```

**Prompt Template**
- Instructs the model to answer based only on provided context
- Ensures accurate, context-grounded responses

**Vector Embedding Function**
```python
def create_vector_embeddings():
```
- Uses Ollama embeddings for local processing
- Loads PDFs from directory using `PyPDFDirectoryLoader`
- Splits documents with `RecursiveCharacterTextSplitter`
- Creates FAISS vector store from document chunks
- Stores everything in `st.session_state` for persistence

**Query Processing**
- Creates document chain combining LLM and prompt
- Sets up retriever from FAISS vector store
- Creates retrieval chain for end-to-end processing
- Tracks response time for performance monitoring

## Configuration

### Adjustable Parameters

- **Chunk Size**: `chunk_size = 1000` - Size of text segments
- **Chunk Overlap**: `chunk_overlap = 50` - Overlap between chunks for context continuity
- **Document Limit**: `docs[:50]` - Number of documents to process (currently limited to 50)
- **LLM Model**: `model="Llama3-8b-8192"` - Can be changed to other Groq models

## Known Issues

1. **Typo in Code**: Line 41 has `as_retriver()` which should be `as_retriever()`
2. **Variable Naming**: Line 40 uses `st.session_state.vector` but the variable is stored as `st.session_state.vectors` (plural)
3. **Document Limit**: Only processes first 50 documents - may need adjustment for larger datasets

## Bug Fixes Required

Replace line 40-41:
```python
# Current (incorrect)
retriever = st.session_state.vector.as_retriver()

# Corrected
retriever = st.session_state.vectors.as_retriever()
```

## Performance Considerations

- Initial embedding creation can take time depending on document count and size
- FAISS provides fast similarity search even with large document collections
- Response time is tracked and printed to console
- Consider GPU acceleration for Ollama embeddings with large datasets

## Dependencies

- `streamlit` - Web application framework
- `langchain-groq` - Groq LLM integration
- `langchain-openai` - OpenAI embeddings (imported but not used)
- `langchain-community` - Community tools (FAISS, document loaders)
- `langchain-ollama` - Ollama embeddings
- `faiss-cpu` - Vector similarity search
- `python-dotenv` - Environment variable management
- `pypdf` - PDF processing

## Future Enhancements

- Support for multiple file formats (Word, TXT, etc.)
- Adjustable chunk size through UI
- Multiple embedding model options
- Chat history and conversation context
- Document upload through UI instead of directory
- Better error handling and validation
- Support for OpenAI embeddings as alternative


