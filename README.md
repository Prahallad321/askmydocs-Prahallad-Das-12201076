# askmydocs-Prahallad-Das-12201076
# 🧠 AskMyDocs
### ChatGPT File Q&A Clone 

AskMyDocs is an AI-driven web application that allows users to upload PDF documents and interact with them through natural language queries.  
It extracts, indexes, and retrieves content from your files using a RAG pipeline powered by Flask, LangChain, FAISS, and GPT-based models.

🚀 Built for the CloudCosmos Hackathon 2025

# Project Structure

askmydocs/
├── app.py                   # Flask entry point
├── requirements.txt         # Dependencies
├── static/
│   ├── style.css            # Custom CSS styles
│   └── script.js            # Frontend interactivity
├── templates/
│   └── index.html           # Web interface for chat and upload
├── utils/
│   ├── pdf_utils.py         # PDF text extraction logic
│   ├── rag_pipeline.py      # Text chunking, embeddings, and FAISS vector store
│   └── llm_utils.py         # LLM query and response generation
├── data/
│   └── uploaded_files/      # Uploaded PDF storage
└── vector_store/
    └── faiss_index/         # Saved FAISS index for retrieval
