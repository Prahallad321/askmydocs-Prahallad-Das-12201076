
<h1>Project Overview</h1>

This project is a **Multi-Modal PDF Question Answering Application** built with **Streamlit**.  
It allows users to upload PDFs, extract both text and images, and ask questions about the content.  
The system leverages **Llama 3.2** and **Llama 3.2-Vision**, multimodal large language models capable of understanding both textual and visual information, to provide concise and context-aware answers.  

The project demonstrates the integration of **LangChain** and **Retrieval-Augmented Generation (RAG)** for intelligent document analysis.  
By combining **vector-based retrieval** and **multimodal LLM reasoning**, the app can understand complex documents containing text, tables, and images — making it ideal for research papers, reports, and knowledge-heavy PDFs.

---

### Workflow Overview

- **PDF Upload:** Users can upload any PDF document through the Streamlit web interface.  
- **Content Extraction:** Text, tables, and images are extracted using the **Unstructured** library. Images are analyzed with **Llama 3.2-Vision** to generate descriptive text.  
- **Text Chunking & Indexing:** Extracted content is split into smaller chunks and converted into **vector embeddings** using **LangChain**, enabling efficient similarity search.  
- **Retrieval-Augmented Generation (RAG):** When a user asks a question, the system retrieves the most relevant chunks and uses **Llama 3.2** to generate a concise answer based on both text and image context.  
- **Concise Responses:** All answers are limited to three sentences to keep them clear, relevant, and focused.


---

<h1>Problem Statement</h1>

Many PDFs contain complex information in both text and images, making it difficult to quickly find specific answers. Traditional search tools only handle text, ignoring visual content like charts, tables, and diagrams. This project addresses the need for a **multimodal question-answering system** that can understand and extract insights from both text and images in PDFs, providing concise and accurate responses to user queries.

---

 <h1>Solution Summary</h1>

This project provides a **multimodal PDF question-answering system** that combines text and image understanding. Users upload PDFs, and the system extracts text, tables, and images. Images are analyzed using the **llama3.2-vision** model to generate descriptive text. All content is split into manageable chunks and converted into embeddings for efficient similarity search. When a user asks a question, the system retrieves relevant chunks and generates a concise answer using the multimodal LLM, ensuring accurate and context-aware responses.

---

<h1>Tech Stack</h1>

### Programming Language
- **Python 3.11**

### Web Framework
- **Streamlit 1.50.0** – Interactive web application interface.

### Natural Language Processing & LLM
- **llama3.2** – Text understanding.
- **llama3.2-vision** – Multimodal model for both text and image processing.
- **LangChain (langchain-core, langchain-community, langchain-ollama)** – Orchestrates LLM prompts, memory, and vector stores.
- **Ollama 0.6.0** – API for llama models and embeddings.

### PDF & Document Processing
- **unstructured 0.18.15** – High-resolution PDF parsing (text, tables, images).
- **PyPDF2 / pypdf 6.1.3** – Optional PDF parsing support.
- **pdf2image 1.17.0** – Convert PDF pages to images.
- **Pillow 11.3.0** – Image processing.
- **Python-dotenv 1.1.1** – Environment variable management.

### Text Chunking & Embeddings
- **langchain-text-splitters 1.0.0** – RecursiveCharacterTextSplitter for text chunking.
- **OllamaEmbeddings** – Converts text chunks into vector embeddings.
- **InMemoryVectorStore** – Stores embeddings for similarity search.

### Computer Vision & Image Analysis
- **opencv-python 4.12.0** – Image handling.
- **google-cloud-vision 3.11.0** – Optional cloud-based image analysis.
- **pi_heif / pikepdf** – Image and PDF handling.

### ML / Deep Learning
- **torch 2.9.0** – Backend for LLMs and embeddings.
- **transformers 4.57.1** – Optional HuggingFace support for LLMs.
- **timm 1.0.21** – Vision model utilities.

### Utilities & Tools
- **numpy, pandas, matplotlib, altair** – Data handling and visualization.
- **rapidfuzz** – Fuzzy matching for text.
- **watchdog** – File monitoring.
- **docker (optional)** – Containerization and deployment.
- **git / GitPython** – Version control.
  
### Cloud
**Render / AWS**

---

<h1>Project Structure</h1>

```bash
multi-modal-rag/
├── app.py                     # Main Streamlit app
├── pdfs/                      # Uploaded PDFs
├── figures/                   # Extracted images
├── requirements.txt           # Python dependencies
└── README.md                  # Documentation
```

---

<h1>Setup Instructions</h1>

### 1. Create a Virtual Environment
```bash
python -m venv venv
.\venv\Scripts\activate
```
### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
### 3. Run the Streamlit Application
```bash
streamlit run app.py
```

---

<h1>Demo Video</h1>
**YouTube Demo URL:** https://youtu.be/aVn44gjAbcg

---

<h1>Features</h1>

- **Multimodal Understanding:** Supports both text and image content from PDFs using **llama3.2-vision**.  
- **PDF Upload & Processing:** Easily upload PDFs; automatically extracts text, tables, and images.  
- **Text Chunking & Embeddings:** Splits content into manageable chunks and indexes them for efficient similarity search.  
- **Context-Aware Q&A:** Retrieves relevant information and generates concise answers to user questions.  
- **Interactive Web Interface:** Built with **Streamlit**, providing a simple and user-friendly interface.

---

<h1>Technical Architecture</h1>

The system is designed as a **multimodal PDF question-answering platform** that integrates PDF processing, vector embeddings, and a language model capable of handling both text and images.

### How It Works

1. **PDF Upload:** Users upload a PDF through the Streamlit interface.  
2. **Content Extraction:** Text, tables, and images are extracted. Images are analyzed by the **llama3.2-vision** model to generate descriptive text.  
3. **Text Chunking & Indexing:** Extracted content is split into chunks and converted into embeddings, stored in an in-memory vector store.  
4. **User Query:** The user submits a question through the chat interface.  
5. **Retrieval & Answering:** Relevant chunks are retrieved based on similarity, and the multimodal LLM generates a concise, context-aware answer.  



### ASCII Diagram

```text
User Uploads PDF
          |
          v
Extract Text & Images from PDF
          |
          v
Describe Images using llama3.2-vision
          |
          v
Split Text into Chunks & Generate Embeddings
          |
          v
Store in InMemoryVectorStore
          |
          v
User Asks Question in Chat Interface
          |
          v
Retrieve Relevant Chunks
          |
          v
Generate Concise Answer with LLM
          |
          v
Display Answer to User
```

---

 <h1>Testing</h1>


### 1. **PDF Upload Validation**
   - Only allows files with `.pdf` extension.
   - Checks that the uploaded file is not empty.

### 2. **Content Extraction Verification**
   - Ensures text, tables, and images are correctly extracted.
   - Validates that extracted images are processed by the **llama3.2-vision** model.

### 3. **Text Chunking and Embeddings**
   - Confirms that text is split into chunks of the specified size with proper overlap.
   - Verifies that each chunk is successfully converted into vector embeddings.

### 4. **Question Answering**
   - Checks that relevant chunks are retrieved from the vector store based on user queries.
   - Ensures that the model produces concise answers and handles unknown questions gracefully.

### 5. **Manual Testing**
   - Upload various sample PDFs containing text, tables, and images.
   - Enter different types of questions and confirm that the system returns accurate, context-aware responses.

---

<h1>Acknowledgment</h1>

I would like to express my sincere gratitude to the open-source community and the developers behind the technologies that made this project possible. This work was built using **Meta AI’s Llama 3.2** and **Llama 3.2-Vision** models, which provide advanced multimodal understanding of both text and images. I would also like to thank the **LangChain** team for their powerful framework that simplifies building Retrieval-Augmented Generation (RAG) systems, and **Streamlit** for enabling an intuitive and interactive web interface. Additionally, tools like **Hugging Face** and **Ollama** played a vital role in seamless model integration and deployment. This project is a result of the collective innovation and collaboration of the AI research and open-source communities.



<h1>References</h1>

- **Llama 3.2 (Meta AI)**  
  - Overview & Blog: [https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices)  
  - Model Access via Ollama: [https://ollama.com/library/llama3.2](https://ollama.com/library/llama3.2)  
  - GitHub Repository: [https://github.com/meta-llama/llama3](https://github.com/meta-llama/llama3)

- **LangChain**  
  - Official Website: [https://www.langchain.com](https://www.langchain.com)  
  - Python Documentation: [https://python.langchain.com/docs/](https://python.langchain.com/docs/)  
  - API Reference: [https://python.langchain.com/docs/reference/](https://python.langchain.com/docs/reference/)  
  - Community GitHub: [https://github.com/hwchase17/langchain](https://github.com/hwchase17/langchain)

- **Unstructured (PDF Parsing)**  
  - GitHub Repository: [https://github.com/Unstructured-IO/unstructured](https://github.com/Unstructured-IO/unstructured)

- **Streamlit**  
  - Official Website: [https://streamlit.io](https://streamlit.io)

- **PyPDF / PyPDF2**  
  - PyPDF2 on PyPI: [https://pypi.org/project/PyPDF2/](https://pypi.org/project/PyPDF2/)  
  - PyPDF on PyPI: [https://pypi.org/project/pypdf/](https://pypi.org/project/pypdf/)

- **pdf2image**  
  - PyPI: [https://pypi.org/project/pdf2image/](https://pypi.org/project/pdf2image/)

- **PyTorch / Transformers**  
  - PyTorch: [https://pytorch.org](https://pytorch.org)  
  - Hugging Face Transformers: [https://huggingface.co/docs/transformers](https://huggingface.co/docs/transformers)































