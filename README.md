
<h1>Project Overview</h1>

This project is a **Multi-Modal PDF Question Answering Application** built with **Streamlit**. It allows users to upload PDFs, extract text and images, and ask questions about the content. The system leverages **llama3.2** and **llama3.2-vision**, a multimodal model capable of understanding both text and images, to provide concise and context-aware answers.  

The project demonstrates the use of **multimodal LLMs** to combine textual and visual information, enabling robust document understanding for research, reports, or any PDF-based knowledge source.

The workflow includes:

- **PDF Upload:** Users can upload any PDF document through the web interface.
- **Content Extraction:** Text, tables, and images are extracted from the PDF. Images are analyzed using the llama3.2-vision model to generate descriptive text.
- **Text Chunking & Indexing:** Extracted content is split into smaller chunks and converted into vector embeddings for efficient similarity search.
- **Question Answering:** Users can enter questions about the PDF content. The system retrieves relevant chunks and generates answers using the multimodal model.
- **Concise Responses:** Answers are limited to three sentences, ensuring they are clear and focused.

---

<h1>Problem Statement</h1>

Many PDFs contain complex information in both text and images, making it difficult to quickly find specific answers. Traditional search tools only handle text, ignoring visual content like charts, tables, and diagrams. This project addresses the need for a **multimodal question-answering system** that can understand and extract insights from both text and images in PDFs, providing concise and accurate responses to user queries.

---

 <h1>Solution Summary</h1>

This project provides a **multimodal PDF question-answering system** that combines text and image understanding. Users upload PDFs, and the system extracts text, tables, and images. Images are analyzed using the **llama3.2-vision** model to generate descriptive text. All content is split into manageable chunks and converted into embeddings for efficient similarity search. When a user asks a question, the system retrieves relevant chunks and generates a concise answer using the multimodal LLM, ensuring accurate and context-aware responses.

---

<h1>Tech Stack</h1>

This project leverages a combination of **technologies, frameworks, APIs, and tools** for building a multimodal PDF question-answering system.

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

This tech stack ensures the application can **process PDFs, handle multimodal data, index content efficiently, and answer user queries in real-time**.
