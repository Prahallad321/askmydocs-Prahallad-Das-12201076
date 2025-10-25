
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
