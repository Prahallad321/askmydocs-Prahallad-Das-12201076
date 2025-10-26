import os
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.utils.constants import PartitionStrategy

# Directories
pdfs_directory = 'multi-modal-rag/pdfs/'
figures_directory = 'multi-modal-rag/figures/'

# Prompt template
template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""

# Embeddings (text only)
embeddings = OllamaEmbeddings(model="llama3.2")
vector_store = InMemoryVectorStore(embeddings)

# Text-only model
text_model = OllamaLLM(model="llama3.2")
# Multi-modal model (text + image)
vision_model = OllamaLLM(model="llama3.2-vision")


# --- Functions ---
def upload_pdf(file):
    with open(pdfs_directory + file.name, "wb") as f:
        f.write(file.getbuffer())


def extract_text(file_path, use_vision=False):
    """Extract text from images in figures folder if use_vision=True"""
    model_to_use = vision_model if use_vision else text_model
    model_with_image = model_to_use.bind(images=[file_path])
    return model_with_image.invoke("Describe what you see in this picture.")


def load_pdf(file_path, use_vision=False):
    elements = partition_pdf(
        file_path,
        strategy=PartitionStrategy.HI_RES,
        extract_image_block_types=["Image", "Table"],
        extract_image_block_output_dir=figures_directory
    )

    # Extract text from PDF elements
    text_elements = [el.text for el in elements if el.category not in ["Image", "Table"]]

    # Extract text from images if using vision model
    if use_vision:
        for img_file in os.listdir(figures_directory):
            img_path = os.path.join(figures_directory, img_file)
            description = extract_text(img_path, use_vision=True)
            text_elements.append(f"Image Description ({img_file}): {description}")

    return "\n\n".join(text_elements)


def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return splitter.split_text(text)


def index_docs(texts):
    vector_store.add_texts(texts)


def retrieve_docs(query):
    return vector_store.similarity_search(query)


def answer_question(question, documents, use_vision=False):
    context = "\n\n".join([doc.page_content for doc in documents])
    prompt = ChatPromptTemplate.from_template(template)
    model_to_use = vision_model if use_vision else text_model
    chain = prompt | model_to_use
    return chain.invoke({"question": question, "context": context})


# --- Streamlit UI ---
uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf",
    accept_multiple_files=False
)

if uploaded_file:
    upload_pdf(uploaded_file)

    # Detect if PDF contains images to decide which model to use
    use_vision_model = False
    # Simple heuristic: check for figures folder after partitioning

    temp_elements = partition_pdf(
    os.path.join(pdfs_directory, uploaded_file.name),
    strategy=PartitionStrategy.HI_RES,
    extract_image_block_types=["Image", "Table"],
    extract_image_block_output_dir=figures_directory
)

    for el in temp_elements:
        if el.category in ["Image", "Table"]:
            use_vision_model = True
            break

    with st.spinner("Processing PDF..."):
        text = load_pdf(os.path.join(pdfs_directory, uploaded_file.name), use_vision=use_vision_model)
        chunks = split_text(text)
        index_docs(chunks)

    st.success("Indexing complete — you can ask a question now.")

    question = st.chat_input()

    if question:
        st.chat_message("user").write(question)
        related_docs = retrieve_docs(question)
        answer = answer_question(question, related_docs, use_vision=use_vision_model)
        st.chat_message("assistant").write(answer)
