import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import time

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "messages" not in st.session_state:
    st.session_state.messages = []

st.set_page_config(
    page_title="DocQA Chatbot",
    page_icon="📄",
    layout="centered"
)

with st.sidebar:
    st.title("📄 Document Setup")
    uploaded_file = st.file_uploader(
        "Upload a PDF file", 
        type=["pdf"],
        accept_multiple_files=False
    )
    
    st.divider()
    st.markdown("### Model Settings")
    top_k = st.slider("Number of document chunks", 1, 5, 3)
    temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.7)
    max_tokens = st.slider("Max response length", 50, 500, 200)

st.title("📄 Document QA Chatbot")
st.caption("Upload a PDF and ask questions about its content")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def process_file(file_path):
    with st.spinner("Processing document..."):
        try:
            # Load and split document
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=512,
                chunk_overlap=50
            )
            texts = text_splitter.split_documents(documents)
            
            # Create embeddings
            embedding_model = HuggingFaceEmbeddings(
                model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            )
            
            # Create vector store
            vector_store = FAISS.from_documents(texts, embedding_model)
            return vector_store
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            return None

if uploaded_file and st.session_state.vector_store is None:
    with open("temp_file.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    vector_store = process_file("temp_file.pdf")
    if vector_store:
        st.session_state.vector_store = vector_store
        st.success("Document processed successfully! You can now ask questions.")
        os.remove("temp_file.pdf")  # Clean up temp file

@st.cache_resource(show_spinner=False)
def load_model():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    device = torch.device('mps')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
    ).to(device)
    return tokenizer, model

def generate_answer(question, top_k=3, temperature=0.7, max_tokens=200):
    if st.session_state.vector_store is None:
        return "Please upload a document first"
    
    try:
        docs = st.session_state.vector_store.similarity_search(question, k=top_k)
        context = "\n".join([d.page_content for d in docs])
        
        prompt = f"""### Instruction:
        Answer the question using ONLY the context below. Be concise.

        ### Context:
        {context}

        ### Question:
        {question}

        ### Response:
        """

        tokenizer, model = load_model()
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device).to(torch.device('mps'))
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_response.split("### Response:")[-1].strip()
        return response
    except Exception as e:
        return f"Error generating answer: {str(e)}"

if prompt := st.chat_input("Ask about your document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("Thinking..."):
            assistant_response = generate_answer(
                prompt,
                top_k=top_k,
                temperature=temperature,
                max_tokens=max_tokens
            )
        
        for chunk in assistant_response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})