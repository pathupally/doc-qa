from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

loader = PyPDFLoader("your_document.pdf") 
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

# 2. Create embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vector_store = FAISS.from_documents(texts, embedding_model)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# 3. Load generator model
model_name = "HuggingFaceTB/SmolLM-3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 4. RAG generation function
def generate_answer(question):
    # Retrieve relevant chunks
    docs = retriever.get_relevant_documents(question)
    context = "\n".join([d.page_content for d in docs])
    
    # Format prompt
    prompt = f"""Use the following context to answer the question. Be concise.
    
    Context:
    {context}
    
    Question: {question}
    Answer:"""
    
    # Generate response
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 5. Test the chatbot
question = "What is the main topic discussed in section 4?"
answer = generate_answer(question)
print(f"Answer: {answer}")