import torch
import streamlit as st
from unsloth import FastLanguageModel
from transformers import TextStreamer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

st.title("Ask questions about the Leiden Guidelines PDF.")
st.caption("Chat with a Llama-3.1 model that has been fine-tuned on a RAFT dataset based on the Leiden Guidelines.")

# Load the data
loader = PyMuPDFLoader("input/leiden_guidelines.pdf")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
splits = text_splitter.split_documents(docs)

# Create Ollama embeddings and vector store
embeddings = OllamaEmbeddings(model="llama3")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

# Load custom RAFT model and tokenizer from Hugging Face
model, tokenizer = FastLanguageModel.from_pretrained("gizembrasser/FineLlama-3.1-8B")

def raft_llm(question, context):
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    messages = [{"from": "human", "value": formatted_prompt}]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    text_streamer = TextStreamer(tokenizer)
    outputs = model.generate(input_ids=inputs['input_ids'], streamer=text_streamer, max_new_tokens=128, use_cache=True)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# RAG setup
retriever = vectorstore.as_retriever()

def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def rag_chain(question):
        retrieved_docs = retriever.invoke(question)
        formatted_context = combine_docs(retrieved_docs)
        return raft_llm(question, formatted_context)

st.success(f"Loaded PDF successfully!")

# Ask a question about the PDF
prompt = st.text_input("Ask any question about the contents of the Leiden Guidelines.")

# Chat with the PDF
if prompt:
     result = rag_chain(prompt)
     st.write(result)
