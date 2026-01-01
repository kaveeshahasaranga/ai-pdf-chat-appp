import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain

# 🔴 නොමිලේ පාවිච්චි කරන්න Google Gemini API Key එක මෙතනට දාන්න
# ඔයාට මේක https://aistudio.google.com/ එකෙන් නොමිලේ ගන්න පුළුවන්
os.environ["GOOGLE_API_KEY"] = "YOUR_GEMINI_API_KEY_HERE"

st.set_page_config(page_title="AI PDF ChatBot", page_icon="📚")
st.header("📚 AI PDF එකත් එක්ක Chat කරමු (Free AI)")

pdf = st.file_uploader("ඔයාගේ PDF එක මෙතනට දාන්න", type="pdf")

if pdf is not None:
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Google Gemini පාවිච්චි කරලා Embeddings හදමු
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embeddings)
    
    user_question = st.text_input("ඔයාට දැනගන්න ඕන දේ අහන්න:")
    
    if user_question:
        docs = vector_store.similarity_search(user_question)
        
        # Gemini AI Model එක පාවිච්චි කරමු
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=user_question)
        
        st.success(response)