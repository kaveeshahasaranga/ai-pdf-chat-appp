import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

# 🔴 වැදගත්: ඔයාගේ OpenAI API Key එක මෙතන " " ඇතුළේ දාන්න.
# (පස්සේ කාලෙක අපි මේක ආරක්ෂිතව හංගන්න ඉගෙන ගමු)
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY_HERE"

st.set_page_config(page_title="AI PDF ChatBot", page_icon="📚")
st.header("📚 AI PDF එකත් එක්ක Chat කරමු")

# 1. PDF එකක් Upload කරන්න ඉඩ දෙන්න
pdf = st.file_uploader("ඔයාගේ PDF එක මෙතනට දාන්න", type="pdf")

# PDF එකක් තියෙනවා නම් විතරක් ඉතිරි ටික කරන්න
if pdf is not None:
    # PDF එක කියවන්න
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    # අකුරු ටික පුංචි කෑලි (Chunks) වලට කඩන්න
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # 2. Smart පුස්තකාලය (Vector Store) හදන්න
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(chunks, embeddings)
    
    # 3. ප්‍රශ්නයක් අහන්න Box එකක් හදන්න
    user_question = st.text_input("ඔයාට දැනගන්න ඕන දේ අහන්න:")
    
    if user_question:
        # අපේ පුස්තකාලයේ ප්‍රශ්නයට අදාළ කෑලි හොයන්න
        docs = vector_store.similarity_search(user_question)
        
        # AI (ChatGPT) ලවා උත්තරේ ලස්සන කරගන්න
        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=user_question)
        
        st.success(response)