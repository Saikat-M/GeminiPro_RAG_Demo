import os
import numpy as np
import pandas as pd
import streamlit as st
from doc_to_text import extract_text
from dotenv import load_dotenv
import google.generativeai as genai

st.header('A basic RAG application using Gemini-Pro')

#method to get values of environment varaibles from .env file 
load_dotenv()

#Configuring Gemini API with the API key
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# # Initialize an empty DataFrame with columns 'Title' and 'Text'
dataF = pd.DataFrame(columns=['title', 'text'])

def embed_text(text):
    response  = genai.embed_content(model = 'models/embedding-001', content=text, task_type='retrieval_document')
    return response['embedding']

def get_vector_similarity(query, vector):
    query_vector = embed_text(query)
    return np.dot(query_vector, vector)

def get_most_similar_document(query):
    dataF['query_similarity'] = dataF['embeddings'].apply(lambda vector: get_vector_similarity(query, vector))
    title = dataF.sort_values('query_similarity', ascending=False)[['title','text']].iloc[0]['title']
    text = dataF.sort_values('query_similarity', ascending=False)[['title','text']].iloc[0]['text']
    return title, text

def rag(query):
    title, text  = get_most_similar_document(query)
    model = genai.GenerativeModel('gemini-pro')
    prompt = f"Answer this query:{query} and only use this context to answer:{text}"
    response = model.generate_content(prompt)
    return f"Source doc: {title}\n\n Response:\n{response.text}" 

uploaded_files = st.file_uploader("Upload a file", accept_multiple_files=True)
# st.text(uploaded_files)
if uploaded_files:
    for file in uploaded_files:
        with open(file.name, 'wb') as f:
            f.write(file.getbuffer())
        data_row = extract_text(file.name)
        dataF = pd.concat([dataF, data_row], ignore_index=True)

    # st.write(dataF)
    dataF['embeddings'] = dataF['text'].apply(embed_text)    
    query = st.chat_input(placeholder ='Please type in your question')
    if query: 
        with st.chat_message("user"):
            st.write(query)
            # st.line_chart(np.random.randn(30, 3))
        response = rag(query)
        # st.write(dataF)
        st.markdown(response)

