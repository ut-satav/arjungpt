import streamlit as st
import pandas as pd
from langchain_community.llms import Ollama
llm = Ollama(model="llama2")

st.write(""" # अर्जुन ARJUN """)

current_query =  st.text_input(label='Label', label_visibility='hidden', 
                               placeholder='Type your question here')

#@st.cache_data
def askingllm(query):
    st.write(llm.invoke(query))

askingllm(current_query) 
# df = pd.read_csv("my_data.csv")
# st.line_chart(df)