import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
st.set_page_config(page_title="Chat Model")

st.markdown("# Chat Model")
st.sidebar.header("Chat Model")

def overwrite_text_file(filename, new_content):
        with open(filename, "w") as file:
            file.write(new_content)

def get_chat_model(question):
    import os
    api_key=os.getenv('api_key')
    st.write(api_key)
    # Load data
    loader = TextLoader("context.txt")
    docs = loader.load()
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)
    # Define the embedding model
    embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=api_key)
    # Create the vector store
    vector = FAISS.from_documents(documents, embeddings)
    # Define a retriever interface
    retriever = vector.as_retriever()
    # Define LLM
    model = ChatMistralAI(mistral_api_key=api_key)
    # Define prompt template
    prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

    <context>
    {context}
    </context>

    Question: {input}""")

    # Create a retrieval chain to answer questions
    document_chain = create_stuff_documents_chain(model, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({"input": question})
    
    return response["answer"]


def main():
    # context = st.text_input(label='Context Article', 
    #                            placeholder='Paste your article here',key='context')
    # if not context:
    #     st.info("Please enter some context...")

    # filename = "context.txt"
    # new_content = context

    #overwrite_text_file(filename, new_content)

    question =  st.text_input(label='Question', 
                               placeholder='Type your question here',key='question')+'?'
    # Wait until the question is not empty
    if not question:
        st.info("Please enter a question...")

    answer = 'None'
    #and context    
    if question:
        # Run the selected function based on the user's choice
        answer = get_chat_model(question)
        # Once text input is provided, display it
        st.write(answer)

if __name__ == "__main__":
    main()