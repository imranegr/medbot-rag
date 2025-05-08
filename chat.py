import os
import streamlit as st
from dotenv import load_dotenv  

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

load_dotenv()  

DB_FAISS_PATH = "VectorDatabase"
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
HF_TOKEN = os.getenv("HF_TOKEN")  

def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def load_llm(repo_id):
    return HuggingFaceEndpoint(
        repo_id=repo_id,
        task="text-generation",
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        max_new_tokens=256,
        repetition_penalty=1.1,
        huggingfacehub_api_token=HF_TOKEN,
    )

def custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

custom_prompt_template = """ 
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say that you don't know. Don't try to make up an answer. 
Don't provide anything out of the given context.
If the user provides a general question not related to the context, say please ask a question related to the context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

embedding_model = get_embedding_model()
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": custom_prompt(custom_prompt_template)}
)

def main():
    st.title("LangChain with HuggingFace LLMs")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    prompt = st.chat_input("Ask a question about the document")
    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        response = qa_chain({"query": prompt})
        answer = response['result']
        source_doc = response['source_documents'][0]
        page_number = source_doc.metadata.get("page", "N/A")
        content = source_doc.page_content

        st.chat_message("assistant").markdown(answer)
        with st.expander(f"Source Document (Page {page_number})"):
            st.write(content)


        st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()
