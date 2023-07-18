import os
import streamlit as st
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter


user_api_key = st.sidebar.text_input(
    label="OpenAI API key",
    placeholder="Paste your openAI API key here",
    type="password")

uploaded_file = st.sidebar.file_uploader("upload", type="pdf")

os.environ['OPENAI_API_KEY'] = user_api_key

text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 2000,
        chunk_overlap  = 100,
        length_function = len,
    
)

if uploaded_file :
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(file_path=tmp_file_path)  
    data = loader.load_and_split(text_splitter)

    embeddings = OpenAIEmbeddings()
    vectors = FAISS.from_documents(data, embeddings)

    chain = ConversationalRetrievalChain.from_llm(llm = ChatOpenAI(temperature=0.0,model_name='gpt-3.5-turbo-16k'),
                                                                      retriever=vectors.as_retriever())

    # This function takes a query as input and returns a response from the ChatOpenAI model.
    def conversational_chat(query):

        # The ChatOpenAI model is a language model that can be used to generate text, translate languages, write different kinds of creative content, and answer your questions in an informative way.
        result = chain({"question": query, "chat_history": st.session_state['history']})
        # The chat history is a list of tuples, where each tuple contains a query and the response that was generated from that query.
        st.session_state['history'].append((query, result["answer"]))
        
        # The user's input is a string that the user enters into the chat interface.
        return result["answer"]
    
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Feel free to ask about anything regarding this" + uploaded_file.name]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hi!"]
        
    # This container will be used to display the chat history.
    response_container = st.container()
    # This container will be used to display the user's input and the response from the ChatOpenAI model.
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            
            user_input = st.text_input("Input:", placeholder="Please enter your message regarding the PDF data.", key='input')
            submit_button = st.form_submit_button(label='Send')
            
        if submit_button and user_input:
            output = conversational_chat(user_input)
            
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")