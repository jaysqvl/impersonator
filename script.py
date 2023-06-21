import streamlit as st
import os
from dotenv import load_dotenv
from streamlit_chat import message
from pypdf import PdfReader
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
import vecs
from supabase import create_client, Client
import openai

# SETUP BEFORE USE
load_dotenv()
oa_api_key = os.getenv('OPENAI_API_KEY')
sb_api_key = os.getenv('SUPABASE_API_KEY')
sb_prj_url = os.getenv('SUPABASE_PROJ_URL')
supabase: Client = create_client(sb_prj_url, sb_api_key)

def generate_answer():
    bot_response = st.session_state.conversation({'question': st.session_state.input_text})
    
    st.session_state.history.append({"message": st.session_state.input_text, "is_user": True})
    st.session_state.history.append({"message": bot_response, "is_user": False})

    for i, chat in enumerate(st.session_state.history):
        message(**chat, key=str(i)) #unpacking

    # Clears the input text
    st.session_state["input_text"] = ""

def ingest(pdfs):
    text = get_text(pdfs)
    text_batches = get_text_batches(text)

    return text_batches

def get_text(pdfs):
    text = ""
    for pdf in pdfs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_batches(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size = 1000,
        chunk_overlap = 0
    )
    return text_splitter.split_text(text)

# Creates Vector DB
def init_vector_db(text_batches):
    # embeddings = embeddings
    # db = FAISS.from_texts(texts = text_batches, embedding = embeddings)
    count = 0
    for text in text_batches:
        response = openai.Embedding.create(
            input = text,
            model = "text-embedding-ada-002"
        )
        supabase.table("langchainpythondemo").insert(
            {"id": count, "content": text, "embedding": response}
            ).execute()
        count += 1

    return supabase.table('langchainpythondemo').select("*").execute()

def get_conversation_chain(vector_db):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_db.asRetriever(),
        memory=memory
    )

    return conversation_chain


def main():
    st.header("Welcome to Jays Trainable Chatbot")
    with st.sidebar:
        st.subheader("Your documents")
        pdfs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                text_batches = ingest(pdfs)

                # Initialize the vector db
                vector_db = init_vector_db(text_batches)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vector_db)

    if "history" not in st.session_state:
        st.session_state.history = []

    st.text_input("Talk to the bot", key="input_text", on_change=generate_answer)

if __name__ == "__main__":
    main()