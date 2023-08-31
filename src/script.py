import streamlit as st
import os
from dotenv import load_dotenv
from streamlit_chat import message as st_message
from pypdf import PdfReader
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import SupabaseVectorStore
from supabase import create_client, Client
import openai
import pinecone
import requests

# Loads the .env file
load_dotenv()

# Global APP Settings
change_default_llm = False
change_default_vdb = False
llm = "OpenAI"
llm_model = "gpt-3.5-turbo"
vector_db = "Supabase"

# Global Variables
temp_key = ""
oa_api_key = os.getenv('OPENAI_API_KEY')
sb_api_key = os.getenv('SUPABASE_API_KEY')
sb_proj_url = os.getenv('SUPABASE_PROJ_URL')
hf_api_key = None
pc_api_key = None
supabase: Client = create_client(sb_proj_url, sb_api_key)
pinecone.init(api_key=os.getenv('PINECONE_API_KEY'), environment="us-west1-gcp")

def generate_answer(user_input):
    response = st.session_state.conversation({'question': user_input})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st_message(message.content, is_user=True)
        else:
            st_message(message.content, is_user=False)
    
    # Clears the input text
    # st.session_state["input_text"] = ""

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
        chunk_overlap = 100
    )
    return text_splitter.split_text(text)

# Creates Vector DB
def init_vector_db(text_batches):
    print(text_batches)
    embeddings = OpenAIEmbeddings(openai_api_key=oa_api_key)
    vector_db = SupabaseVectorStore.from_texts(text_batches, embeddings, client=supabase, table_name="documents")
    
    # for switching to pinecone
    # vecter_db = pinecone.create_index("python-index", dimension=1536, metric="cosine")

    return vector_db

def get_conversation_chain(vector_db):
    
    llm = ChatOpenAI(openai_api_key=oa_api_key, model="gpt-3.5-turbo")

    # for switching to huggingface to use different LLMs
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_db.as_retriever(search_type="similarity"),
        memory=memory
    )

    return conversation_chain

# Checks the validity of the OpenAI API Key before applying it
def check_openai_api_key(api_key):
    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    response = requests.post(
        "https://api.openai.com/v1/engines/text-davinci-001/completions",
        headers=headers,
        json={
            "prompt": "Once upon a time,",
            "max_tokens": 5
        }
    )

    if response.status_code == 200:
        print("Your OpenAI API key is valid.")
        return True
    elif response.status_code == 401:
        print("Your OpenAI API key is invalid or unauthorized.")
        return False
    else:
        print(f"An error occurred: {response.status_code}, {response.json()}")
        return False

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Setters for the API Keys
def change_openai_api_key(api_key):
    global oa_api_key
    os.environ['OPENAI_API_KEY'] = api_key
    oa_api_key = api_key

def change_supabase_api_key(api_key):
    global sb_api_key
    os.environ['SUPABASE_API_KEY'] = api_key
    sb_api_key = api_key

def change_huggingface_api_key(api_key):
    # No default key
    global hf_api_key
    hf_api_key = api_key

def change_pinecone_api_key(api_key):
    # No default key
    global pc_api_key
    pc_api_key = api_key

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def model_select_section():
    global oa_api_key
    global hf_api_key

    # LLM Provider Selector Dropdown
    llm_brand_chosen = st.selectbox("Select LLM Provider", ["(default) OpenAI", "HuggingFace"])

    # What shows up if you select OpenAI as your LLM provider
    if llm_brand_chosen == "(default) OpenAI":
        # Toggle for changing the default GPT version
        activated = st.toggle("Change GPT Version (Requires own api key)", value=False)
        
        # Warning message
        st.write("WARNING: Changing default GPT version requires your own OpenAI API Key which will bill you for use of both the Ada Embedding model and selected LLM model")
        st.write("Toggle off sets the model back to the default LLM model automatically and uses the hosts API key (e.g. gpt-3.5-turbo)")

        # Model Selector Dropdown
        display = ("(default) gpt-3.5-turbo (recommended, cheapest)", 
                    "gpt-3.5-turbo-16k (4x context)", 
                    "gpt-4 (expensive)", 
                    "gpt-4-32k (most expensive, 4x context))")
        options = list(range(len(display))) # Enumerates the display list
        llm_brand_model_chosen = st.selectbox("Select LLM Model", 
                                                options,
                                                format_func=lambda x: display[x],
                                                disabled=(not activated))

        # API Key Input Text Box
        if (activated & (llm_brand_model_chosen != 0)):
            llm_api_key = st.text_input("Enter Your Own OpenAI API Key")

            if llm_api_key:
                if (check_openai_api_key(llm_api_key)):
                    change_openai_api_key(llm_api_key)
                    st.write("OpenAI API Key Successfully Updated!")
                else:
                    st.write("OpenAI API Key Invalid or Unauthorized. Please Try Again Or Use The Default Model.")

    # What shows up if you select HuggingFace as your LLM provider
    elif llm_brand_chosen == "HuggingFace":
        # Toggle for changing the default HuggingFace LLM model
        activated = st.toggle("Change Default HF Model", value=False)

        display = ("(default) google/flan-t5-xxl (recommended, cheapest)", 
                    "google/flan-t5-xxl (4x context)", 
                    "google/flan-t5-xxl (most expensive, 4x context)")
        options = list(range(len(display))) # Enumerates the display list
        llm_brand_model_chosen = st.selectbox("Select LLM Model",
                                                options,
                                                format_func=lambda x: display[x], 
                                                disabled=(not activated))
        
        llm_api_key = st.text_input("Enter HuggingFace API Key")
        if llm_api_key != "":
            hf_api_key = llm_api_key

def vector_db_select_section():
    global sb_api_key
    global sb_proj_url

    # VectorDB Provider Selector Dropdown
    vdb_activated = st.toggle("Use Your Own VectorDB Provider", value=False)
    vdb_chosen = st.selectbox("Select VectorDB Provider", ["Supabase", "Pinecone"], disabled=(not vdb_activated))
    if vdb_activated:
        vdb_api_key = st.text_input("Enter VectorDB API Key")
        changed = st.button("Save Key", change_supabase_api_key, args = [vdb_api_key])
        if (changed):
            st.write("Supabase API Key Successfully Updated!")
        else:
            st.write("Supabase API Key Invalid or Unauthorized. Please Try Again.")

def document_upload_section():
    pdfs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
    if st.button("Process"):
        with st.spinner("Processing"):
            # get pdf text
            text_batches = ingest(pdfs)

            # Initialize the vector db
            vector_db = init_vector_db(text_batches)

            # create conversation chain
            st.session_state.conversation = get_conversation_chain(vector_db)

# All the streamlit stuff
def main():
    # Sidebar Configuration
    global sb_api_key
    global sb_proj_url

    with st.sidebar:
        st.subheader("Model Selection", divider = True)
        model_select_section()
        
        # VectorDB Provider Selector Dropdown
        st.subheader("VectorDB Selection", divider = True)
        vector_db_select_section()
        
        # Document Upload Section
        st.subheader("Documents to Ask Questions To", divider = True)
        document_upload_section()

    # Main Page Configuration
    st.header("Welcome to Jays Trainable Chatbot")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.history = None
    
    # Not ready if no pdfs uploaded, no vector db, or no conversation chain
    if (True):
        st.write("Status: Not ready")
    else:
        st.write("Status: Ready")

    user_input = st.text_input("Talk to the bot", key="input_text")
    if user_input:
        generate_answer(user_input)

if __name__ == "__main__":
    main()