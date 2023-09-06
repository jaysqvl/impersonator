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
from langchain.vectorstores import Pinecone
from supabase import create_client, Client
from langchain.llms import HuggingFaceHub
import pinecone
import requests
import pyautogui

# Loads the .env file
load_dotenv()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Global Dynamic APP Settings
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# If all of the keys are filled in, then the app is ready to go
ready=True
initialized_app_for_questions = False # For keeping track of whether someone has already hit process
change_after_init = False # For keeping track of whether the settings have been changed after initialization

# For keeping track of which settings have been changed
change_default_llm = False
change_default_vdb = False

# Default Settings
llm = 0 # 0 == OpenAI
        # 1 == HuggingFAce
llm_model_oa = 0 # 0 == gpt-3.5-turbo (default)
                 # 1 == gpt-3.5-turbo-16k (4x context) 
                 # 2 == "gpt-4 (expensive)
                 # 3 == gpt-4-32k (most expensive, 4x context)
llm_model_hf = 0 # 0 == (default) google/flan-t5-xxl (recommended, cheapest)
                 # 1 ==  google/flan-t5-xxl (4x context)
                 # 2 == google/flan-t5-xxl (most expensive, 4x context)
vector_db = 0 # 0 == Supabase
              # 1 == Pinecone

oa_api_key = os.getenv('OPENAI_API_KEY')
sb_api_key = os.getenv('SUPABASE_API_KEY')
sb_proj_url = os.getenv('SUPABASE_PROJ_URL')
hf_api_key = None
pc_api_key = None
pc_env_key = None
supabase_client: Client = create_client(sb_proj_url, sb_api_key)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Global Reset Functions
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def reset_app_hard():
    # Refreshes the page on both windows and mac
    pyautogui.hotkey("ctrl", "F5")
    pyautogui.hotkey("command", "R")

def reset_settings():
    # Resets all the global APP settings
    global llm; llm_model_oa; llm_model_hf, vector_db; oa_api_key; sb_api_key; sb_proj_url; hf_api_key; pc_api_key

    # Default Settings
    llm = 0 # 0 == OpenAI
            # 1 == HuggingFAce
    llm_model_oa = 0 # 0 == gpt-3.5-turbo (default)
                    # 1 == gpt-3.5-turbo-16k (4x context) 
                    # 2 == "gpt-4 (expensive)
                    # 3 == gpt-4-32k (most expensive, 4x context)
    llm_model_hf = 0 # 0 == (default) google/flan-t5-xxl (recommended, cheapest)
                    # 1 ==  google/flan-t5-xxl (4x context)
                    # 2 == google/flan-t5-xxl (most expensive, 4x context)
    vector_db = 0 # 0 == Supabase
                  # 1 == Pinecone
    temp_key = ""

    oa_api_key = os.getenv('OPENAI_API_KEY')
    sb_api_key = os.getenv('SUPABASE_API_KEY')
    sb_proj_url = os.getenv('SUPABASE_PROJ_URL')
    hf_api_key = None
    pc_api_key = None

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Settings Selection Section
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def change_openai_api_key(api_key):
    global oa_api_key
    oa_api_key = api_key

def change_huggingface_api_key(api_key):
    global hf_api_key
    hf_api_key = api_key

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

def hf_inputs():
    global hf_api_key

    activated = st.toggle("Change Default HF Model", value=False)

    display = ("(default) google/flan-t5-xxl (recommended, cheapest)", 
                "google/flan-t5-xxl (4x context)", 
                "google/flan-t5-xxl (most expensive, 4x context)")
    options = list(range(len(display))) # Enumerates the display list
    llm_model_chosen = st.selectbox("Select LLM Model",
                                            options,
                                            format_func=lambda x: display[x], 
                                            disabled=(not activated), key="hf_model_selectbox")
    if activated:
        llm_api_key = st.text_input("Enter HuggingFace API Key")
        if llm_api_key != "":
            hf_api_key = llm_api_key
        

def oa_inputs():
    activated = st.toggle("Change GPT Version (Requires own api key)", value=False)

    if activated:
        # Warning message
        st.write("WARNING: Changing default GPT version requires your own OpenAI API Key which will bill you for use of both the Ada Embedding model and selected LLM model")
        st.write("Toggle off sets the model back to the default LLM model automatically and uses the hosts API key (e.g. gpt-3.5-turbo)")

    # Model Selector Dropdown
    display = ("(default) gpt-3.5-turbo (recommended, cheapest)", 
                "gpt-3.5-turbo-16k (4x context)", 
                "gpt-4 (expensive)", 
                "gpt-4-32k (most expensive, 4x context))")
    options = list(range(len(display))) # Enumerates the display list
    llm_model_chosen = st.selectbox("Select LLM Model", 
                                            options,
                                            format_func=lambda x: display[x],
                                            disabled=(not activated))

    # API Key Input Text Box
    if (activated & (llm_model_chosen != 0)):
        ready = False
        llm_api_key = st.text_input("Enter Your Own OpenAI API Key (and press enter)")

        if llm_api_key:
            if (check_openai_api_key(llm_api_key)):
                change_openai_api_key(llm_api_key)
                st.write("OpenAI API Key Successfully Updated!")
            else:
                llm_api_key = None
                st.write("OpenAI API Key Invalid or Unauthorized. Please Try Again Or Use The Default Model.")

def model_select_section():
    global oa_api_key, hf_api_key, ready

    # LLM Provider Selector Dropdown
    display = ("(default) OpenAI", "HuggingFace") 
    options = list(range(len(display))) # Enumerates the display list
    llm_brand_selectbox = st.selectbox("Select LLM Provider", options, format_func=lambda x: display[x])

    # What shows up if you select OpenAI as your LLM provider
    if llm_brand_selectbox == 0: # Represents OpenAI
        # Toggle for changing the default GPT version
        oa_inputs()

    # What shows up if you select HuggingFace as your LLM provider
    elif llm_brand_selectbox == 1: # Represents HuggingFace
        # Toggle for changing the default HuggingFace LLM model
        hf_inputs()

def initialize_supabase(api_key, proj_url):
    global sb_api_key, sb_proj_url, supabase_client
    sb_api_key = api_key
    sb_proj_url = proj_url

    try: 
        supabase_client = create_client(sb_proj_url, sb_api_key)
    except:
        st.write("Your Supabase API Key or Project URL is invalid. Keys removed. Please try again")
        sb_api_key = os.environ("SUPABASE_API_KEY")
        sb_proj_url = os.environ("SUPABASE_PROJ_URL")
    else:
        st.write("Supabase API Key and Project URL Successfully Updated!")

def initialize_pinecone(api_key, env_key):
    global pc_api_key, pc_env_key
    pc_api_key = api_key
    pc_env_key = env_key

    try:
        pinecone.init(api_key=pc_api_key, environment=pc_env_key)
    except:
        st.write("Your Pinecone API Key or Environment Key is invalid. Keys removed. Please try again")
        pc_api_key = None
        pc_env_key = None
    else:
        st.write("Pinecone API Key and Environment Key Successfully Updated!")

def vector_db_select_section():
    global sb_api_key
    global sb_proj_url

    # VectorDB Provider Selector Dropdown
    vdb_activated = st.toggle("Use Your Own VectorDB Provider", value=False)

    display = ("(default) Supabase", 
               "Pinecone")  
    options = list(range(len(display))) # Enumerates the display list
    vdb_chosen = st.selectbox("Select VectorDB Provider", 
                                            options,
                                            format_func=lambda x: display[x],
                                            disabled=(not vdb_activated))
    
    if vdb_activated:
        st.write("WARNING: Does not have API key or project URL verification. Requires a compatible VectorDB Provider, API Key and properly set-up/pre-configured project's URL, otherwise the app will not work. Visit Supabase's blog post for more information:")

        st.write("https://supabase.com/blog/openai-embeddings-postgres-vector")

        st.write("In case of error, please reset the app using the above button or contact the developer.")
        if vdb_chosen == 0:
            sb_vdb_api_key = st.text_input("Enter Supabase API Key (and press enter)")
            sb_proj_url = st.text_input("Enter Supabase Project URL (and press enter)")

            if sb_vdb_api_key and sb_proj_url:
                initialize_supabase(sb_vdb_api_key, sb_proj_url)
                
        elif vdb_chosen == 1:
            pc_vdb_api_key = st.text_input("Enter Pinecone API Key (and press enter))")
            pc_env_key = st.text_input("Enter Pinecone Environment Key (and press enter))")
            if pc_vdb_api_key and pc_env_key:
                initialize_pinecone(pc_vdb_api_key, pc_env_key)
        

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# VectorDB Document Upload and Integration into LLM Pipeline
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Step 3
def get_text_batches(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size = 1000,
        chunk_overlap = 100
    )
    return text_splitter.split_text(text)

# Step 2
def get_text(pdfs):
    text = ""
    for pdf in pdfs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Step 1
def ingest(pdfs):
    text = get_text(pdfs)
    text_batches = get_text_batches(text)

    return text_batches

# Step 4
# Creates Vector DB
def init_vector_db(text_batches):
    print(text_batches)
    embeddings = OpenAIEmbeddings(openai_api_key=oa_api_key)
    if vector_db == 0:
        vector_db_client = SupabaseVectorStore.from_texts(text_batches, embeddings, client=supabase_client, table_name="documents")
    elif vector_db == 1:
        if "openai" not in pinecone.list_indexes():
            pinecone.create_index("openai", dimension=1536, metric="cosine")
        index = pinecone.Index("openai")
        
        vector_db_client = Pinecone.from_texts(text_batches, embeddings, index="openai")
        
    return vector_db_client

# Step 5
def get_conversation_chain(vector_db):
    global llm, llm_model_oa, llm_model_hf

    if llm == 0:
        llm = ChatOpenAI(openai_api_key=oa_api_key, model=llm_model_oa)
    elif llm == 1:
        llm = HuggingFaceHub(repo_id=llm_model_hf, model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_db.as_retriever(search_type="similarity"),
        memory=memory)

    return conversation_chain

def document_upload_section():
    global initialized_app_for_questions
    pdfs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
    if st.button("Process", disabled=(not ready)):
        with st.spinner("Processing"):
            initialized_app_for_questions = True

            # get pdf text
            text_batches = ingest(pdfs)

            # Initialize the vector db
            vector_db = init_vector_db(text_batches)

            # create conversation chain
            st.session_state.conversation = get_conversation_chain(vector_db)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Runner
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

def main():
    # Sidebar Configuration
    with st.sidebar:
        # Document Upload Section
        st.subheader("Documents to Ask Questions To", divider = True)
        st.write("This app works without changing any settings, but you can change the settings to your liking.")
        st.write("If you decide to do so and you already used it, to avoid bugs/errors, reload/reset the app, make your changes, and then 'Process' again.")
        document_upload_section()

        st.subheader("(Optional) Model Selection", divider = True)
        model_select_section()
        
        # VectorDB Provider Selector Dropdown
        st.subheader("(Optional) VectorDB Selection", divider = True)
        vector_db_select_section()

        # Reset Button
        st.subheader("Reset", divider = True)
        st.write("WARNING: Resets all settings and clears all uploaded documents")
        if st.button("Hard Reset App"):
            reset_app_hard()

    # Main Page Configuration
    st.header("Welcome to Jays Trainable Chatbot")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.history = None
    
    # Not ready if no pdfs uploaded, no vector db, or no conversation chain
    if (ready):
        st.write("Status: Ready to go!")
    else:
        st.write("Status: Not Ready. Please upload PDFs and select a VectorDB Provider.")

    user_input = st.text_input("Talk to the bot", key="input_text")
    if user_input:
        generate_answer(user_input)

if __name__ == "__main__":
    main()