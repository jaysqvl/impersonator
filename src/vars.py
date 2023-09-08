import os
from dotenv import load_dotenv
from langchain.vectorstores import Pinecone
from supabase import create_client, Client


# Loads the .env file
load_dotenv()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Global Dynamic APP Settings/Initialization
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# If all of the keys are filled in, then the app is ready to go
ready=True
initialized_app_for_questions = False # For keeping track of whether someone has already hit process
change_after_init = False # For keeping track of whether the settings have been changed after initialization

# For keeping track of which settings have been changed
change_default_llm = False
change_default_vdb = False

# Default Settings
llm_brand = 0 # 0 == OpenAI
        # 1 == HuggingFAce
llm_model_oa = 0 # 0 == gpt-3.5-turbo (default)
                 # 1 == gpt-3.5-turbo-16k (4x context) 
                 # 2 == "gpt-4 (expensive)
                 # 3 == gpt-4-32k (most expensive, 4x context)
llm_model_hf = 0 # 0 == (default) google/flan-t5-xxl (recommended, cheapest)
                 # 1 ==  google/flan-t5-xxl (4x context)
                 # 2 == google/flan-t5-xxl (most expensive, 4x context)
vdb_chosen = 0 # 0 == Supabase
              # 1 == Pinecone

oa_api_key = os.getenv('OPENAI_API_KEY')
sb_api_key = os.getenv('SUPABASE_API_KEY')
sb_proj_url = os.getenv('SUPABASE_PROJ_URL')
hf_api_key = None
pc_api_key = None
pc_env_key = None
pc_index_name = "openai"
supabase_client: Client = create_client(sb_proj_url, sb_api_key)

def reset_settings():
    # Resets all the global APP settings
    global llm_brand, llm_model_oa, llm_model_hf, vector_db, oa_api_key, sb_api_key, sb_proj_url, hf_api_key, pc_api_key

    # Default Settings
    llm_brand = 0 # 0 == OpenAI
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