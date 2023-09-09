import os
from dotenv import load_dotenv
from langchain.vectorstores import Pinecone
from supabase import create_client, Client

class Vars:
    def __init__(self):
        load_dotenv()

        self.ready = True
        self.initialized_app_for_questions = False # For keeping track of whether someone has already hit process
        self.change_after_init = False # For keeping track of whether the settings have been changed after initialization

        # For keeping track of which settings have been changed
        self.change_default_llm = False
        self.change_default_vdb = False
        
        # Default Settings
        self.llm_brand = 0      # 0 == OpenAI
                                # 1 == HuggingFAce

        self.llm_model_oa = 0   # 0 == gpt-3.5-turbo (default)
                                # 1 == gpt-3.5-turbo-16k (4x context)
                                # 2 == "gpt-4 (expensive)
                                # 3 == gpt-4-32k (most expensive, 4x context)

        self.llm_model_hf = 0   # 0 == (default) google/flan-t5-xl (recommendeded)
                                # 1 ==  google/flan-t5-xxl (risks timeout)
                                # 2 == google/flan-t5-large
        
        self.vdb_chosen = 0     # 0 == Supabase
                                # 1 == Pinecone

        self.oa_api_key = os.getenv('OPENAI_API_KEY')
        self.sb_api_key = os.getenv('SUPABASE_API_KEY')
        self.sb_proj_url = os.getenv('SUPABASE_PROJ_URL')
        self.hf_api_key = None
        self.pc_api_key = None
        self.pc_env_key = None
        self.pc_index_name = "openai"
        self.supabase_client: Client = create_client(self.sb_proj_url, self.sb_api_key)

    def reset_settings():
        llm_brand = 0       # 0 == OpenAI
                            # 1 == HuggingFAce

        llm_model_oa = 0    # 0 == gpt-3.5-turbo (default)
                            # 1 == gpt-3.5-turbo-16k (4x context)
                            # 2 == "gpt-4 (expensive)
                            # 3 == gpt-4-32k (most expensive, 4x context)

        llm_model_hf = 0    # 0 == (default) google/flan-t5-xl (recommendeded)
                            # 1 ==  google/flan-t5-xxl (risks timeout)
                            # 2 == google/flan-t5-large

        vdb_chosen = 0      # 0 == Supabase
                            # 1 == Pinecone

        oa_api_key = os.getenv('OPENAI_API_KEY')
        sb_api_key = os.getenv('SUPABASE_API_KEY')
        sb_proj_url = os.getenv('SUPABASE_PROJ_URL')
        hf_api_key = None
        pc_api_key = None