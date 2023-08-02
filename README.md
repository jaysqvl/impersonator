# impersonator
Use the LLM and VectorDB of your choice to query any set of raw text e.g. PDFs, .txt, ebooks, etc.
Answers any knowledge based questions by using retrieval from the embeddings stored in a VectorDB and returns them using the LLM as an interface to deliver the information.

# Instructions
1. Install all dependencies\
pip install -r requirements.txt

2. Create a .env file with your OpenAI api key using the following line\
OPENAI_API_KEY=<INSERT YOUR API KEY HERE>

3. Run the app!\
streamlit run script.py

# To-Do
- Validate the uploaded files
- Pre-train using the text
- Add a chat feature with history
