# impersonator
Use the LLM and VectorDB of your choice to query any set of raw text e.g. PDFs, .txt, ebooks, etc.
Answers any knowledge based questions by using retrieval from the embeddings stored in a VectorDB and returns them using the LLM as an interface to deliver the information.

## Instructions
1. Clone the repository to the directory of your choice
~~~
git clone https://github.com/jaysqvl/impersonator.git
~~~

2. Open a terminal in the impersonator folder

3. Create a .env file with the following format in the "src/" folder (or rename .env1 to .env and use that)
~~~
OPENAI_API_KEY=YOUR_API_KEY
SUPABASE_API_KEY=YOUR_API_KEY
SUPABASE_PROJ_URL=YOUR_PROJECT_URL
PINECONE_API_KEY=YOUR_API_KEY
~~~

### With Docker
4. Build the docker image
~~~
docker build -t impersonator .
~~~

5. Run with docker binding the docker containers ports to the host machines port!
~~~ 
docker run -p 8501:8501 impersonator
~~~

6. Go to the ip bound in the dockerfile
~~~
http://localhost:8501
~~~

### Without Docker
4. Change directories into the src folder
~~~
cd src/
~~~

5. Install all dependencies
~~~
pip install -r requirements.txt
~~~

6. Run the app!
~~~
streamlit run script.py
~~~

7. Go to the local host link provided in the CLI!

## To-Do
- Validate the uploaded files
- Pre-train using the text
- Add a chat feature with history