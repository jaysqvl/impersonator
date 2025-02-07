# Impersonator 🤖

A powerful document-based chatbot powered by LangChain and OpenAI that lets you have intelligent conversations with your documents. Upload PDFs and interact with their content through natural language queries.

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![License](https://img.shields.io/badge/license-GPL--3.0-green.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.0.312-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25.0-red.svg)

## ✨ Features

- **Document Intelligence**: Upload PDFs and have natural conversations about their content
- **Smart Text Processing**: Advanced chunking algorithms for optimal context preservation
- **Vector Database Integration**: Efficient document embedding storage with Supabase
- **Flexible Architecture**: Support for multiple LLMs (OpenAI, HuggingFace) and Vector DBs (Supabase, Pinecone)
- **Containerized Deployment**: Ready for production with Docker support
- **Interactive UI**: Built with Streamlit for a seamless user experience

## 🚀 Quick Start

### Using Docker (Recommended)

1. Clone the repository

2. Open a terminal in the impersonator folder

3. Create a `.env` file in the `src/` directory with the following format:
~~~
OPENAI_API_KEY=your_openai_key
SUPABASE_API_KEY=your_supabase_key
SUPABASE_PROJ_URL=your_project_url
PINECONE_API_KEY=your_pinecone_key
~~~

4. Build the docker image
~~~
docker build -t impersonator .
~~~

5. Run with docker binding the docker containers ports to the host machines port!
~~~ 
docker run -p 8501:8501 impersonator
~~~

6. Visit `http://localhost:8501` in your browser

### Manual Setup

1. Navigate to source directory:
~~~
cd src/
~~~

2. Install all dependencies
~~~
pip install -r requirements.txt
~~~

3. Run the app!
~~~
streamlit run script.py
~~~

7. Go to the local host link provided in the CLI!

## To-Do
- Validate the uploaded files
- Pre-train using the text
- Add a chat feature with history

## 🛠️ Technical Stack

- **Frontend**: Streamlit with streamlit-chat for UI components
- **Backend**: Python with LangChain for LLM operations
- **Document Processing**: PyPDF for PDF parsing
- **Vector Embeddings**: OpenAI's embedding model (with HuggingFace support)
- **Vector Storage**: Supabase (with Pinecone integration ready)
- **Containerization**: Docker for consistent deployment
- **Environment Management**: python-dotenv for configuration

## 💡 How It Works

1. **Document Upload**: Users upload PDF documents through the Streamlit interface
2. **Text Processing**: Documents are parsed and split into manageable chunks
3. **Vector Embedding**: Text chunks are converted to vector embeddings
4. **Storage**: Embeddings are stored in Supabase's vector database
5. **Query Processing**: User questions are processed using conversational AI
6. **Response Generation**: LLM generates contextual responses based on retrieved information

## 🔜 Roadmap

- [ ] File validation and error handling
- [ ] Pre-training capabilities
- [ ] Enhanced chat history with conversation memory
- [ ] Support for additional document formats
- [ ] Alternative LLM integrations
- [ ] Advanced vector database options

## 📄 License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

---

Built with ❤️ by [jaysqvl](https://github.com/jaysqvl)