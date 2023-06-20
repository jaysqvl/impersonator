import torch
import streamlit as st
import os

from dotenv import load_dotenv
from transformers import OpenAiAgent
from transformers import load_tool
from transformers import AutoTokenizer, AutoModel
from streamlit_chat import message

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

# INPUT: An array of files uploaded
def processinput(files):
    sentence_embeddings = []

    for f in files:
        # Read the file content and decode it as text
        text = f.getvalue().decode('utf-8')
        # Let's assume each line in the file is a sentence
        sentences = text.split('\n')
        # Compute embeddings for all sentences in the file
        file_embeddings = [get_sentence_embedding(sentence) for sentence in sentences]
        # Add the embeddings from this file to the overall list
        sentence_embeddings.extend(file_embeddings)

    return sentence_embeddings

def encode_text(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state

def get_sentence_embedding(text):
    output = encode_text(text)
    sentence_embedding = output.mean(dim=1)  # We take the mean to get a sentence embedding
    return sentence_embedding

def main():
    # SETUP
    print(torch.cuda.is_available())
    load_dotenv()
    oa_api_key = os.getenv('OPENAI_API_KEY')
    agent = OpenAiAgent(model="gpt-3.5-turbo", api_key=oa_api_key)
    st.title("Welcome to Jays HuggingFace AI Playground")
    
    # This is the trained data
    vectors = processinput(st.file_uploader("Raw text files only", accept_multiple_files=True))

    # TRANSFORMERS CHATBOT
    message("Welcome! I am an AI chatbot ready to be trained")
    message("wassup", is_user=True)

    # Transformers
    boat = agent.run("Generate an image of a baby hippo")
    st.image(boat)

    caption = agent.run("Caption the following 'image' trendily like a millenial would for a post on instagram", image = boat, max_new_tokens = 20)
    st.write(caption)

    st.write("finished Execution")

if __name__ == "__main__":
    main()