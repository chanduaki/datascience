import streamlit as st
import requests


def get_openai_response(input_text):
    response = requests.post("http://localhost:8000/chat_open_ai_essay/invoke",
                             json={'input': {'topic': input_text}})

    return response.json()['output']['content']

def get_ollama_response(input_text):
    response = requests.post("http://localhost:8000/chat_llama2_essay/invoke",
                             json={'input': {'topic': input_text}})

    return response



st.title('Langchain Handson with LLAMA2 API')
input_1 = st.text_input("Write a Essay on (Using OpenAI")
input_2 = st.text_input("Write a Essay on (Using Ollama (Llama2)")


if input_1:
    st.write(get_openai_response(input_1))

if input_2:
    st.write(get_ollama_response(input_2))
