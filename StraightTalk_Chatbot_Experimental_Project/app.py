import streamlit as st
import os
import json
import datetime
import os
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

from agents import *

llm=Ollama(model='llama3.2:3b')

def st_ollama(model_name, user_question, chat_history_key):

    if chat_history_key not in st.session_state.keys():
        st.session_state[chat_history_key] = []

    print_chat_history_timeline(chat_history_key)

    # run the model
    if user_question:
        st.session_state[chat_history_key].append({"content": f"{user_question}", "role": "user"})
        with st.chat_message("question", avatar="🧑‍🚀"):
            st.write(user_question)

        messages = [dict(content=message["content"], role=message["role"]) for message in st.session_state[chat_history_key]]
        
        # Currently a placeholder chat history can be nadded here if needed
        chat_history_content = ''
        
        def llm_stream(response):
            response = planner_agent(user_question)

            yield response

        # streaming response
        with st.chat_message("response", avatar="🤖"):
            chat_box = st.empty()
            response_message = chat_box.write_stream(llm_stream(messages))

        st.session_state[chat_history_key].append({"content": f"{response_message}", "role": "assistant"})
        return response_message

def print_chat_history_timeline(chat_history_key):
    for message in st.session_state[chat_history_key]:
        role = message["role"]
        if role == "user":
            with st.chat_message("user", avatar="🧑‍🚀"): 
                question = message["content"]
                st.markdown(f"{question}", unsafe_allow_html=True)
        elif role == "assistant":
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(message["content"], unsafe_allow_html=True)

def save_conversation(llm_name, conversation_key):

    OUTPUT_DIR = "llm_conversations"
    OUTPUT_DIR = os.path.join(os.getcwd(), OUTPUT_DIR)

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{OUTPUT_DIR}/{timestamp}_{llm_name.replace(':', '-')}"

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    if st.session_state[conversation_key]:

        if st.sidebar.button("Save conversation"):
            with open(f"{filename}.json", "w") as f:
                json.dump(st.session_state[conversation_key], f, indent=4)
            st.success(f"Conversation saved to {filename}.json")

if __name__ == "__main__":

    st.set_page_config(layout="wide", page_title="StraightTalk Customer Support", page_icon="🦙")
    
    # Steamlit framework

    st.title('StraightTalk Customer Support')
    st.sidebar.title("Ollama Chat 🦙")
    llm_name = "llama3.2:3b"
    conversation_key = f"model_{llm_name}"

    user_question = st.text_input("How can I help you today?")

    st_ollama(llm_name, user_question, conversation_key)

    # Buttons to be visible after the conversation

    # save conversation to file
    save_conversation(llm_name, conversation_key)