from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import json
import datetime
import os
from vectordb import *
import re
import torch
import datetime
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_conditional_check_result(six_digit_string):
    df = pd.read_excel("app/orders_table.xlsx")
    try:
        obtained_record = df.loc[df["Order ID"]==int(six_digit_string),:].to_dict('records')
        timestamp_dt = obtained_record[0]['Order Date']
        today = datetime.datetime.now()
        difference = today - timestamp_dt

        is_date_exceeded = ""
        final_sale_message = ""

        if obtained_record[0]['Final sale'].lower() == 'no':
            if difference.days>30:
                is_date_exceeded = f"Its been {difference.days} since purchase of the product and thus it is strictly NOT eligible for return or exchange as per our policy. That should be your final response."
            else:
                is_date_exceeded = f"Its been {difference.days} is within the eligible limit for return or exchange as per our policy. Therefore it can be performed."
        else:
            final_sale_message = "This product is a Final Sale and therefore strictly not eligble for exchange, return or refunds. That should be your final response."

        main_content = f"""The customer {obtained_record[0]['Customer Name']} whose Order ID is {obtained_record[0]['Order ID']} has purchased {obtained_record[0]['Quantity']} {obtained_record[0]['Product Name']}
        of product category {obtained_record[0]['Product Category']} of size {obtained_record[0]['Size']}. The current status of the oder is {obtained_record[0]['Status']}."""

        total_content =  ' '.join([main_content, final_sale_message, is_date_exceeded])
        return total_content.replace("\n","")
        
    except:
        return ""

def phi_is_order_id_present(model_name, user_question, prompt, llm, output_parser):

    def find_six_digit_string(text):
        match = re.search(r'\d{5}', text)
        if match:
            return match.group()
        else:
            return None

    order_id_present = False

    # run the model
    if user_question:
        chain=prompt|llm|output_parser
        response = chain.invoke({"question":user_question})
        print("The obtained response from miniLM is : ", response)
        if (response.lower().find('none')==-1):
            order_id_present = True
            six_digit_string = find_six_digit_string(response)
            print("The identified Order ID is : ",six_digit_string)
    
    additonal_context = ""
    if order_id_present:
        additonal_context = get_conditional_check_result(six_digit_string)
    
    print(additonal_context)
    return additonal_context


def get_summary_of_previous_messages(session_state_chat_history, llm, chat_history_key):

    summary_prompt_instruction = """Generate a detailed summary of the following list of conversations between an e-commerce service bot 
    and a customer, specifically focusing on returns and exchanges. Ensure all key details are preserved, including any information 
    provided by the customer. The summary should be clear, concise, and maintain the integrity of the customer's inputs, requests, and 
    any resolutions offered by the service bot."""

    # Only last 4 conversations are considered for summary
    to_summarize = ' '.join([x["role"] +" : "+ x["content"] + "\n" for x in st.session_state[chat_history_key][:-4] if(x["role"]=="user")])

    summary_prompt =  ChatPromptTemplate.from_messages([
        ("system",summary_prompt_instruction),
        ("user","Summarize the following :{content}")
    ])

    generated_summary_response = to_summarize

    if to_summarize:
        summary_chain=summary_prompt|llm
        generated_summary_response = summary_chain.invoke({"content":to_summarize})

    print("Summary of the previous conversations have been generated for additional context : ", generated_summary_response)
    return generated_summary_response

def st_ollama(model_name, user_question, content_response, chat_history_key, prompt, llm, output_parser):

    if chat_history_key not in st.session_state.keys():
        st.session_state[chat_history_key] = []

    print_chat_history_timeline(chat_history_key)

    # run the model
    if user_question:
        st.session_state[chat_history_key].append({"content": f"{user_question}", "role": "user"})
        with st.chat_message("question", avatar="🧑‍🚀"):
            st.write(user_question)

        messages = [dict(content=message["content"], role=message["role"]) for message in st.session_state[chat_history_key]]


        if(len([1 for x in st.session_state[chat_history_key] if(x["role"]=="user")])>4):
            chat_history_content = '\n Here is your chat History for reference:' + get_summary_of_previous_messages(st.session_state[chat_history_key], llm, chat_history_key)
        else:
            chat_history_messages = ' '.join([x["role"] +" : "+ x["content"] + "\n" for x in st.session_state[chat_history_key] if(x["role"]=="user")])
            chat_history_content = '\n Here is your Chat History for reference:' + ' '.join(chat_history_messages)

        print("Session state content:", chat_history_messages)

        full_content = ' '.join([content_response, chat_history_content])

        def llm_stream(response):

            chain=prompt|llm|output_parser
            response = chain.invoke({"question":user_question,"content_response":full_content})
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

def reset_conversation(conversation_key):
  st.session_state.conversation = None
  st.session_state.chat_history = None
  st.session_state[conversation_key] = []
  conversation_key = ""

def check_safety(user_query, guard_llm):

    guard_prompt = ChatPromptTemplate.from_messages([
        ("user","{question}")
    ])

    if user_query:
        chain=guard_prompt|guard_llm
        guard_response = chain.invoke({"question":user_query})
        if("unsafe" in guard_response):
                exact_type = guard_response.split("\n")[1]
                print("Exact type : ",exact_type)
                return exact_type
        else:
            return False
        

# Ollama llama3.2:3b model
llm=Ollama(model='llama3.2:3b')
internal_llm=Ollama(model='phi3:mini-4k')
guardrail_llm = Ollama(model='llama-guard3:1b')

output_parser = StrOutputParser()

guardrail_map_file_path = "app/guardrail.json"

with open(guardrail_map_file_path, 'r') as j:
     gd_dict = json.loads(j.read())

if __name__ == "__main__":

    st.set_page_config(layout="wide", page_title="E-Commerce Support", page_icon="🦙")
    
    # Steamlit framework

    st.title('E-Commerce service desk')
    st.sidebar.title("Ollama Chat 🦙")
    llm_name = "llama3.2:3b"
    conversation_key = f"model_{llm_name}"

    internal_llm_name = "phi3:mini-4k"
    user_question = st.text_input("Ask me your queries related to your order.")

    guard_response = check_safety(user_question, guardrail_llm)

    guard_rail_instruction = ""
    if(guard_response):
        observed_topic = gd_dict[guard_response]
        print("The topic is mapped to : ", observed_topic)
        guard_rail_instruction = f"""User seems to be asking about {observed_topic}, which is a prohibited topic of discussion. Refuse to Discuss about {observed_topic} and discuss only about Returns and Exchanges."""

    # Checking if its a general question or an order id is provided.
    system_message_to_check_for_order_number = """Determine whether the user is asking a general question about returns and 
    exchanges or if they have provided a 5-digit order ID to initiate the return or exchange process. Check if the user has provided a 5-digit order ID. If an ID is detected, respond only with the ID. 
    If no ID is detected, respond with "none". Follow the above rules strictly and your response should be NO MORE THAN 10 characters. Example responses like "None" or "010101" etc are only allowed."""
    prompt_to_check_for_order_number = ChatPromptTemplate.from_messages([
        ("system",system_message_to_check_for_order_number),
        ("user","Question:{question}")
    ])
    
    additonal_conditional_check = phi_is_order_id_present(llm_name, user_question, prompt_to_check_for_order_number, llm, output_parser)
    
    # Uncomment the following if actively using pinecone db
    # content_response = fetch_best_context(user_question)

    # Use the following for testing if active use of pinecode db is not needed.
    content_response = additonal_conditional_check + """ Refund Policy • Refunds will be issued to the original payment method. • 
    Shipping costs are non-refundable unless the return is due to a manufacturing defect or an error on our part. 
    • If the original payment method is no longer available, store credit will be issued.6  Payment Policies 
    • We accept major credit/debit cards, PayPal, and other payment methods as listed at checkout. 
    • Orders are charged at the time of purchase. • In case of a refund, the amount will be credited back to the original 
    payment method within 7-10 business days."""
    
    # print(content_response)

    system_message = guard_rail_instruction + """You are a dedicated e-commerce chatbot assistant specializing in returns, exchanges, and policy guidance. 
    Respond courteously and efficiently to customer queries, ensuring all chat responses strictly adhere to the details provided in the context and include them in your chat response. 
    Utilize all customer information provided in the context. Find any relevant details like orderid, name, product name etc from the chat history provided as content. 
    Avoid discussing topics outside of returns, exchanges, and related policy information.
    DO NOT FABRICATE ANY NEW INFOMRATION OUTISDE THE PROVIDED CONTEXT"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system",system_message),
        ("user","Question:{question} <context>{content_response}</context>")
    ])

    
    st_ollama(llm_name, user_question,content_response, conversation_key, prompt, llm, output_parser)

    # Buttons to be visible after the conversation

    # save conversation to file
    save_conversation(llm_name, conversation_key)

