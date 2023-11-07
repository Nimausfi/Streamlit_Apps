from langchain.llms import OpenAI
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks import StreamlitCallbackHandler
import streamlit as st

def get_api_key():
    input_text = st.text_input(label="OpenAI API Key",  placeholder="Enter Your Key: ", key="openai_api_key_input")
    return input_text

openai_api_key = get_api_key()

if openai_api_key:
    llm = OpenAI(temperature=0, streaming=True, openai_api_key=openai_api_key)
    tools = load_tools(["ddg-search"])
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        # verbose=True
    )
