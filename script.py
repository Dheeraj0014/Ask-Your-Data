import streamlit as st
from langchain_experimental.agents import create_csv_agent
#from langchain.llms import OpenAI
from dotenv import load_dotenv
from langchain_community.llms import OpenAI
# from langchain_experimental.agents.agent_toolkits.csv.base import create_csv_agent
# from langchain.schema.language_model import BaseLanguageModel


def main():

    load_dotenv()

    st.set_page_config(page_title="Ask Your Data ️")  

    st.header("Ask Your Data")

    csv_file = st.file_uploader("Upload a CSV file", type="csv")

    user_question = None  # Initialize user_question here

    if csv_file is not None:
        user_question = st.text_input("Ask the question about your Data: ")
             
        llm = OpenAI(temperature = 0) # (0 will be non creative and as we increase it upto 10 it will become more creative)
        agent = create_csv_agent(llm , csv_file , verbose = True) # verbose is print the output after thinking  
    
    if user_question is not None and user_question != "" :
        response = agent.run(user_question)
        st.write(response)
    else:
        st.write("Please upload a CSV file to proceed.")

    
if __name__ == "__main__":
    main()