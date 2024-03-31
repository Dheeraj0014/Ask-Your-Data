import streamlit as st
import os
from langchain_experimental.agents import create_csv_agent
from dotenv import load_dotenv
from langchain_community.llms import OpenAI
from lida import Manager, TextGenerationConfig
from llmx import llm
import openai
from PIL import Image
from io import BytesIO
import base64

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

def base64_to_image(base64_string):
    # Decode the base64 string
    byte_data = base64.b64decode(base64_string)
    
    # Use BytesIO to convert the byte data to image
    return Image.open(BytesIO(byte_data))


lida = Manager(text_gen = llm("openai"))
textgen_config = TextGenerationConfig(n=1, temperature=0.5, model="gpt-3.5-turbo-0301", use_cache=True)

menu = st.sidebar.selectbox("Choose an Option", ["Ask Your Data", "Summarize", "Question based Graph"])

if menu == "Summarize":
    st.header("Summarization of your Data")
    file_uploader = st.file_uploader("Upload your CSV", type="csv")
    if file_uploader is not None:
        path_to_save = "filename.csv"
        with open(path_to_save, "wb") as f:
            f.write(file_uploader.getvalue())
        summary = lida.summarize("filename.csv", summary_method="default", textgen_config=textgen_config)
        st.write(summary)
        goals = lida.goals(summary, n=2, textgen_config=textgen_config)
        for goal in goals:
            st.write(goal)
        i = 0
        library = "seaborn"
        textgen_config = TextGenerationConfig(n=1, temperature=0.2, use_cache=True)
        charts = lida.visualize(summary=summary, goal=goals[i], textgen_config=textgen_config, library=library)  
        img_base64_string = charts[0].raster
        img = base64_to_image(img_base64_string)
        st.image(img)
        
       
elif menu == "Question based Graph":
    st.header("Query your Data to Generate Graph")
    file_uploader = st.file_uploader("Upload your CSV", type="csv")
    if file_uploader is not None:
        path_to_save = "filename1.csv"
        with open(path_to_save, "wb") as f:
            f.write(file_uploader.getvalue())
        text_area = st.text_area("Query your Data to Generate Graph", height=200)
        if st.button("Generate Graph"):
            if len(text_area) > 0:
                st.info("Your Query: " + text_area)
                lida = Manager(text_gen = llm("openai")) 
                textgen_config = TextGenerationConfig(n=1, temperature=0.2, use_cache=True)
                summary = lida.summarize("filename1.csv", summary_method="default", textgen_config=textgen_config)
                user_query = text_area
                charts = lida.visualize(summary=summary, goal=user_query, textgen_config=textgen_config)  
                charts[0]
                image_base64 = charts[0].raster
                img = base64_to_image(image_base64)
                st.image(img)

elif menu == "Ask Your Data":

    st.header("Ask Your Data")

    csv_file = st.file_uploader("Upload a CSV file", type="csv")

    user_question = None  # Initialize user_question here

    if csv_file is not None:
        user_question = st.text_input("Ask the question about your Data: ") 
        llm = OpenAI(temperature = 0) # (0 will be least creative and as we increase it upto 10 it will become more creative)
        agent = create_csv_agent(llm , csv_file , verbose = True) # verbose is print the output after thinking  
    
    if user_question is not None and user_question != "" :
        response = agent.run(user_question)
        st.write(response)
    else:
        st.write("Please upload a CSV file to proceed.")
               

