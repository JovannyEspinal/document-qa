import streamlit as st

from agent import open_ai_df_agent, invoke_agent
from data import dataframe_from_uploaded_files
from config import SYSTEM_PROMPT

def show_title_and_description():
    st.title("📄 CSV FAQ Agent")
    st.write(
        "Upload a CSV file below and ask a question about it – GPT will answer! "
        "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
    )


show_title_and_description()

openai_api_key = st.text_input("OpenAI API Key", type="password")

if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:
    # Let the user upload a file via `st.file_uploader`.
    uploaded_files = st.file_uploader(
        "Upload a CSV document",
        type=("csv"),
        accept_multiple_files=True
    )

    dataframes = dataframe_from_uploaded_files(uploaded_files)

    # Ask the user for a question via `st.text_area`.
    question = st.text_area(
        "Now ask a question about the CSVs!",
        placeholder="Ask away!",
        disabled=not uploaded_files,
    )

    if uploaded_files and question:
        agent = open_ai_df_agent(openai_api_key, dataframes)

        final_query = SYSTEM_PROMPT + "\n\nQuestion: " + question

        stream = invoke_agent(agent, final_query)

        st.write(stream)
