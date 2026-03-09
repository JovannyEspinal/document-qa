import pandas as pd
import streamlit as st
from langchain_openai.chat_models import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# Show title and description.
st.title("📄 CSV FAQ Agent")
st.write(
    "Upload a CSV file below and ask a question about it – GPT will answer! "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
)

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
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

    dataframes = []

    if uploaded_files:
        try:
            for file in uploaded_files:
                df = pd.read_csv(file)
                dataframes.append(df)
            print(f"\SUCCESS: Loaded '{file}' ({len(df)} rows)")
        except Exception as e:
            st.error(f"Error reading CSV files: {e}")
            st.stop()

    # Ask the user for a question via `st.text_area`.
    question = st.text_area(
        "Now ask a question about the CSVs!",
        placeholder="Ask away!",
        disabled=not uploaded_files,
    )

    system_prompt = """
    You are a smart data assistant capable of reading multiple CSV files.
    When asked a question, determine which DataFrame is most relevant.
    - Do NOT answer from general knowledge.
    - Answer in plain English.
    """

    if uploaded_files and question:

        try:
            llm = ChatOpenAI(
                model="gpt-5-nano",
                temperature=1,
                api_key=openai_api_key
            )

            agent = create_pandas_dataframe_agent(
                llm,
                dataframes,
                verbose=True,
                agent_type="openai-functions",
                allow_dangerous_code=True
            )
        except Exception as e:
            st.error(f"Error creating agent: {e}")
            st.stop()

        final_query = system_prompt + "\n\nQuestion: " + question


        # Generate an answer using the OpenAI API.
        try:
            stream = agent.invoke(final_query)['output']
        except Exception as e:
            st.error(f"Error generating answer: {e}")
            st.stop()

        # Stream the response to the app using `st.write_stream`.
        st.write(stream)
