import streamlit as st
from langchain_openai.chat_models import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

def open_ai_df_agent(api_key, dataframes):
    try:
        llm = ChatOpenAI(
            model="gpt-5-nano",
            temperature=1,
            api_key=api_key
        )

        agent = create_pandas_dataframe_agent(
            llm,
            dataframes,
            agent_type="openai-functions",
            verbose=True,
            allow_dangerous_code=True
        )

        return agent
    except Exception as e:
        st.error(f"Error creating agent: {e}")
        st.stop()


def invoke_agent(agent, final_query):
    try:
        stream = agent.invoke(final_query)['output']
        return stream
    except Exception as e:
        st.error(f"Error generating answer: {e}")
        st.stop()
