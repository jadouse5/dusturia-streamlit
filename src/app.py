import os
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

def init_database(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
    db_uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
    return SQLDatabase.from_uri(db_uri)

def get_sql_chain(db, llm):
    template = """
    You are a legal analyst specializing in the Moroccan Constitutional Court's decisions. You are interacting with a user who is asking you questions about the court's database.
    Based on the table schema below, write a SQL query that would retrieve the year, specialty (type of decision), and summary from the court's decisions database. Limit the results to avoid retrieving too much data at once.
    
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
    
    For example:
    Question: Retrieve all decisions made in the year 2022.
    SQL Query: SELECT year, specialty, LEFT(summary, 200) as summary_excerpt FROM dusturia_records WHERE year = 2022 LIMIT 10;
    
    Question: What were the decisions related to electoral disputes?
    SQL Query: SELECT year, specialty, LEFT(summary, 200) as summary_excerpt FROM dusturia_records WHERE specialty = 'Electoral Disputes' LIMIT 10;
    
    Question: List the year, type, and summary of decision number 121/1963.
    SQL Query: SELECT year, specialty, LEFT(summary, 200) as summary_excerpt FROM dusturia_records WHERE decision_number = '121/1963' LIMIT 1;
    
    Your turn:
    
    Question: {question}
    SQL Query:
    """
    
    prompt = ChatPromptTemplate.from_template(template)

    def get_schema(_):
        return db.get_table_info()
    
    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )
    
def get_response(user_query: str, db: SQLDatabase, chat_history: list, llm):
    sql_chain = get_sql_chain(db, llm)
    if not sql_chain:
        return "Unable to process request due to missing or invalid API key."
    
    # Generate SQL query
    query = sql_chain.invoke({
        "question": user_query,
        "chat_history": chat_history,
    })
    st.write(f"Generated SQL Query: {query}")

    try:
        # Execute the query
        result = db.run(query)
        st.write(f"SQL Query Result: {result}")

        # Parse and return a user-friendly response
        if "COUNT" in query.upper():
            return f"There were {result[0][0]} decisions."
        else:
            # For other queries, display the result directly
            return result
    except Exception as e:
        st.error(f"SQL Execution Error: {e}")
        return None


# Initialize chat history if not already done
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I'm DusturIA. Ask me anything about the Moroccan Constitutional Court.."),
    ]

# Set Streamlit page config
st.set_page_config(page_title="Chat with MySQL", page_icon=":speech_balloon:")

st.title("Chat with MySQL")

# Sidebar for database connection settings
with st.sidebar:
    st.subheader("Settings")
    st.write("This is a simple chat application using MySQL. Connect to the database and start chatting.")
    
    host = st.text_input("Host", value="bxz3tw0pspj5uoxaxc0j-mysql.services.clever-cloud.com", key="Host")
    port = st.text_input("Port", value="3306", key="Port")
    user = st.text_input("User", value="upu2m1qun4syetdj", key="User")
    password = st.text_input("Password", type="password", value="8P9gZbUJaRy0RijKJ4bQ", key="Password")
    database = st.text_input("Database", value="bxz3tw0pspj5uoxaxc0j", key="Database")
    api_key = st.text_input("API Key", type="password", value="", key="APIKey")

    llm_provider = st.selectbox("LLM Provider", options=["Groq", "OpenAI"])

    # Connect to the database
    if st.button("Connect"):
        with st.spinner("Connecting to database..."):
            try:
                db = init_database(
                    st.session_state["User"],
                    st.session_state["Password"],
                    st.session_state["Host"],
                    st.session_state["Port"],
                    st.session_state["Database"]
                )
                st.session_state.db = db
                st.success("Connected to database!")

                # Initialize LLM based on provider selection
                if llm_provider == "Groq":
                    st.session_state.llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=2, groq_api_key=api_key)
                else:
                    st.session_state.llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=api_key)

                st.success(f"Connected to {llm_provider} model!")
            except Exception as e:
                st.error(f"Failed to connect to the database or LLM: {e}")

# Ensure database connection is established before proceeding
if "db" not in st.session_state or "llm" not in st.session_state:
    st.warning("Please connect to the database and LLM first.")
else:
    user_query = st.chat_input("Type a message...")
    if user_query is not None and user_query.strip() != "":
        st.session_state.chat_history.append(HumanMessage(content=user_query))

        with st.chat_message("Human"):
            st.markdown(user_query)

        with st.chat_message("AI"):
            response = get_response(user_query, st.session_state.db, st.session_state.chat_history, st.session_state.llm)
            if response:
                st.markdown(response)

        st.session_state.chat_history.append(AIMessage(content=response if response else "No response"))
