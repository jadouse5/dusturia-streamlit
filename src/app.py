import os
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
import streamlit as st

# Retrieve API keys from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

def init_database(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
    db_uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
    return SQLDatabase.from_uri(db_uri)

def get_sql_chain(db):
    template = """
    You are a legal analyst specializing in the Moroccan Constitutional Court's decisions. You are interacting with a user who is asking you questions about the court's database.
    Based on the table schema below, write a SQL query that would retrieve the relevant information from the court's decisions database. Take the conversation history into account.
    
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
    
    For example:
    Question: Retrieve all decisions made in the year 2022.
    SQL Query: SELECT * FROM decisions WHERE year = 2022;
    
    Question: What were the decisions related to electoral disputes?
    SQL Query: SELECT * FROM decisions WHERE specialty = 'Electoral Disputes';
    
    Question: List the summary and content of decision number 121/1963.
    SQL Query: SELECT summary, content FROM decisions WHERE decision_number = '121/1963';
    
    Your turn:
    
    Question: {question}
    SQL Query:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Choose between ChatGroq and ChatOpenAI based on availability of the API key
    if groq_api_key:
        llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0, groq_api_key=groq_api_key)
    else:
        llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=openai_api_key)
    
    def get_schema(_):
        return db.get_table_info()
    
    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )
    
def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    sql_chain = get_sql_chain(db)
    
    template = """
    You are a legal analyst specializing in the Moroccan Constitutional Court's decisions. You are interacting with a user who is asking you questions about the court's database.
    Based on the table schema below, question, sql query, and sql response, write a natural language response.
    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Use the selected LLM (either ChatGroq or ChatOpenAI)
    if groq_api_key:
        llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0, groq_api_key=groq_api_key)
    else:
        llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=openai_api_key)
    
    chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            schema=lambda _: db.get_table_info(),
            response=lambda vars: db.run(vars["query"]),
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain.invoke({
        "question": user_query,
        "chat_history": chat_history,
    })

# Initialize chat history if not already done
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I'm a SQL assistant. Ask me anything about your database."),
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
            except Exception as e:
                st.error(f"Failed to connect to the database: {e}")

# Ensure database connection is established before proceeding
if "db" not in st.session_state:
    st.warning("Please connect to the database first.")
else:
    user_query = st.chat_input("Type a message...")
    if user_query is not None and user_query.strip() != "":
        st.session_state.chat_history.append(HumanMessage(content=user_query))

        with st.chat_message("Human"):
            st.markdown(user_query)

        with st.chat_message("AI"):
            response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
            st.markdown(response)

        st.session_state.chat_history.append(AIMessage(content=response))
