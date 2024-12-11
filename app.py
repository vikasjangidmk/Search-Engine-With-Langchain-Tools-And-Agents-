import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchRun(name="Search")

# Initialize caching for queries
query_cache = {}

def cached_search(tool, query, retries=5, initial_wait=1):
    """
    Perform a cached search with retries and exponential backoff for rate limit handling.
    """
    if query in query_cache:
        return query_cache[query]

    wait_time = initial_wait
    for attempt in range(retries):
        try:
            result = tool.run(query)
            query_cache[query] = result
            return result
        except Exception as e:
            # Handle rate limit or any other exception
            if "ratelimit" in str(e).lower():
                st.warning(f"Rate limit hit. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                wait_time *= 2  # Exponential backoff
            else:
                raise e

    raise Exception("Rate limit exceeded. Please try again later.")

# Streamlit UI setup
st.title("🔍 LangChain - Chat with Search")
"""
This app lets you interact with LangChain's agents using tools like Arxiv, Wikipedia, and DuckDuckGo Search.
Explore more LangChain + Streamlit examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
"""

# Sidebar for settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Handle user input
if prompt := st.chat_input(placeholder="Ask me anything, e.g., 'What is machine learning?'"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    try:
        # Initialize LLM
        llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)

        # Define tools
        tools = [search, arxiv, wiki]

        # Initialize agent
        search_agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True
        )

        # Process user input
        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

            # Run the agent with rate limiting and caching
            response = search_agent.run(prompt, callbacks=[st_cb])

            # Validate the response to ensure meaningful content
            if not response or response.lower() == "none":
                response = "I couldn't find a proper answer to your query. Please try rephrasing it."

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)

    except Exception as e:
        st.session_state.messages.append({"role": "assistant", "content": f"An error occurred: {str(e)}"})
        st.chat_message("assistant").write(f"An error occurred: {str(e)}")
