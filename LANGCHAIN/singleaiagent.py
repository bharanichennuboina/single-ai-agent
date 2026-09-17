import os
import requests
import streamlit as st

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain.tools import tool
from langchain.agents import create_agent


# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")


# --------------------------------------------------
# 2. Streamlit page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI Agent",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Single AI Agent")
st.write("Ask me questions, search the web, or check the weather.")


# --------------------------------------------------
# 3. Check API keys
# --------------------------------------------------

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is not set.")

if not TAVILY_API_KEY:
    st.error("TAVILY_API_KEY is not set.")

if not WEATHERSTACK_API_KEY:
    st.error("WEATHERSTACK_API_KEY is not set.")


# Stop the application if keys are missing
if not GROQ_API_KEY or not TAVILY_API_KEY or not WEATHERSTACK_API_KEY:
    st.stop()


# --------------------------------------------------
# 4. Tavily Search Tool
# --------------------------------------------------

search_tool = TavilySearch(
    max_results=3,
    tavily_api_key=TAVILY_API_KEY
)


# --------------------------------------------------
# 5. Weather Tool
# --------------------------------------------------

@tool
def get_weather(city: str) -> str:
    """Fetch current weather information for a city."""

    api_key = WEATHERSTACK_API_KEY

    if not api_key:
        return "WEATHERSTACK_API_KEY is not set"

    url = "http://api.weatherstack.com/current"

    params = {
        "access_key": api_key,
        "query": city
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

    except requests.exceptions.RequestException as e:
        return f"Weather request failed: {e}"

    except ValueError:
        return "Weather API returned invalid JSON."

    if "current" not in data:
        return f"Could not fetch weather data for {city}: {data}"

    return (
        f"City: {city}\n"
        f"Temperature: {data['current']['temperature']}°C\n"
        f"Weather: {data['current']['weather_descriptions'][0]}\n"
        f"Humidity: {data['current']['humidity']}%"
    )


# --------------------------------------------------
# 6. Create Groq LLM
# --------------------------------------------------

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=GROQ_API_KEY
)


# --------------------------------------------------
# 7. Add tools
# --------------------------------------------------

tools = [
    search_tool,
    get_weather
]


# --------------------------------------------------
# 8. Create Agent
# --------------------------------------------------

agent = create_agent(
    model=llm,
    tools=tools
)


# --------------------------------------------------
# 9. Store chat history
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------
# 10. Display previous messages
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --------------------------------------------------
# 11. Chat input
# --------------------------------------------------

user_input = st.chat_input(
    "Ask something..."
)


# --------------------------------------------------
# 12. Run the agent
# --------------------------------------------------

if user_input:

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # Generate AI response
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                response = agent.invoke({
                    "messages": [
                        {
                            "role": "user",
                            "content": user_input
                        }
                    ]
                })

                # Get the final AI message
                final_message = response["messages"][-1]

                answer = final_message.content

                st.markdown(answer)

                # Save AI response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as e:

                st.error(
                    f"An error occurred: {e}"
                )