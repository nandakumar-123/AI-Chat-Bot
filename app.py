import streamlit as st
from groq import Groq

# Load API key (make sure no trailing spaces in key name)
client = Groq(api_key=st.secrets.get("GROQ_API_KEY"))

# Initialize session state
if "llm" not in st.session_state:
    st.session_state["llm"] = None

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Function to reset the chat when model changes
def reset_chat():
    st.session_state["messages"] = []
    st.toast(f"Model selected: {st.session_state.llm}")

# UI layout
st.header("Chatbot Playground", divider="orange", anchor=False)
st.title(":orange[Chat App]", anchor=False)
st.subheader("Powered by Groq")

st.sidebar.title("Parameters")

# Model selection (using currently supported models)
st.session_state.llm = st.sidebar.selectbox(
    "Select Model",
    ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "openai/gpt-oss-20b", "openai/gpt-oss-120b"],
    index=0,
    on_change=reset_chat
)

# Sidebar controls
temperature = st.sidebar.slider("Temperature", 0.0, 2.0, value=1.0)
max_tokens = st.sidebar.slider("Max Tokens", 0, 131072, value=1024)
streaming = st.sidebar.toggle("Stream Mode", value=True)
json_mode = st.sidebar.toggle("JSON Mode", help="Enable JSON structured response output.")

with st.sidebar.expander("Advanced"):
    top_p = st.slider("Top P", 0.0, 1.0, value=1.0,
                      help="Less commonly changed if you adjust temperature.")
    stop_sequence = st.text_input("Stop Sequence")

# Display chat history
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Type your message..."):
    # Add user message
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        response_place = st.empty()
        full_response = ""

        # Make the API call
        completion = client.chat.completions.create(
            model=st.session_state.llm or "llama-3.1-8b-instant",
            messages=st.session_state["messages"],
            stream=streaming,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json"} if json_mode else {"type": "text"},
            top_p=top_p if top_p else None,
            stop=stop_sequence if stop_sequence else None
        )

        if streaming:
            for chunk in completion:
                # Assuming chunk.choices[0].delta.content exists for streaming
                delta = chunk.choices[0].delta.content or ""
                full_response += delta
                response_place.write(full_response)
        else:
            full_response = completion.choices[0].message.content
            response_place.write(full_response)

        # Add assistant message to history
        st.session_state["messages"].append({"role": "assistant", "content": full_response})
