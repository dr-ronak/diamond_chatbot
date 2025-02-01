import streamlit as st
import cohere
import pandas as pd
import time  # Import time for precise timestamps
from datetime import datetime  # For formatting timestamps

# Initialize Cohere API
COHERE_API_KEY = "vPjqsnaockV4uYWjNcv56thveWsekn4D3jJVzBLE"  # Replace with your API key
co = cohere.Client(COHERE_API_KEY)

# Load the knowledge base
@st.cache_data
def load_knowledge_base(file_path):
    return pd.read_csv(file_path)

knowledge_base = load_knowledge_base("supply_chain_management.csv")  # Replace with your CSV file

# Function to query the knowledge base
def query_knowledge_base(query, knowledge_base, top_n=3):
    matches = knowledge_base[knowledge_base["content"].str.contains(query, case=False, na=False)]
    return matches.head(top_n).to_dict(orient="records")

# Function to get a response from Cohere's LLM
def get_cohere_response(prompt):
    response = co.generate(
        model='command-r-plus-08-2024',
        prompt=prompt,
        max_tokens=300,
        temperature=0.7,
    )
    return response.generations[0].text.strip()

# Initialize session state for messages and history
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'history' not in st.session_state:
    st.session_state.history = {}

if 'selected_history' not in st.session_state:
    st.session_state.selected_history = None

# "New Conversation" button
if st.sidebar.button("New Conversation"):
    st.session_state.messages = []
    st.session_state.selected_history = None

# First render the "Current Conversation" button at the top
if st.sidebar.button("Current Conversation"):
    st.session_state.selected_history = None

# History section: Only show distinct questions in the sidebar
st.sidebar.subheader("History")
displayed_questions = set()  # To track displayed questions and avoid duplicates

for key in sorted(st.session_state.history.keys(), reverse=True):  # Show most recent history first
    # Get the user's question from the history entry (first user message)
    user_question = st.session_state.history[key][0][1]  # Get the first message (User's question)

    if user_question not in displayed_questions:
        button_key = f"view_{key}"  # Generate a unique key for each button
        if st.sidebar.button(f"View: {user_question}", key=button_key):  # Use unique key for each button
            st.session_state.selected_history = key
        displayed_questions.add(user_question)  # Add question to the set to prevent duplicates


# Main interface
st.title("Supply Chain Management Chatbot")

# Function to handle user input and generate response
def handle_input():
    user_input = st.session_state.user_input
    if user_input:
        # Query the knowledge base
        kb_results = query_knowledge_base(user_input, knowledge_base)

        # Generate a Cohere response
        prompt = f"You are a supply chain management assistant. Answer the following query based on general knowledge: {user_input}"
        cohere_response = get_cohere_response(prompt)

        # Combine results
        if kb_results:
            kb_response = "\n\n".join([f"- {res['content']}" for res in kb_results])
            final_response = f"{cohere_response}\n\nAdditionally, here's what we found in our knowledge base:\n{kb_response}"
        else:
            final_response = cohere_response

        # Save messages in session state
        st.session_state.messages.append(("User", user_input))
        st.session_state.messages.append(("Bot", final_response))

        # Save conversation to history with precise timestamp (milliseconds)
        conversation_id = f"{time.time():.6f}"  # Using time.time() with 6 decimal places for milliseconds
        st.session_state.history[conversation_id] = st.session_state.messages[:]

        # Clear user input field
        st.session_state.user_input = ""

# Display selected history or ongoing conversation
if st.session_state.selected_history:
    # Display the question-answer pair of the selected history
    selected_messages = st.session_state.history[st.session_state.selected_history]
    for i in range(0, len(selected_messages), 2):
        st.write(f"**You:** {selected_messages[i][1]}")
        st.write(f"**Bot:** {selected_messages[i + 1][1]}")
else:
    # Display the most recent message first (current question and its answer)
    col1, col2 = st.columns([9, 1])
    with col1:
        st.text_input("You: ", key="user_input", label_visibility="collapsed", on_change=handle_input)
    with col2:
        st.button("⬆️", key="arrow_button", on_click=handle_input)

    # Display the most recent message first (current question and its answer)
    if len(st.session_state.messages) > 0:
        st.write(f"**You:** {st.session_state.messages[-2][1]}")
        st.write(f"**Bot:** {st.session_state.messages[-1][1]}")

    # Then, display the previous messages below
    for i in range(0, len(st.session_state.messages) - 2, 2):
        st.write(f"**You:** {st.session_state.messages[i][1]}")
        st.write(f"**Bot:** {st.session_state.messages[i + 1][1]}")
