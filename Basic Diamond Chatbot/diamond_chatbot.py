import streamlit as st
import pandas as pd
import cohere
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO


# Initialize Cohere API
cohere_api = "vPjqsnaockV4uYWjNcv56thveWsekn4D3jJVzBLE"
co = cohere.Client(cohere_api)

#Load dataset
def load_data(file_path):
    return pd.read_csv(file_path)


data = load_data("diamonds.csv")

# Function to analyze data and generate response
def generate_response(query):
    columns = ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'price', 'x', 'y', 'z']
    relevant_column = None

    #Match query to a relevant column
    for col in columns:
        if col.lower() in query.lower():
            relevant_column = col
            break

    
    if relevant_column:
        # Handle numeric columns
        if data[relevant_column].dtype in ['int64', 'float64']:
            fig, ax = plt.subplots(figsize=(10,6))
            sns.histplot(data[relevant_column], bins=30, kde=True, ax=ax, color='blue')
            ax.set_title(f"Distribution of {relevant_column}")
            ax.set_xlabel(relevant_column)
            ax.set_ylabel("Frequency")


            # Save the graph as a byte stream
            img_buffer = BytesIO()
            fig.savefig(img_buffer,format='png')
            img_buffer.seek(0)


            # Generate description using cohere
            prompt =f"""
            you are a diamond industry assitant. Based on the following data trend:
            - Column: {relevant_column}
            - Summary Statistics:
                - Mean: {data[relevant_column].mean():.2f}
                - Median: {data[relevant_column].median():.2f}
                - Standard Deviation: {data[relevant_column].std():.2f}

                write a short and meaningful insight about the column {relevant_column}.            
            """

            response = co.generate(
                model='command-r-plus-08-2024',
                prompt=prompt,
                max_tokens=100,
                temperature=0.7
            )

            description = response.generations[0].text.strip()

            return{
                "type" : "graph",
                "description" : description,
                "graph" : img_buffer
            }


        # Handle Categorical data
        elif data[relevant_column].dtype == 'object':
            fig, ax = plt.subplots(figsize =(10,6))
            sns.countplot(x=relevant_column, data=data, ax=ax, palette='viridis')
            ax.set_title(f"Distribution of {relevant_column}")
            ax.set_xlabel(relevant_column)
            ax.set_ylabel("Count")


            #save the graph as a byte stream
            img_buffer = BytesIO()
            fig.savefig(img_buffer, format='png')
            img_buffer.seek(0)


            # Generate description using cohere
            top_value = data[relevant_column].value_counts().idxmax()
            top_count = data[relevant_column].value_counts().max()

            prompt = f"""
            you are a diamond industry assistant. Based on the following data trend:
            - Column: {relevant_column}
            - Most Common Value: {top_value} ({top_count} occurrences)

            write a short and meaningful insight aboout the column {relevant_column}
            """

            response = co.generate(
                model='command-r-plus-08-2024',
                prompt=prompt,
                max_tokens=100,
                temperature=0.7
            )

            description = response.generations[0].text.strip()

            return{
                "type":"graph",
                "description":description,
                'graph': img_buffer
            }

        else:
            return{"type" : "text", "description":"No relevant data found"}


# Chatbot UI
st.title("Diamond Chatbot")

if 'messages' not in st.session_state:
    st.session_state.messages=[]


def handle_input():

    user_input = st.session_state.user_input

    if user_input:
        #Analyze the data and generate a response
        response = generate_response(user_input)

         # Save messages in session state
        st.session_state.messages.append(("You", user_input))

        # Append bot response, including graph if present
        if response['type'] == 'graph':
            st.session_state.messages.append(("Bot", response['description'], response['graph']))
        else:
            st.session_state.messages.append(("Bot", response['description'], None))

       

        
        # Clear the user input
        st.session_state.user_input=""


# Input and chat display
col1, col2 = st.columns([9,1])

with col1:
    st.text_input("You", key='user_input', label_visibility='collapsed', on_change=handle_input)
    
with col2:
    st.button("⬆️", key='send_button', on_click=handle_input)




# Display messages in reverse order (most recent first)
for i in range(len(st.session_state.messages) - 2, -1, -2):
    # User message
    st.write(f"**You:** {st.session_state.messages[i][1]}")

    # Bot response
    bot_response = st.session_state.messages[i + 1]
    st.write(f"**Bot:** {bot_response[1]}")

    # If there's a graph, display it below the bot's text response
    if len(bot_response) > 2 and bot_response[2]:
        st.image(bot_response[2])
