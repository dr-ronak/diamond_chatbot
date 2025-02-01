import streamlit as st
import pandas as pd
import cohere
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# Initialize Cohere API
cohere_api = "vPjqsnaockV4uYWjNcv56thveWsekn4D3jJVzBLE"
co = cohere.Client(cohere_api)

# Load dataset
def load_data(file_path):
    return pd.read_csv(file_path,encoding='latin1')

data = load_data("SampleData_For_AI.csv")

# Function to analyze data and generate response
def generate_response(query):
    columns = ['SEQNO', 'SUB_SEQNO', 'FORM_NAME', 'COID', 'BRID', 'LOTNO', 'STONE_ID',
       'Ast Pktno', 'Gem Pktno', 'ISS_ID', 'JAN_ISS_ID', 'REPORT NO',
       'TO TRANS TYPE', 'TO PARENT ENTITY TYPE', 'TO PARENT ENTITY',
       'TO ENTITY TYPE', 'TO ENTITY', 'TO NODE', 'ITEM', 'TRANS. DATE', 'PCS',
       'CTS', 'RET CTS', 'LOSS CTS', 'FROM TRANS TYPE',
       'FROM PARENT ENTITY TYPE', 'FROM PARENT ENTITY ', 'FROM ENTITY TYPE',
       'FROM ENTITY', 'FROM NODE', 'J NO', 'J DATE', 'J TIME', 'ACK', 'ACK BY',
       'ACK DATE', 'PROC REMARK', 'SUB PROC REMARK', 'T_TYPE', 'TRANS REMARK',
       'ENTRY DATE', 'ENTRY TERM', 'ENTRY USER', 'Sales No', 'Consignment No',
       'REF_DEMD_ID', 'O_REF_DEMD_ID', 'PAIR_REF_STONE_ID', 'O_REF_PAIR_ID',
       'ACT_AMT', 'SHP', 'CLR', 'CLA', 'CUT', 'POL', 'SYMM', 'FLOUR', 'BRN',
       'TENSION', 'DIAMOND_TYPE', 'TAG,']
    relevant_column = None

    # Match query to a relevant column
    for col in columns:
        if col.lower() in query.lower():
            relevant_column = col
            break

    if relevant_column:
        # Handle numeric columns
        if data[relevant_column].dtype in ['int64', 'float64']:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data[relevant_column], bins=30, kde=True, ax=ax, color='blue')
            ax.set_title(f"Distribution of {relevant_column}")
            ax.set_xlabel(relevant_column)
            ax.set_ylabel("Frequency")

            # Save the graph as a byte stream
            img_buffer = BytesIO()
            fig.savefig(img_buffer, format='png')
            img_buffer.seek(0)
            plt.close(fig)  # Close the figure to save memory

            # Generate description using Cohere
            prompt = f"""
            You are a diamond industry assistant, Please don't use dataset instead use market. Based on the following data trend:
            - Column: {relevant_column}
            - Summary Statistics:
                - Mean: {data[relevant_column].mean():.2f}
                - Median: {data[relevant_column].median():.2f}
                - Standard Deviation: {data[relevant_column].std():.2f}

            Write a short and meaningful insight about the column {relevant_column}.
            """

            response = co.generate(
                model='command-r-plus-08-2024',
                prompt=prompt,
                max_tokens=100,
                temperature=0.7
            )

            description = response.generations[0].text.strip()

            return {
                "type": "graph",
                "description": description,
                "graph": img_buffer
            }

        # Handle categorical data
        elif data[relevant_column].dtype == 'object':
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.countplot(x=relevant_column, data=data, ax=ax, palette='viridis')
            ax.set_title(f"Distribution of {relevant_column}")
            ax.set_xlabel(relevant_column)
            ax.set_ylabel("Count")

            # Save the graph as a byte stream
            img_buffer = BytesIO()
            fig.savefig(img_buffer, format='png')
            img_buffer.seek(0)
            plt.close(fig)  # Close the figure to save memory

            # Generate description using Cohere
            top_value = data[relevant_column].value_counts().idxmax()
            top_count = data[relevant_column].value_counts().max()

            prompt = f"""
            You are a diamond industry assistant. Based on the following data trend:
            - Column: {relevant_column}
            - Most Common Value: {top_value} ({top_count} occurrences)

            Write a short and meaningful insight about the column {relevant_column}.
            """

            response = co.generate(
                model='command-r-plus-08-2024',
                prompt=prompt,
                max_tokens=100,
                temperature=0.7
            )

            description = response.generations[0].text.strip()

            return {
                "type": "graph",
                "description": description,
                "graph": img_buffer
            }

    # Default response if no relevant column is found
    return {"type": "text", "description": "No relevant data found"}

# Function to filter diamonds based on user preferences
def filter_diamonds(carat_range, color, cut):
    filtered_data = data.copy()

    if carat_range:
        filtered_data = filtered_data[(filtered_data['carat'] >= carat_range[0]) & (filtered_data['carat'] <= carat_range[1])]
    if color:
        filtered_data = filtered_data[filtered_data['color'] == color]
    if cut:
        filtered_data = filtered_data[filtered_data['cut'] == cut]

    return filtered_data

# Function to generate recommendations based on user preferences
def generate_recommendations(carat_range, color, cut):
    filtered_data = filter_diamonds(carat_range, color, cut)

    if filtered_data.empty:
        return "No diamonds match your criteria."
    
    # Get top 5 recommendations
    recommendations = filtered_data[['carat', 'cut', 'color', 'clarity', 'price']].head(5)

    # Prepare the prompt for Cohere to generate a dynamic summary
    prompt = "You are an expert in diamonds. Based on the following top 5 diamond recommendations, please write a detailed and insightful description:\n"
    
    for idx, row in recommendations.iterrows():
        prompt += f"- {row['carat']} carat {row['cut']} cut diamond, {row['color']} color, clarity {row['clarity']} - Price: ${row['price']:.2f}\n"
    
    # Prepare a summary
    summary = "Here are the recomendations based on your preferences:"

    # Generate a dynamic description from Cohere
    response = co.generate(
        model='command-r-plus-08-2024',
        prompt=prompt,
        max_tokens=200,
        temperature=0.7
    )

    # Extract and return the generated text
    return response.generations[0].text.strip(), summary, recommendations

# Chatbot UI
st.title("Diamond Chatbot")

if 'messages' not in st.session_state:
    st.session_state.messages = []

def handle_input():
    user_input = st.session_state.user_input

    if user_input:
        # Analyze the data and generate a response
        response = generate_response(user_input)

        # Save messages in session state
        st.session_state.messages = [("You", user_input)]  # Reset to only include the current conversation

        # Append bot response, including graph if present
        if response['type'] == 'graph':
            st.session_state.messages.append(("Bot", response['description'], response['graph']))
        else:
            st.session_state.messages.append(("Bot", response['description'], None))

        # Clear the user input
        st.session_state.user_input = ""

# Input and chat display
col1, col2 = st.columns([9, 1])

with col1:
    st.text_input("You", key='user_input', label_visibility='collapsed', on_change=handle_input)
with col2:
    st.button("⬆️", key='send_button', on_click=handle_input)

# Display messages in reverse order (most recent first)
for message in reversed(st.session_state.messages):
    # Ensure the message follows the expected structure
    if len(message) < 2:
        continue
    
    # Display User message first
    if message[0] == "You":
        st.write(f"**You:** {message[1]}")
    
    # Display Bot response
    elif message[0] == "Bot":
        st.write(f"**Bot:** {message[1]}")
        
        # Separate handling for Graph (Image) and Table (DataFrame)
        if len(message) > 2:
            if isinstance(message[2], BytesIO):  # If it's an image
                st.image(message[2])
            elif isinstance(message[2], pd.DataFrame):  # If it's a DataFrame
                st.dataframe(message[2])


# Sidebar for recommendations
st.sidebar.header("Get Diamond Recommendations")

# Initialize session state for sidebar inputs
if 'carat_range' not in st.session_state:
    st.session_state.carat_range = (1.0, 2.0)

if 'color' not in st.session_state:
    st.session_state.color = "D"

if 'cut' not in st.session_state:
    st.session_state.cut = "Fair"

# Sidebar input for carat range
carat_range = st.sidebar.slider(
    "Select Carat Range",
    min_value=0.2,
    max_value=5.0,
    value=st.session_state.carat_range,
    step=0.1
)

# Sidebar input for color
color = st.sidebar.selectbox(
    "Select Diamond Color",
    options=["D", "E", "F", "G", "H", "I", "J"],
    index=["D", "E", "F", "G", "H", "I", "J"].index(st.session_state.color)
)

# Sidebar input for cut
cut = st.sidebar.selectbox(
    "Select Diamond Cut",
    options=["Fair", "Good", "Very Good", "Ideal", "Excellent"],
    index=["Fair", "Good", "Very Good", "Ideal", "Excellent"].index(st.session_state.cut)
)

# Button to trigger recommendations
if st.sidebar.button("Get Recommendations"):
    # Update session state with current selections
    st.session_state.carat_range = carat_range
    st.session_state.color = color
    st.session_state.cut = cut

    # Generate recommendations if the button is clicked
    description, summary, recommendations = generate_recommendations(
        st.session_state.carat_range, 
        st.session_state.color, 
        st.session_state.cut
    )

    # Display the recommendations table and dynamic description
    st.session_state.messages.append(("Bot", summary))
    st.session_state.messages.append(("Bot", description))
    st.session_state.messages.append(("Bot", "Top 5 recommended diamonds:", recommendations))
    st.write(recommendations)


    #   # Add the query details to the chatbot conversation
    # st.session_state.messages.append(( "You", f"Get Recommendations for {st.session_state.carat_range} carat, {st.session_state.color} color, {st.session_state.cut} cut"))

    # # If recommendations are found, add both summary and description to chatbot conversation
    # if not recommendations.empty:
    #     # Append summary and description
    #     st.session_state.messages.append(("Bot", summary))
    #     st.session_state.messages.append(("Bot", description))

    #     # Show the recommendations table as part of the response
    #     st.session_state.messages.append(("Bot", "Here are the top 5 recommended diamonds:", None))

    #      # Directly append the recommendation table to the bot's response as a message
    #     st.session_state.messages.append(("Bot", "Recommendation Table", recommendations))

    #     # Display the recommendations table
    #     st.write("#### Recommendations")
    #     st.dataframe(recommendations)
    # else:
    #     # If no recommendations match, display the appropriate message
    #     st.session_state.messages.append(("Bot", "No diamonds match your criteria.", None))
