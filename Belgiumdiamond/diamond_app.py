import streamlit as st
import pandas as pd
import cohere
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from datetime import datetime
from pandas.api.types import CategoricalDtype
import os

# Initialize Cohere API
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "RJ0sqkX4xHHilGex3Jtu2ewvj8sdQAwAUE6FNQ2H")
co = cohere.Client(COHERE_API_KEY)

# Load dataset
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path, sep=";", encoding="latin1", low_memory=False)
    for col in df.columns:           # Convert numeric columns
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass  # Ignore non-numeric columns

    
    date_cols = ["TRANS. DATE", "ENTRY DATE", "ACK DATE"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

# Ensure the correct dataset is loaded
data = load_data("SampleData_For_AI.csv")

# Function to process queries
def process_query(query, data):
    query = query.lower()
    
    # If user asks for column names
    if "column names" in query:
        return {"type": "text", "data": list(data.columns)}
    
    # If user asks for unique values in a column
    elif "unique " in query:
        column_name = None
        for col in data.columns:
            if col.lower() in query:
                column_name = col
                break

        if column_name:  # If the column is found
            unique_values = data[column_name].unique()
            unique_count = len(unique_values)

            #Create Dataframe 
            unique_df = pd.DataFrame({column_name: unique_values})

            return {"type": "combined",
                    "text":f"There are {unique_count} unique values in {column_name}",
                    "table": unique_df
                    }

        else:  # If the column is not found
            return {"type": "text", "data": "Sorry, the specified column is not available in the dataset."}

    # If user asks for the number of rows in the dataset
    elif "rows" in query or "records" in query:
        num_rows = data.shape[0]  # Get the number of rows
        return {"type": "text", "data": f"The dataset contains {num_rows} rows."}

    # If user asks for non-null values in a column
    elif "not null" in query or "not missing" in query or "no null" in query or "no missing" in query:
        column_name = None
        for col in data.columns:
            # Check if the column name is exactly the same as in the query (case-insensitive)
            if col.lower() == query.lower().split(' ')[-3]:  # Attempt to match last part of query for column name
                column_name = col
                break  # Stop once we find the exact column name

        if column_name:  # If a column is found
            # Get the count of non-null values in the selected column
            not_null_count = data[column_name].notna().sum()  # Count non-null values directly

            # Return the response with the count
            return {
                "type": "text",
                "data": f"There are {not_null_count} non-null values in {column_name}."
            }
        else:
            return {"type": "text", "data": "Sorry, the specified column is not available in the dataset."}

    # If user asks for missing values in a column
    elif "missing" in query or "null" in query:
        column_name = None
        for col in data.columns:
            if col.lower() in query:
                column_name = col
                break

        if column_name:  # If the column is found
            # Get missing values for the specific column
            missing_values = data[column_name].isnull().sum()
            
            # Create a DataFrame for the response (return as a table)
            missing_data = pd.DataFrame({'Column': [column_name], 'Missing Values': [missing_values]})
            
            # Return a table response
            return {"type": "table", "data": missing_data}
        else:  # If the column is not found
            return {"type": "text", "data": "Sorry, the specified column is not available in the dataset."}

    # If user asks for the total of a numeric column
    elif "total" in query or "sum" in query:
        # Check if the query is asking about a numeric column
        for col in data.select_dtypes(include=['number']).columns:
            if col.lower() in query:
                total_amount = data[col].sum()  # Calculate the sum
                return {"type": "text", "data": f"The total amount for {col} is {total_amount:.2f}."}

    # If user asks for unique dates
    elif "unique dates" in query:
        column_name = "TRANS. DATE"  # Specify the correct column name for dates
        if column_name in data.columns:  # Check if the specified column exists in the dataset
            unique_dates = data[column_name].dropna().unique()  # Get unique dates in the specified column
            return {"type": "text", "data": list(unique_dates)}
        else:
            return {"type": "text", "data": f"Sorry, '{column_name}' column is not available in the dataset."}

    # If user asks about a specific date
    elif "how many transactions happened on" in query.lower():
        # Extract the date from the query (assuming format like '2025-01-01')
        date_str = query.split("on")[-1].strip()  # Extract everything after 'on' and strip extra spaces

        # Strip both single and double quotes in one line, just in case.
        date_str = date_str.replace("'", "").replace('"', '')

        try:
            # Convert the string date into a datetime object for comparison
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            return {"type": "text", "data": f"Sorry, the date format is invalid. Please provide a valid date like 'YYYY-MM-DD'. You entered: {date_str}"}

        # Check if 'TRANS. DATE' column exists in the dataset
        if "TRANS. DATE" in data.columns:
            # Filter data for transactions that happened on the specified date
            transactions_on_date = data[data["TRANS. DATE"] == date_obj]
            transaction_count = len(transactions_on_date)
            
            if transaction_count > 0:
                return {"type": "text", "data": f"There were {transaction_count} transactions on {date_str}."}
            else:
                return {"type": "text", "data": f"Sorry, there were no transactions on {date_str}."}
        else:
            return {"type": "text", "data": "Sorry, 'TRANS. DATE' column is not available in the dataset."}

    # If user asks for the count of unique dates in the "TRANS. DATE" column
    elif "count of unique dates" in query or "how many unique dates" in query:
        unique_dates_count = data["TRANS. DATE"].nunique()  # Count unique dates in the "TRANS. DATE" column
        return {"type": "text", "data": f"There are {unique_dates_count} unique transaction dates."}

    # If user asks for the count of different values in the "TO TRANS TYPE" column
    elif "to trans type" in query and ("count" in query or "different" in query):
        if "TO TRANS TYPE" in data.columns:
            # Get the count of different "TO TRANS TYPE"
            trans_type_counts = data["TO TRANS TYPE"].value_counts().reset_index()
            trans_type_counts.columns = ["TO TRANS TYPE", "Count"]
            return {"type": "table", "data": trans_type_counts}
        else:
            return {"type": "text", "data": "'TO TRANS TYPE' column is not available in the dataset."}

    # If user asks for the average of any numeric column
    elif "average" in query or "avg" in query:
        # Extract the column name from the query (search for specific column mentioned in the query)
        column_name = None
        for col in data.columns:
            if col.lower() in query:
                column_name = col
                break

        if column_name and column_name in data.columns:  # Check if the column exists
            if pd.api.types.is_numeric_dtype(data[column_name]):  # Check if the column is numeric
                avg_value = data[column_name].mean()  # Calculate the average
                return {"type": "text", "data": f"The average of {column_name} is {avg_value:.2f}."}
            else:
                return {"type": "text", "data": f"Sorry, {column_name} is not a numeric column."}
        else:
            return {"type": "text", "data": "Sorry, the specified column is not available in the dataset."}

    # If user asks for a graph, plot histogram, count plot, pie chart, or line chart based on column type
    elif "distribution" in query or "graph" in query or "plot" in query:
        column_name = None
        # Try to identify which column the user is referring to for the graph
        for col in data.columns:
            if col.lower() in query:
                column_name = col
                break

        if column_name and column_name in data.columns:
            # If it's a numeric column, plot a histogram
            if pd.api.types.is_numeric_dtype(data[column_name]):
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(data[column_name].dropna(), bins=30, kde=True, ax=ax, color="blue")
                ax.set_title(f"Distribution of {column_name} (Histogram)")
                
                # Save the plot to a buffer to display in Streamlit
                img_buffer = BytesIO()
                fig.savefig(img_buffer, format="png")
                img_buffer.seek(0)
                
                # Return the graph
                return {"type": "graph", "graph": img_buffer}
            
            # If it's a categorical column, plot a count plot or pie chart
            elif isinstance(data[column_name].dtype, CategoricalDtype) or data[column_name].dtype == 'object':
                # Check if the user specifically wants a pie chart
                if "pie" in query.lower():
                    # Plot a pie chart for categorical data
                    fig, ax = plt.subplots(figsize=(10, 6))
                    data[column_name].value_counts().plot.pie(autopct='%1.1f%%', ax=ax, colors=sns.color_palette("Set2", len(data[column_name].unique())))
                    ax.set_title(f"Distribution of {column_name} (Pie Chart)")
                    ax.set_ylabel('')  # Remove y-axis label for pie chart
                    
                    # Save the plot to a buffer to display in Streamlit
                    img_buffer = BytesIO()
                    fig.savefig(img_buffer, format="png")
                    img_buffer.seek(0)
                    
                    # Return the graph
                    return {"type": "graph", "graph": img_buffer}
                else:
                    # Plot a count plot if it's not a pie chart request
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.countplot(data=data, x=column_name, ax=ax, palette="Set2", hue=column_name, legend=False)
                    ax.set_title(f"Distribution of {column_name} (Count Plot)")
                    
                    # Save the plot to a buffer to display in Streamlit
                    img_buffer = BytesIO()
                    fig.savefig(img_buffer, format="png")
                    img_buffer.seek(0)
                    
                    # Return the graph
                    return {"type": "graph", "graph": img_buffer}

            # If it's a datetime column, plot a line chart for trends over time
            elif pd.api.types.is_datetime64_any_dtype(data[column_name]):
                # Plot a line chart to show the trend over time
                fig, ax = plt.subplots(figsize=(10, 6))
                data[column_name].dropna().value_counts().sort_index().plot(kind='line', ax=ax, color="green")
                ax.set_title(f"Trend over Time for {column_name} (Line Chart)")
                ax.set_xlabel("Date")
                ax.set_ylabel("Count")

                # Save the plot to a buffer to display in Streamlit
                img_buffer = BytesIO()
                fig.savefig(img_buffer, format="png")
                img_buffer.seek(0)

                # Return the graph
                return {"type": "graph", "graph": img_buffer}

            
    # If user asks for filtered data
    elif "where" in query:
        conditions = query.split("where")[-1].strip()
        try:
            filtered_data = data.query(conditions)
            return {"type": "table", "data": filtered_data}
        except Exception as e:
            return {"type": "text", "data": f"Error processing query: {str(e)}"}

    # If user asks for a graph
    elif "graph" in query or "plot" in query or "distribution" in query:
        for col in data.select_dtypes(include=['number']).columns:
            if col.lower() in query:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(data[col].dropna(), bins=30, kde=True, ax=ax, color="blue")
                ax.set_title(f"Distribution of {col}")
                img_buffer = BytesIO()
                fig.savefig(img_buffer, format="png")
                img_buffer.seek(0)
                return {"type": "graph", "graph": img_buffer}

    # Default response using Cohere if query is ambiguous
    else:
        response = co.generate(
            model="command-r-plus-08-2024",
            prompt=f"Provide insights based on the dataset for: {query}",
            max_tokens=150,
            temperature=0.7,
        )
        return {"type": "text", "data": response.generations[0].text.strip()}

# Streamlit Chatbot UI
st.title("💎 Diamond Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

def handle_input():
    user_input = st.session_state.user_input.strip()
    if user_input:
        response = process_query(user_input, data)
        
        # Append the user's input to the conversation history
        st.session_state.messages.append(("You", user_input))
        
        if response is None:
            st.session_state.messages.append(("Bot", "Sorry, I couldn't process the query. Please try asking in a different way."))
        else:
            if response["type"] == "graph":
                st.session_state.messages.append(("Bot", "Here is the graph:"))
                st.session_state.messages.append(("Bot", "", response["graph"]))  # Graph image

            elif response["type"] == "table":
                st.session_state.messages.append(("Bot", "Here is the table:"))
                st.session_state.messages.append(("Bot", "", response["data"]))  # Table data

            elif response["type"] == "combined":
                        # Append bot's message to chat history first
                        st.session_state.messages.append(("Bot", response["text"]))

                        # Append table as a separate chat response
                        st.session_state.messages.append(("Bot", response["table"]))  # Append table below bot's message
                        
            else:
                st.session_state.messages.append(("Bot", response["data"]))  # Text response

        st.session_state.user_input = ""

col1, col2 = st.columns([9, 1])
with col1:
    st.text_input("You", key="user_input", label_visibility="collapsed", on_change=handle_input)
with col2:
    st.button("⬆️", key="send_button", on_click=handle_input)

# Now, render all the messages in order
for i in range(len(st.session_state.messages) - 2, -1, -2):
    # Display user message
    st.write(f"**You:** {st.session_state.messages[i][1]}")
        
    # Display bot response
    bot_response = st.session_state.messages[i + 1]
    st.write(f"**Bot:** {bot_response[1]}")
        
    # Render the graph or table below the bot's message
    if len(bot_response) > 2:
        if isinstance(bot_response[2], BytesIO):  # Graph (image)
            st.image(bot_response[2])
        else:  # Table data
            st.write(bot_response[2])  # Render the table below the bot's message
