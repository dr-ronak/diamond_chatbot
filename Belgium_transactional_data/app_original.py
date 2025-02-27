import streamlit as st
import boto3
import json
import os
import pandas as pd
from dotenv import load_dotenv


load_dotenv()

aws_region = "us-east-1"  # Change this to your AWS region
aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

# Initialize Bedrock Client
bedrock = boto3.client(
    "bedrock-runtime",
    region_name=aws_region,
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key
)

def chat_with_csv(df, user_query):
    # Convert the first few rows of the CSV to string
    csv_sample = df.head(5).to_string(index=False)

    # Correctly format the prompt
    formatted_prompt = f"""
    Human: Here is a sample of my dataset:\n{csv_sample}\n\nNow, answer this question: {user_query}

    Role: You are an expert data analyst specializing in diamond processing and supply chain management. Your task is to generate insightful, precise, and data-driven responses to user queries based on the provided dataset.

    Dataset Columns:
    The dataset includes inventory transactions, processing details, entity movements, and quality attributes related to diamonds. Below are the relevant fields:

    Transaction Details: Lotno, Pkt, Dept, Pktno, Item, Trans Date, InvNo, InvDate, Number, Process, Fr Process, Time Diff, Transaction Time, Return Time, Droid, Rfid  
    Inventory Metrics: Issue Carats, F Issue Days, Return Cts, Loss Cts, Org Pcs, Value  
    Diamond Quality: Shape, Cut, Purity, Color, Flour, Polish, Symm, Exp Pol  
    Entity Details: Fr P Entity Type, Fr P Entity, Fr Entity Type, Fr Entity, To P Entity Type, To P Entity, To Entity Type, To Entity  

    Efficiency Analysis & Comparison:
    - If the user asks about efficiency, compare metrics **quarter-over-quarter** if historical data is available.
    - Key Efficiency KPIs to track:
    - **Total Issue Carats**: Sum of all issue carats per quarter.
    - **Average Processing Time Per Stone**: Mean `Time Diff` for transactions.
    - **Loss Percentage**: `(Loss Cts / Issue Carats) * 100`.
    - **Return Efficiency**: `(Return Cts / Issue Carats) * 100`.

    Efficiency Response Guidelines:
    - If data is available for multiple quarters:
    - Compare the latest quarter with the previous quarter.
    - Mention **% increase/decrease** in key metrics.
    - Identify efficiency trends and operational insights.
    - If historical data is missing:
    - Respond: "I currently do not have enough historical data to compare efficiency. Please provide aggregated quarterly metrics for accurate analysis."
    - Format insights with:
    - ✅ *Positive Trends*: "Efficiency improved! We processed **X% more diamonds**, and processing time decreased by **Y%**."
    - ⚠️ *Negative Trends*: "Warning: Loss percentage increased by **X%**, indicating possible inefficiencies."
    - 📈 *Improved Returns*: "Good news! **Return Efficiency increased by X%**, ensuring better handling of returned items."

    Response Style:
    - Provide concise yet insightful responses.
    - Use natural language and industry-relevant terminology.
    - When possible, include trends, comparisons, averages, and key takeaways.
    - If applicable, generate **tabular summaries** for better clarity.
    - If a query lacks context, ask for clarification.

    Assistant:"""


    payload = {
        "prompt": formatted_prompt.strip(),  # Ensure correct structure
        "max_tokens_to_sample": 500  # Use correct key name for AWS Bedrock
        

    }

    # Invoke AWS Bedrock Model
    response = bedrock.invoke_model(
        modelId="anthropic.claude-v2",
        body=json.dumps(payload)
    )

    # Extract and return the response
    result = json.loads(response['body'].read())['completion']
    return result

st.set_page_config(layout='wide')
st.title("ChatCSV powered by AWS Bedrock")

input_csv = st.file_uploader("Upload your CSV file", type=['csv'])

if input_csv is not None:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.info("CSV Uploaded Successfully")
        data = pd.read_csv(input_csv, encoding="ISO-8859-1")  # Alternative: "latin1"
        st.dataframe(data, use_container_width=True)

    with col2:
        st.info("Chat Below")
        input_text = st.text_input("Enter your query")

        if input_text and st.button("Chat with data"):
            st.info("Your Query: " + input_text)
            result = chat_with_csv(data, input_text)
            st.success(result)
