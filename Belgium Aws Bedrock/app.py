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
