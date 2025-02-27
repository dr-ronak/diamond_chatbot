import streamlit as st
import boto3
import json
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


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

def preprocess_timestamps(df, columns):
    """Convert timestamp columns to numerical format."""
    for col in columns:
        try:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            df[col] = df[col].apply(lambda x: x.timestamp() if pd.notnull(x) else np.nan)
        except Exception as e:
            print(f"Error converting {col}: {e}")
    return df

def predict_machine_failures(df):
    """Train a model and predict failures based on historical data."""
    try:
        features = ["F Issue Days", "Return Cts", "Loss Cts", "Time Diff", "Transaction Time", "Return Time", "Issue Carats", "Value"]
        timestamp_columns = ["Transaction Time", "Return Time"]
        df = preprocess_timestamps(df, timestamp_columns)
        
        df = df[features].dropna()
        df["Failure"] = (df["Loss Cts"] > df["Loss Cts"].median()).astype(int)
        
        X = df.drop(columns=["Failure"])
        y = df["Failure"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        feature_importances = model.feature_importances_
        
        df_result = pd.DataFrame(X_test)
        df_result["Predicted Failure"] = predictions
        failure_rate = df_result["Predicted Failure"].mean() * 100
        important_factors = sorted(zip(features, feature_importances), key=lambda x: x[1], reverse=True)
        top_factors = [f"{factor[0]} (Importance: {factor[1]:.2f})" for factor in important_factors[:3]]
        
        # Generate explanations and suggestions dynamically using AWS Bedrock
        bedrock_prompt = f"""
        Human: Based on machine failure predictions, the failure rate is {failure_rate:.2f}%. The top contributing factors are:
        - {top_factors[0]}
        - {top_factors[1]}
        - {top_factors[2]}
        
        Please analyze this data and provide insights on why failures are occurring and actionable suggestions to reduce failures.
        
        Assistant:"""
        
        payload = {"prompt": bedrock_prompt.strip(), "max_tokens_to_sample": 500}
        response = bedrock.invoke_model(modelId="anthropic.claude-v2", body=json.dumps(payload))
        bedrock_result = json.loads(response['body'].read())['completion']
        
        return f"Predicted machine failure rate: **{failure_rate:.2f}%**\n\nTop Contributing Factors:\n- " + "\n- ".join(top_factors) + "\n\nInsights & Improvement Suggestions:\n" + bedrock_result
    except Exception as e:
        return f"Error in prediction: {str(e)}"

def chat_with_csv(df, user_query):
    if "predict" in user_query.lower() and "machine failure" in user_query.lower():
        return predict_machine_failures(df)
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

    Additional Feature: If the user asks about **predicting machine failures**, analyze historical trends and usage patterns to provide a **failure probability**.

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
