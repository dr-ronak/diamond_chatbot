import streamlit as st
import pandas as pd
import os
import pdfplumber
import re
import nltk
from nltk.tokenize import word_tokenize
from openai import OpenAI
import openai
import cohere

# client = OpenAI(
#     api_key=os.environ.get("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"),  # This is the default and can be omitted
# )

openai.api_key = "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"

nltk.download('punkt_tab')

st.markdown(
    "<h1 style='text-align: center;'>Resume Analysis</h1>",
    unsafe_allow_html=True
)

st.sidebar.header("Pages")
nav = st.sidebar.radio('Navigation',['Page1','Page2','Page3'])

def extract_linkedin(text):
    linkedin_pattern =  r"(https?://(?:www\.)?linkedin\.com/[^\s]+)"
    linkedin_url = re.findall(linkedin_pattern, text)
    if linkedin_url:
        return linkedin_url[0]
    return "No linkedin Profile Found"

# Function to extract email address using regex
def extract_email(text):
    email_pattern = r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})"
    emails = re.findall(email_pattern, text)
    if emails:
        return emails[0]  # Return the first email found
    return None

if nav == 'Page1':

    

    upload_file = st.file_uploader("Upload Your Resume", type=['pdf'])

    Upload_Folder = 'uploaded_files'

    if not os.path.exists(Upload_Folder):
        os.makedirs(Upload_Folder)

    if upload_file is not None:
        file_path = os.path.join(Upload_Folder, upload_file.name)

        with open(file_path, 'wb') as f:
            f.write(upload_file.getbuffer())

    # Predefined skills list (you can expand it)
    skills_list = [
        "Python", "Java", "C++", "JavaScript", "HTML", "CSS", "SQL", 
        "Machine Learning", "Data Science", "Deep Learning", "TensorFlow", 
        "Keras", "PyTorch", "SQL", "Excel", "Power BI", "Data Analysis", 
        "Natural Language Processing", "Node.js", "React", "Django", "Flask"
    ]

    # Check file upload
    if upload_file is not None:
        # Extract text from file
        with pdfplumber.open(upload_file) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()

        
        linkedin_url = extract_linkedin(text)
        email_id = extract_email(text)
        
        # Tokenize the extracted text(Converts the sentences into word and stores it in lits form)
        words = word_tokenize(text.lower())   # convert the word in lower case for insensitivity

        # Find skills mentioned in resume
        mentioned_skill = [skill for skill in skills_list if skill.lower() in words]
        
        # Check if skilss were mentioned in the resume
        if mentioned_skill:

            # Display a selectbox with mentioned skills
            select_box = st.multiselect('Select Skill from the resume', mentioned_skill)

            # If LinkedIn URL is found, display a clickable link
        if linkedin_url:
            # Display the LinkedIn URL as a clickable link
            st.write("Your LinkedIn Profile")

            # linkedin = st.text_input("Your Linkedin id: ",linkedin_url)
            
            st.markdown(linkedin_url)
        else:
            st.warning("No LinkedIn profile found in the resume.")

       
        if email_id:
            # Display the email id as a clickable link
            st.write("Your Email Id")
            
            st.markdown(email_id)
        else:
            st.warning("No Email Id found in the resume.")



if nav == 'Page2':
    def analyze_resume(resume_text):
        prompt = f"""
        Analyze the following resume and extract key information suck as skill, work experience & Education

        Resume:
        {resume_text}

        Please return ther information in a structured format(JSON or bullet points ).
        """

        response = openai.chat.completions.create(
        messages=[
        {
            "role": "user",
            "content": "Say this is a test",
        }
        ],
         model="gpt-3.5-turbo",
        )
        return response['choices'][0]['message']['content'].strip()
    
    # App Layout
    st.write("Upload your resume in pdf format, and the tool will analyze it for skills, experience and education.")
    
    upload_file = st.file_uploader("Upload Your Resume", type=['pdf'])

    Upload_Folder = 'uploaded_files'

    if not os.path.exists(Upload_Folder):
        os.makedirs(Upload_Folder)

    if upload_file is not None:
        file_path = os.path.join(Upload_Folder, upload_file.name)

        with open(file_path, 'wb') as f:
            f.write(upload_file.getbuffer())

    # Check file upload
    if upload_file is not None:
        # Extract text from file
        with pdfplumber.open(upload_file) as pdf:
            resume_text = ""
            for page in pdf.pages:
                resume_text += page.extract_text()
            
            # Analyze the resume text
            if resume_text:
                analysis = analyze_resume(resume_text)
                st.subheader("Resume Analysis")
                st.write(analysis)
            else:
                st.error("The file is empty or cannot be processed.")
    # else:
    #     resume_text = upload_file.getvalue().decode("utf-8")
        
    
if nav == "Page3":

    co = cohere.ClientV2(api_key='vPjqsnaockV4uYWjNcv56thveWsekn4D3jJVzBLE')

    def analyze_resume(resume_text, question):
        prompt = f"""
            Based on the following resume, please answer the question:

            Resume:
            {resume_text}

            Question: {question}

            Answer:
            """
            # Generate a response from Cohere based on the prompt
        response = co.generate(
            model='command-r-plus-08-2024',  # Use a valid model if needed
            prompt=prompt,
            max_tokens=500  # Adjust the response length as necessary
        )

       # Access the generated text using response.generations
        return response.generations[0].text.strip()  # This gives you the response text

    # Streamlit App Layout
    st.write("Upload your resume in PDF format, and the tool will analyze it for skills, experience, and education.")

    upload_file = st.file_uploader("Upload Your Resume", type=['pdf'])

    Upload_Folder = 'uploaded_files'

    if not os.path.exists(Upload_Folder):
        os.makedirs(Upload_Folder)

    if upload_file is not None:
        file_path = os.path.join(Upload_Folder, upload_file.name)

        with open(file_path, 'wb') as f:
            f.write(upload_file.getbuffer())

    # Check file upload
    if upload_file is not None:
        # Extract text from the PDF file
        with pdfplumber.open(upload_file) as pdf:
            resume_text = ""
            for page in pdf.pages:
                resume_text += page.extract_text()

            #     # Show the resume text preview
            # st.subheader("Resume Preview")
            # st.write(resume_text)

                # Ask the user to input a question
            question = st.text_input("Ask a question about the resume (e.g., What are your skills?)")
          
            if question:
                # If the user asks a question, analyze the resume and get the answer
                analysis = analyze_resume(resume_text, question)
                st.subheader("Answer")
                st.write(analysis)
            
