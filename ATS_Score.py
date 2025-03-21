import streamlit as st
import fitz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import os
import pickle
import io
from googleapiclient.http import MediaIoBaseDownload
import smtplib
from email.mime.text import MIMEText
import re
import torch
from transformers import BertTokenizer, BertModel
import numpy as np

# Google Drive API Setup
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

# Load BERT model and tokenizer (cached for performance)
@st.cache_resource
def load_bert():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    return tokenizer, model

tokenizer, model = load_bert()

def authenticate_drive():
    """Authenticate with Google Drive API and return service object."""
    creds = None
    token_path = 'token.pickle'
    if os.path.exists(token_path):
        with open(token_path, 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
            with open(token_path, 'wb') as token:
                pickle.dump(creds, token)
    return build('drive', 'v3', credentials=creds)

def list_files_in_folder(service, folder_id):
    """List all files in a Google Drive folder."""
    query = f"'{folder_id}' in parents"
    results = service.files().list(q=query, fields="files(id, name, mimeType)").execute()
    return results.get('files', [])

def download_file(service, file_id):
    """Download a file from Google Drive as a BytesIO object."""
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    fh.seek(0)
    return fh

# Text Processing Functions
def extract_text_from_pdf(file):
    """Extract text from a PDF file (file object or BytesIO)."""
    try:
        pdf = fitz.open(stream=file.read(), filetype="pdf") if isinstance(file, io.BufferedReader) else fitz.open(stream=file.getvalue(), filetype="pdf")
        text = "".join(page.get_text() for page in pdf)
        pdf.close()
        return text
    except Exception as e:
        st.error(f"Error extracting text: {e}")
        return ""

def get_tfidf_keywords(text, top_n=10):
    """Extract top TF-IDF keywords from text."""
    vectorizer = TfidfVectorizer(stop_words='english', max_features=50)
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray()[0]
    keyword_indices = scores.argsort()[-top_n:][::-1]
    return [feature_names[i] for i in keyword_indices]

def get_bert_embeddings(text):
    """Generate BERT embeddings for text."""
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

def calculate_hybrid_score(resume_text, jd_text, tfidf_weight=0.4, bert_weight=0.6):
    """Calculate a hybrid score using TF-IDF and BERT embeddings."""
    # TF-IDF Score
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])
    tfidf_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    
    # Extract top keywords and compute BERT score on filtered text
    resume_keywords = " ".join(get_tfidf_keywords(resume_text))
    jd_keywords = " ".join(get_tfidf_keywords(jd_text))
    resume_bert = get_bert_embeddings(resume_keywords)
    jd_bert = get_bert_embeddings(jd_keywords)
    bert_score = cosine_similarity(resume_bert, jd_bert)[0][0]
    
    # Hybrid score (weighted average)
    hybrid_score = (tfidf_weight * tfidf_score) + (bert_weight * bert_score)
    return hybrid_score * 100  # Return as percentage

def extract_candidate_info(resume_text):
    """Extract name and email from resume text using regex."""
    email = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", resume_text)
    name_match = re.search(r"^[A-Za-z]+\s+[A-Za-z]+", resume_text, re.MULTILINE)
    name = name_match.group() if name_match else "Unknown"
    return name, email.group() if email else "Unknown"

# Email Sending Function
def send_email(to_email, subject, body, sender_email, sender_password):
    """Send an email using Gmail SMTP."""
    try:
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = sender_email
        msg['To'] = to_email
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        return True
    except Exception as e:
        st.error(f"Email sending failed: {e}")
        return False

# Streamlit UI
def main():
    st.title("Resume Screening Tool")
    st.write("Choose your mode below to get started:")

    mode = st.radio("Select Mode", ("Job Seeker", "HR"))

    if mode == "Job Seeker":
        st.subheader("Job Seeker Mode")
        st.write("Upload your resume and enter a job description to see your matching score.")
        
        resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
        jd_text = st.text_area("Enter Job Description", height=150)
        
        if st.button("Calculate Score"):
            if resume_file and jd_text:
                resume_text = extract_text_from_pdf(resume_file)
                if resume_text:
                    score = calculate_hybrid_score(resume_text, jd_text)
                    st.success(f"Matching Score: {score:.2f}% (Hybrid TF-IDF + BERT)")
                    st.info("Score combines TF-IDF keyword relevance (40%) and BERT semantic similarity (60%).")
                else:
                    st.error("Could not process the resume.")
            else:
                st.warning("Please upload a resume and enter a job description.")

    elif mode == "HR":
        st.subheader("HR Mode")
        st.write("Enter a job description, Google Drive folder ID, and email credentials to analyze resumes.")
        
        jd_text = st.text_area("Enter Job Description", height=150)
        folder_id = st.text_input("Google Drive Folder ID")
        sender_email = st.text_input("Your Gmail Address")
        sender_password = st.text_input("Gmail App Password", type="password")
        
        if st.button("Analyze Resumes"):
            if jd_text and folder_id and sender_email and sender_password:
                try:
                    service = authenticate_drive()
                    files = list_files_in_folder(service, folder_id)
                    if not files:
                        st.warning("No files found in the folder.")
                        return
                    
                    candidates = []
                    for file in files:
                        if file['mimeType'] == 'application/pdf':
                            pdf_content = download_file(service, file['id'])
                            resume_text = extract_text_from_pdf(pdf_content)
                            if resume_text:
                                score = calculate_hybrid_score(resume_text, jd_text)
                                name, email = extract_candidate_info(resume_text)
                                candidates.append({"name": name, "email": email, "score": score, "file": file['name']})
                    
                    if candidates:
                        st.subheader("Candidate Rankings")
                        candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
                        selected = []
                        for cand in candidates:
                            if st.checkbox(f"{cand['name']} ({cand['email']}) - Score: {cand['score']:.2f}% - File: {cand['file']}"):
                                selected.append(cand)
                        
                        if selected and st.button("Send Emails to Selected"):
                            for cand in selected:
                                body = f"Dear {cand['name']},\n\nYou have been shortlisted based on your resume score of {cand['score']:.2f}%.\n\nRegards,\nHR Team"
                                if send_email(cand['email'], "Shortlisting Notification", body, sender_email, sender_password):
                                    st.success(f"Email sent to {cand['name']}!")
                    else:
                        st.warning("No valid PDF resumes found.")
                except Exception as e:
                    st.error(f"Error processing resumes: {e}")
            else:
                st.warning("Please fill all fields.")

if __name__ == "__main__":
    main()