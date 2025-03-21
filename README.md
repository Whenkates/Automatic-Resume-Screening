<h1> Automatic-Resume-Screening</h1>

# Overview
The <b>Automatic Resume Screening Tool</b> is a Python-based application designed to streamline the resume evaluation process for both job seekers and HR professionals. Built with Streamlit, it offers a dual-mode interface: one for individuals to assess resume-job description compatibility and another for HR to rank candidates from a Google Drive folder and send shortlisting emails. Leveraging BERT embeddings, Google Drive API, and NLP, this tool provides an efficient, semantic-driven alternative to traditional ATS systems.<br>

# Features
- <b>Job Seeker Mode:</b> Upload a resume (PDF) and input a job description to calculate a matching score using BERT embeddings for semantic accuracy.
- <b>HR Mode:</b> Connect to a Google Drive folder to process multiple resumes, rank candidates by relevance, extract names/emails with spaCy, and send shortlisting notifications via email.
- <b>Semantic Scoring:</b> Utilizes transformers and cosine similarity to analyze text beyond keywords, ensuring precise candidate-job alignment.
- <b>Dual-Mode Interface:</b> Offers role-based functionality (Job Seeker or HR) with an intuitive UI, integrating NLP and APIs for scalability.

 # Technologies Used
- <b>Python:</b> Core language for development.
- <b>Streamlit:</b> Interactive UI framework.
- <b>PyMuPDF (fitz):</b> PDF text extraction.
- <b>Transformers (BERT):</b> Semantic text analysis with bert-base-uncased.
- <b>spaCy:</b> NLP for preprocessing and entity extraction.
- <b>google-api-python-client:</b> Google Drive folder access.
- <b>smtplib:</b> Email sending functionality.
- <b>scikit-learn:</b> Cosine similarity computation.
- <b>torch:</b> Backend for BERT embeddings.
