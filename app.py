import streamlit as st
import re
import uuid
from sklearn.feature_extraction.text import TfidfVectorizer#type:ignore
from sklearn.metrics.pairwise import cosine_similarity#type:ignore
import nltk#type:ignore
from nltk.corpus import stopwords#type:ignore
from nltk.stem import PorterStemmer # IMPORT for stemming#type:ignore
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- REQUIRED EXCEL COLUMN NAMES ---
RESUME_TEXT_COL = 'Resume Text'
AVAILABILITY_COL = 'Availability'
EXPERTISE_COL = 'Expertise Level' # Used for tie-breaking/display

# NLTK Setup
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')

STOPWORDS = set(stopwords.words('english'))
# Initialize Stemmer for linguistic normalization
STEMMER = PorterStemmer()

# *** UPDATED CRITICAL KEYWORDS FOR HVAC/TECHNICAL TRIAGE ***
CRITICAL_KEYWORDS = {
    # Core HVAC/Refrigeration Knowledge
    "HVAC_CORE": ["hvac", "thermostat", "compressor", "refrigerant", "freon", "cooling cycle", "r-22", "r-410a"],
    
    # Electrical/Control Systems Troubleshooting
    "ELECTRICAL_CTRL": ["capacitor", "contactor", "wiring diagram", "low voltage control", "relay", "breaker", "sensor"],
    
    # Airflow, Mechanics, and Components (e.g., fan repair)
    "AIRFLOW_MECH": ["fan motor", "blower", "ductwork", "cfm", "coil", "filter", "bearing", "vibration"],
    
    # Diagnostics, Safety, and Tools
    "DIAGNOSTICS_SAFETY": ["leak detection", "pressure gauge", "safety valve", "troubleshooting", "epa certified", "multimeter", "triage"]
}
GROUP_BONUS = 50

# --- UTILITY FUNCTIONS ---

def custom_button(label, key, color="#1E88E5", hover_color="#1565C0"):
    """Generates a custom-styled, interactive button using HTML and CSS."""
    unique_id = str(uuid.uuid4())
    st.markdown(f"""
    <style>
    #{unique_id} .stButton > button {{
        background-color: {color};
        color: white;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 8px;
        border: none;
        transition-duration: 0.4s;
        width: 100%;
    }}
    #{unique_id} .stButton > button:hover {{
        background-color: {hover_color};
        color: white;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    }}
    </style>
    <div id="{unique_id}">
    </div>
    """, unsafe_allow_html=True)
    return st.button(label, key=key, use_container_width=True)

def preprocess_text(text):
    """Clean text: lowercase, remove non-alphanumeric, remove stopwords, and ADD STEMMING."""
    if not isinstance(text, str):
        return ""
        
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in STOPWORDS and len(word) > 2]
    tokens = [STEMMER.stem(word) for word in tokens]
    
    return ' '.join(tokens)

def calculate_match_percentage(resume_text, job_description):
    """Calculates match percentage using TF-IDF and Cosine Similarity (ML/NLP), with Hybrid Scoring."""
    
    processed_resume = preprocess_text(resume_text)
    processed_jd = preprocess_text(job_description) # JD is now the Problem Statement
    
    if not processed_resume or not processed_jd:
        return 0, []
    
    documents = [processed_resume, processed_jd]
    
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3))
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    base_percentage = round(similarity_score * 100)
    
    # --- HYBRID SCORING: SKILL GROUP BONUS ---
    resume_processed_words = set(processed_resume.split())
    group_hits = 0
    found_groups = set()
    
    for _, keywords in CRITICAL_KEYWORDS.items():
        if len(found_groups) < len(CRITICAL_KEYWORDS):
            for keyword in keywords:
                if STEMMER.stem(keyword.lower()) in resume_processed_words:
                    group_hits += 1
                    found_groups.add(keyword)
                    break

    bonus_score = group_hits * GROUP_BONUS
    final_percentage = min(100, base_percentage + bonus_score)
    
    # Keyword Extraction (Top 5 for reporting)
    feature_names = vectorizer.get_feature_names_out()
    jd_vector = tfidf_matrix[1].toarray()[0]
    top_jd_indices = jd_vector.argsort()[-5:][::-1]
    top_jd_keywords = [
        feature_names[i] for i in top_jd_indices
        if jd_vector[i] > 0.1 and feature_names[i] in processed_jd
    ]
    
    return final_percentage, top_jd_keywords

def process_all_candidates(df, problem_statement):
    """Processes all candidates, filters by availability, and calculates match scores."""
    
    if RESUME_TEXT_COL not in df.columns or AVAILABILITY_COL not in df.columns or EXPERTISE_COL not in df.columns:
        st.error(f"Excel file must contain columns: '{RESUME_TEXT_COL}', '{AVAILABILITY_COL}', and '{EXPERTISE_COL}'.")
        return pd.DataFrame()

    results = []
    
    # Define a simple scoring system for expertise level for better sorting
    expertise_map = {'Senior': 3, 'Mid': 2, 'Junior': 1}
    
    with st.spinner('Calculating skill match and checking availability...'):
        for index, row in df.iterrows():
            
            # 1. Filter Check (Availability)
            availability = str(row[AVAILABILITY_COL]).strip().title()
            
            # IMPORTANT: Only process candidates marked 'Available'
            if availability != 'Available':
                continue

            resume_text = row[RESUME_TEXT_COL]
            
            # Fallback for Name column
            candidate_name = row.get('Name', f"Employee {index + 1}")
            
            try:
                # 2. Skill Match Calculation
                percentage, top_keywords = calculate_match_percentage(resume_text, problem_statement)
                
                # 3. Prepare result row
                expertise_level = str(row[EXPERTISE_COL]).strip().title()
                
                results.append({
                    "Name": candidate_name,
                    "Match Percentage (%)": percentage,
                    "Expertise": expertise_level,
                    "Expertise Score": expertise_map.get(expertise_level, 0),
                    "Top 3 Missing Keywords": ", ".join(top_keywords[:3]), # Use keywords for reporting
                })
            except Exception as e:
                # Log error but continue processing other candidates
                print(f"Error processing {candidate_name}: {e}")

    df_results = pd.DataFrame(results)
    
    if df_results.empty:
        return pd.DataFrame()
        
    # 4. Final Ranking: Sort by Match Percentage (Primary) and Expertise Score (Secondary)
    df_results = df_results.sort_values(
        by=["Match Percentage (%)", "Expertise Score"],
        ascending=[False, False]
    ).drop(columns=['Expertise Score']).reset_index(drop=True)

    return df_results


# --- STREAMLIT APP LAYOUT ---
    
st.set_page_config(page_title="Triage Skill Matcher", layout="wide")
st.header("ATS for Technicians")

st.info("Upload your employee skills database (Excel) and describe the problem to find the most suitable, available technician.")

# --- USER INPUTS ---
col_file, col_problem = st.columns([1, 2])

with col_file:
    uploaded_excel_file = st.file_uploader(
        "Upload Employee Database (Excel .xlsx / .xls)",
        type=["xlsx", "xls"],
        accept_multiple_files=False,
        key='uploaded_excel_input',
        help=f"Must contain columns: **{RESUME_TEXT_COL}**, **{AVAILABILITY_COL}**, and **{EXPERTISE_COL}**."
    )
    
with col_problem:
    problem_statement = st.text_area(
        "Problem Statement (AC is broken): ",
        key='input',
        height=150,
        placeholder="E.g., The AC unit is blowing warm air and the condenser fan is not spinning. We suspect a bad capacitor or fan motor failure.",
        help="The problem statement is treated as the Job Description."
    )

# --- EXECUTION BUTTON ---
submit_ranked_list = custom_button(
    label="FIND AVAILABLE TECHNICIANS (Rank by Skill Match)",
    key="ranked_list_btn",
    color="#008000",
    hover_color="#006400"
)

# --- MAIN LOGIC EXECUTION ---

if submit_ranked_list:
    if uploaded_excel_file and problem_statement:
        
        try:
            df_full = pd.read_excel(uploaded_excel_file)
        except Exception as e:
            st.error(f"Failed to read Excel file: {e}")
            st.stop()
            
        # Call the new processing function
        df_ranked = process_all_candidates(df_full, problem_statement)
        
        if df_ranked.empty:
            st.warning("No employees were matched and available for triage. Check your Excel column names, or ensure at least one employee is marked 'Available'.")
        else:
            st.success(f"Found {len(df_ranked)} available employees matched to the problem.")
            st.subheader("Recommended Technical Team (Highest Match First)")
            
            # Display the Final Ranked Table
            st.dataframe(
                df_ranked.style.format({"Match Percentage (%)": "{:.0f}%"}).apply(
                    lambda x: [
                        'background-color: #e6ffe6' if i == 0 else '' # Highlight the top candidate
                        for i in range(len(x))
                    ],
                    axis=0,
                    subset=pd.IndexSlice[0, ['Name', 'Match Percentage (%)', 'Expertise']]
                ),
                use_container_width=True,
                height=min(500, (len(df_ranked) + 1) * 35 + 3),
            )
            
            st.markdown(f"""<div style='background-color:#f0f2f6; padding:10px; border-radius:8px; margin-top:15px;'>
                         The recommended team members are all confirmed to be **'Available'**.</div>""", unsafe_allow_html=True)
            
    else:
        st.warning("Please upload the Employee Database Excel file and provide a Problem Statement.")