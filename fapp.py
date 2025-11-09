# your_streamlit_app.py
import streamlit as st
import re
import uuid
import json
import time
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.metrics.pairwise import cosine_similarity # type: ignore
import nltk # type: ignore
from nltk.corpus import stopwords # type:ignore
from nltk.stem import PorterStemmer # type:ignore

# --- CONFIGURATION ---
DATA_FILE = "signal.json"
EXCEL_FILE = "technician.xlsx"
# Assuming these column names are correct in your 'technician.xlsx' file
RESUME_TEXT_COL = 'Resume Text'
AVAILABILITY_COL = 'Availability'
EXPERTISE_COL = 'Expertise Level'
REFRESH_INTERVAL_SECONDS = 5 # How often the Streamlit app checks for a new signal

# NLTK Setup
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')

STOPWORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()

# --- CRITICAL KEYWORDS ---
CRITICAL_KEYWORDS = {
    "HVAC_CORE": ["hvac", "thermostat", "compressor", "refrigerant", "freon", "cooling cycle", "r-22", "r-410a"],
    "ELECTRICAL_CTRL": ["capacitor", "contactor", "wiring diagram", "low voltage control", "relay", "breaker", "sensor"],
    "AIRFLOW_MECH": ["fan motor", "blower", "ductwork", "cfm", "coil", "filter", "bearing", "vibration"],
    "DIAGNOSTICS_SAFETY": ["leak detection", "pressure gauge", "safety valve", "troubleshooting", "epa certified", "multimeter", "triage"]
}
GROUP_BONUS = 5 # Reduced the group bonus to make the final score closer to 100%
EXPERTISE_MAP = {'Senior': 3, 'Mid': 2, 'Junior': 1} # Mapping for final ranking

# --- UTILITY FUNCTIONS ---

@st.cache_data
def load_employee_data():
    """Loads and caches the employee Excel data to avoid re-reading on every refresh."""
    try:
        df = pd.read_excel(EXCEL_FILE)
        if RESUME_TEXT_COL not in df.columns or AVAILABILITY_COL not in df.columns or EXPERTISE_COL not in df.columns:
            st.error(f"Excel file must contain columns: '{RESUME_TEXT_COL}', '{AVAILABILITY_COL}', and '{EXPERTISE_COL}'.")
            return None
        return df
    except FileNotFoundError:
        st.error(f"FATAL ERROR: Employee Excel file not found at '{EXCEL_FILE}'. Please ensure it exists.")
        return None
    except Exception as e:
        st.error(f"Error loading employee data: {e}")
        return None

def read_latest_signal():
    """Reads the latest signal data from the JSON file updated by FastAPI."""
    try:
        with open(DATA_FILE, 'r') as f:
            data = json.load(f)
        return data.get('problemStatement', ''), data.get('deviceID', 'N/A'), data.get('timestamp', 'N/A')
    except FileNotFoundError:
        # File doesn't exist yet, which is fine before the first signal
        return None, None, None
    except Exception as e:
        # Handle cases where file is corrupted or being written
        print(f"Error reading signal file: {e}")
        return None, None, None

# Using st.cache_data for this heavy function to optimize performance
@st.cache_data(show_spinner=False)
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in STOPWORDS and len(word) > 2]
    tokens = [STEMMER.stem(word) for word in tokens]
    return ' '.join(tokens)

@st.cache_data(show_spinner=False)
def calculate_match_percentage(resume_text, problem_statement):
    processed_resume = preprocess_text(resume_text)
    processed_jd = preprocess_text(problem_statement)
    if not processed_resume or not processed_jd:
        return 0, []
        
    documents = [processed_resume, processed_jd]
    
    # Use try-except block to handle cases where documents might be empty after processing
    try:
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3))
        tfidf_matrix = vectorizer.fit_transform(documents)
        
        # Cosine similarity for base score
        similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        base_percentage = round(similarity_score * 100)
    except ValueError:
        return 0, [] # Return 0 if vectorization fails

    # --- Keyword Bonus Logic ---
    resume_processed_words = set(processed_resume.split())
    group_hits = 0
    
    # Check if resume contains at least one keyword from each critical group
    for group, keywords in CRITICAL_KEYWORDS.items():
        if any(STEMMER.stem(keyword.lower()) in resume_processed_words for keyword in keywords):
            group_hits += 1

    bonus_score = group_hits * GROUP_BONUS
    final_percentage = min(100, base_percentage + bonus_score)

    # --- Identify Missing/Relevant Keywords from Problem Statement ---
    feature_names = vectorizer.get_feature_names_out()
    jd_vector = tfidf_matrix[1].toarray()[0]
    # Get top 5 keywords relevant to the problem statement
    top_jd_indices = jd_vector.argsort()[-5:][::-1]
    top_jd_keywords = [
        feature_names[i]
        for i in top_jd_indices
        if jd_vector[i] > 0.1 and feature_names[i] in processed_jd # Filter out low-importance and irrelevant words
    ]
    
    return final_percentage, top_jd_keywords

def process_all_candidates(df: pd.DataFrame, problem_statement: str):
    """Filters by availability and calculates match scores for all remaining candidates."""
    
    # 1. Availability Filter
    df_available = df[df[AVAILABILITY_COL].astype(str).str.strip().str.title() == 'Available'].copy()
    
    if df_available.empty:
        return pd.DataFrame()

    results = []
    
    for index, row in df_available.iterrows():
        # Using st.cache_data for the core matching function
        percentage, top_keywords = calculate_match_percentage(
            row[RESUME_TEXT_COL],
            problem_statement
        )
        
        candidate_name = row.get('Name', f"Employee {index + 1}")
        expertise_level = str(row[EXPERTISE_COL]).strip().title()
        
        results.append({
            "Name": candidate_name,
            "Match Percentage (%)": percentage,
            "Expertise": expertise_level,
            "Expertise Score": EXPERTISE_MAP.get(expertise_level, 0),
            "Relevant Problem Keywords": ", ".join(top_keywords[:3]),
        })

    df_results = pd.DataFrame(results)
    
    if df_results.empty:
        return pd.DataFrame()
        
    # 2. Final Ranking (Match Score then Expertise)
    df_results = df_results.sort_values(
        by=["Match Percentage (%)", "Expertise Score"],
        ascending=[False, False]
    ).drop(columns=['Expertise Score']).reset_index(drop=True)
    
    return df_results

# --- STREAMLIT APP LAYOUT & MAIN LOGIC ---

st.set_page_config(page_title="ATS for Technicians", layout="wide")
st.header("Automatic Technician Assignment")

# Create a placeholder for the live status and results
live_placeholder = st.empty()

# --- Main Streamlit Loop (The magic of automatic refresh) ---
while True:
    
    # Use the placeholder to update the entire section
    with live_placeholder.container():
        
        df_full = load_employee_data()
        if df_full is None:
            break # Stop processing if data loading fails

        problem_statement, device_id, timestamp = read_latest_signal()
        
        if problem_statement:
            try:
                formatted_time = pd.to_datetime(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                st.info(f"**LIVE SIGNAL RECEIVED** at {formatted_time}")
            except Exception:
                st.info(f"**LIVE SIGNAL RECEIVED** at (Time format error: using raw time) {timestamp}")
            st.markdown(f"**Device ID:** `{device_id}`")
            st.markdown(f"**Problem Statement:** ***{problem_statement}***")

            # --- AUTOMATIC PROCESSING TRIGGERED BY SIGNAL ---
            with st.spinner('Calculating skill match and checking real-time availability...'):
                df_ranked = process_all_candidates(df_full, problem_statement)
            
            # --- Display Results ---
            if df_ranked.empty:
                st.warning("No employees were matched and available for this problem.")
            else:
                st.success(f"Found {len(df_ranked)} available employees matched to the problem.")
                st.subheader("Recommended Technical Team (Highest Match First)")
                
                # Highlight the top row (the assigned technician)
                def highlight_top_row(row):
                    if row.name == 0:
                        return ['background-color: #d4edda'] * len(row) # Light green for the top match
                    return [''] * len(row)

                st.dataframe(
                    df_ranked.style
                        .format({"Match Percentage (%)": "{:.0f}%"})
                        .apply(highlight_top_row, axis=1),
                    use_container_width=True,
                    height=min(500, (len(df_ranked) + 1) * 35 + 3),
                )
                
                top_match_name = df_ranked.iloc[0]['Name']
                st.markdown(f"""
                    <div style='background-color:#f0f2f6; padding:10px; border-radius:8px; margin-top:15px; border-left: 5px solid #008000;'>
                        **DISPATCH ALERT:** The system recommends and assigns the task to: <b>{top_match_name}</b>.
                    </div>""", unsafe_allow_html=True)

        else:
            st.markdown(f"""
                <div style='text-align: center; padding: 20px; background-color: #fff3cd; border-radius: 10px;'>
                    <h3>Awaiting Signal...</h3>
                    <p>The system is actively listening for the HTTP POST request from the mobile simulator.</p>
                    <p>Last checked: {pd.Timestamp.now().strftime('%H:%M:%S')}</p>
                </div>
            """, unsafe_allow_html=True)
            
    # --- Auto-Refresh ---
    time.sleep(REFRESH_INTERVAL_SECONDS)