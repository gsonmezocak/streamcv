import streamlit as st
import google.generativeai as genai
import firebase_admin
from firebase_admin import credentials, firestore, auth
import json
import numpy as np
import re
import pyrebase 
import time
import concurrent.futures
import pandas as pd

# --- THEME-FRIENDLY NEUMORPH + GLASS CSS ---
def load_custom_css():
    st.markdown("""
    <style>
    :root {
      /* Theme-aware fallbacks (will be overridden by Streamlit theme vars if present) */
      --bg: #0f1221;
      --bg-soft: #12162a;
      --text: #e6e8ef;
      --muted: #a9afc6;
      --primary: #7c83ff;
      --accent: #9a7cff;
      --success: #28c790;
      --warning: #ffb020;
      --danger: #ff5876;

      /* Neumorph base */
      --neu-bg: #12162a;
      --neu-shadow-light: #1e2342;
      --neu-shadow-dark: #0a0d1c;

      /* Radii and depth */
      --radius-s: 10px;
      --radius-m: 14px;
      --radius-l: 20px;
      --depth-1: 10px;
      --depth-2: 18px;

      /* Glass */
      --glass-bg: rgba(255,255,255,0.06);
      --glass-stroke: rgba(255,255,255,0.15);

      /* Gradients */
      --grad-1: linear-gradient(135deg,#7c83ff 0%,#9a7cff 50%,#c97cff 100%);
      --grad-2: linear-gradient(135deg,#2ee6a6 0%,#7cf0d1 100%);

      /* Typo */
      --heading-weight: 800;
      --subheading-weight: 700;
    }

    /* Respect Streamlit theme variables when available */
    :root, .stApp {
      --bg: var(--background-color, #0f1221);
      --text: var(--text-color, #e6e8ef);
      --primary: var(--primary-color, #7c83ff);
    }

    /* Page background */
    .stApp {
      background:
        radial-gradient(1200px 600px at 10% -10%, rgba(124,131,255,0.15), transparent 40%),
        radial-gradient(900px 500px at 110% 10%, rgba(154,124,255,0.16), transparent 45%),
        linear-gradient(180deg, #0e1120 0%, #0b0e1a 100%);
      color: var(--text);
    }

    /* Main pane glass container */
    [data-testid="stAppViewContainer"] > .main {
      background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
      backdrop-filter: blur(10px);
      border: 1px solid var(--glass-stroke);
      border-radius: var(--radius-l);
      padding: 1.25rem;
      margin: 1rem;
      box-shadow: 0 20px 60px rgba(0,0,0,0.35);
    }

    /* Headings */
    h1, h2, h3 {
      color: var(--text);
      letter-spacing: -0.02em;
    }
    h1 {
      font-weight: var(--heading-weight);
      background: var(--grad-1);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 0.25rem;
    }
    h2 {
      font-weight: var(--subheading-weight);
      border-bottom: 2px solid rgba(255,255,255,0.08);
      padding-bottom: 6px;
      margin-top: 1.5rem;
    }
    h3 { opacity: 0.95; }

    /* Cards (neumorphism + glass) */
    .card, [data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] {
      background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
      border: 1px solid var(--glass-stroke);
      border-radius: var(--radius-m);
      box-shadow:
        var(--depth-1) var(--depth-1) 30px var(--neu-shadow-dark),
        calc(var(--depth-1) * -1) calc(var(--depth-1) * -1) 30px rgba(255,255,255,0.02);
      transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease;
    }
    .card:hover, [data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"]:hover {
      transform: translateY(-2px);
      border-color: rgba(255,255,255,0.22);
      box-shadow:
        var(--depth-2) var(--depth-2) 50px var(--neu-shadow-dark),
        calc(var(--depth-2) * -1) calc(var(--depth-2) * -1) 50px rgba(255,255,255,0.03);
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
      font-size: 2.2rem !important;
      font-weight: 800 !important;
      background: var(--grad-2);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    [data-testid="stMetricLabel"] { color: var(--muted) !important; font-weight: 600 !important; }

    /* Buttons */
    .stButton > button {
      background: var(--grad-1);
      color: #0b0e1a;
      border: none;
      border-radius: 12px;
      padding: 0.7rem 1.2rem;
      font-weight: 700;
      letter-spacing: .2px;
      transition: transform .15s ease, box-shadow .2s ease, filter .15s ease;
      box-shadow: 0 10px 24px rgba(124,131,255,0.25);
    }
    .stButton > button:hover { transform: translateY(-1px); filter: brightness(1.05); }
    .stButton > button:active { transform: translateY(0); }

    /* Inputs */
    .stTextInput input, .stTextArea textarea {
      background: rgba(255,255,255,0.03) !important;
      border: 1px solid rgba(255,255,255,0.12) !important;
      color: var(--text) !important;
      border-radius: 12px !important;
      box-shadow: inset 4px 4px 8px rgba(0,0,0,0.35), inset -4px -4px 8px rgba(255,255,255,0.02);
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
      outline: none !important;
      border-color: var(--primary) !important;
      box-shadow: 0 0 0 3px rgba(124,131,255,0.18) !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
      gap: 10px;
      background: rgba(255,255,255,0.04);
      border: 1px solid var(--glass-stroke);
      border-radius: 12px;
      padding: .35rem;
    }
    .stTabs [data-baseweb="tab"] {
      color: var(--muted);
      border-radius: 10px;
      padding: .6rem 1rem;
      font-weight: 700;
      transition: background .2s ease, color .2s ease;
    }
    .stTabs [aria-selected="true"] {
      background: var(--grad-1);
      color: #0b0e1a !important;
    }

    /* Progress */
    .stProgress > div > div > div > div {
      background: var(--grad-2);
      border-radius: 10px;
      box-shadow: 0 6px 18px rgba(46,230,166,0.25);
    }

    /* Expander */
    .streamlit-expanderHeader {
      background: rgba(255,255,255,0.04);
      border-radius: 10px;
      color: var(--text);
      border: 1px solid var(--glass-stroke);
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
      background: rgba(255,255,255,0.03);
      border: 1.5px dashed rgba(255,255,255,0.18);
      border-radius: 14px;
      padding: 1.6rem;
    }
    [data-testid="stFileUploader"]:hover { border-color: var(--primary); }

    /* Dividers */
    hr {
      border: none; height: 1px;
      background: linear-gradient(90deg, transparent, rgba(255,255,255,0.22), transparent);
      margin: 1.25rem 0;
    }

    /* Alerts */
    .stAlert { border-radius: 12px; border: 1px solid var(--glass-stroke); }
    .stSuccess { background: rgba(40,199,144,0.08); }
    .stWarning { background: rgba(255,176,32,0.08); }
    .stError { background: rgba(255,88,118,0.08); }

    /* Rank card */
    .rank-card {
      background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.03));
      border: 1px solid var(--glass-stroke);
      border-left: 4px solid var(--primary);
      border-radius: 14px;
      padding: 1rem 1.25rem;
      transition: transform .2s ease, border-color .2s ease;
    }
    .rank-card:hover { transform: translateX(4px); border-left-color: #9a7cff; }

    /* Subtle appear animation */
    .fade-in { animation: fadeIn .35s ease-out; }
    @keyframes fadeIn { from {opacity:0; transform: translateY(8px);} to {opacity:1; transform:none;} }
    </style>
    """, unsafe_allow_html=True)

# --- Page Config (unchanged logic) ---
st.set_page_config(
    page_title="AI Powered CV Matching",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load CSS
load_custom_css()

# --- 1. FIREBASE ADMIN BAĞLANTISI ---
@st.cache_resource
def init_firebase_admin():
    try:
        creds_dict = dict(st.secrets["firebase_credentials"])
        creds_dict["private_key"] = creds_dict["private_key"].replace(r'\\n', '\\n')
        creds = credentials.Certificate(creds_dict)
        firebase_admin.initialize_app(creds)
    except ValueError:
        pass
    except Exception as e:
        st.error(f"🔥 FİREBASE ADMİN HATASI: {e}")
        st.stop()
    return firestore.client()

# --- 2. FIREBASE AUTH BAĞLANTISI ---
@st.cache_resource
def init_firebase_auth():
    try:
        firebase_config = {
            "apiKey": st.secrets["FIREBASE_WEB_API_KEY"],
            "authDomain": f"{st.secrets['firebase_credentials']['project_id']}.firebaseapp.com",
            "projectId": st.secrets['firebase_credentials']['project_id'],
            "storageBucket": f"{st.secrets['firebase_credentials']['project_id']}.appspot.com",
            "databaseURL": f"https://{st.secrets['firebase_credentials']['project_id']}-default-rtdb.firebaseio.com",
        }
        firebase = pyrebase.initialize_app(firebase_config)
        return firebase.auth()
    except Exception as e:
        st.error(f"🔥 FİREBASE AUTH HATASI: {e}")
        st.stop()

# --- 3. GEMINI AI BAĞLANTISI ---
@st.cache_resource
def init_gemini():
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        generation_config = genai.types.GenerationConfig(response_mime_type="application/json")
        analysis_model = genai.GenerativeModel('models/gemini-flash-latest', generation_config=generation_config)
        embedding_model = genai.GenerativeModel('models/text-embedding-004')
        return analysis_model, embedding_model
    except Exception as e:
        st.error(f"💎 GEMİNİ BAĞLATMA HATASI: {e}")
        st.stop()

# --- UYGULAMA BAŞLANGICI ---
try:
    db = init_firebase_admin()
    auth_client = init_firebase_auth()
    gemini_model, embedding_model = init_gemini()
except Exception as e:
    st.error("Uygulama başlatılırken kritik bir hata oluştu.")
    st.stop()

# --- SESSION ---
if 'user_email' not in st.session_state:
    st.session_state['user_email'] = None
if 'user_token' not in st.session_state:
    st.session_state['user_token'] = None

# --- HELPERS (unchanged functions) ---
@st.cache_data(ttl=300)
def get_platform_stats():
    try:
        job_docs = db.collection("job_postings").stream()
        total_jobs = sum(1 for _ in job_docs)
        profile_docs = db.collection("user_profiles").stream()
        total_profiles = sum(1 for _ in profile_docs)
        return total_jobs, total_profiles
    except Exception as e:
        return 0, 0

@st.cache_data(ttl=3600)
def get_total_user_count():
    try:
        page = auth.list_users()
        all_users = list(page.iterate_all())
        return len(all_users)
    except Exception as e:
        return 0

@st.cache_data(ttl=300)
def get_job_postings_with_vectors():
    jobs = []
    try:
        docs = db.collection("job_postings").stream()
        for doc in docs:
            job_data = doc.to_dict()
            if 'vector' in job_data:
                jobs.append({
                    "id": doc.id,
                    "title": job_data.get("title", "No Title"),
                    "description": job_data.get("description", "No Description"),
                    "vector": job_data.get("vector")
                })
        return jobs
    except Exception as e:
        st.error(f"İş ilanları çekilirken hata oluştu: {e}")
        return []

def get_gemini_analysis(cv, job_post):
    prompt = f"""
    You are a senior Human Resources (HR) specialist.
    Analyze the following CV and JOB POSTING.

    Your response MUST be a valid JSON object with the following exact structure:
    {{
        "score": <number from 0-100>,
        "pros": ["<strength 1>", "<strength 2>", "<strength 3>"],
        "cons": ["<weakness 1>", "<weakness 2>", "<weakness 3>"],
        "summary": "<A 2-3 sentence evaluation summary>"
    }}

    ---[CV TEXT]----
    {cv}
    -----------------

    ---[JOB POSTING TEXT]---
    {job_post}
    -----------------
    """
    try:
        response = gemini_model.generate_content(prompt)
        clean_json_text = re.sub(r"^```
        clean_json_text = re.sub(r"\\n```$", "", clean_json_text).strip()
        analysis_data = json.loads(clean_json_text)
        return analysis_data
    except Exception as e:
        print(f"JSON Parse Hatası: {e}")
        print(f"AI Ham Yanıtı: {response.text}")
        return None

def get_embedding(text):
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="RETRIEVAL_DOCUMENT"
        )
        return result['embedding']
    except Exception as e:
        st.error(f"Metnin 'parmak izi' alınırken hata oluştu: {e}")
        return None

def get_user_cv(user_id):
    try:
        doc_ref = db.collection("user_profiles").document(user_id).get()
        if doc_ref.exists:
            return doc_ref.to_dict().get("cv_text", "")
        return ""
    except Exception as e:
        st.error(f"Profilinizden CV'niz çekilirken hata oluştu: {e}")
        return ""

# --- MAIN APP (UI only restyled) ---
def main_app():
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.title("🤖 AI CV Matching Platform")
        st.caption("Find your next role with AI precision")
    with col2:
        right = f"<div style='text-align:right'><span style='padding:.4rem .8rem;border-radius:999px;background:var(--grad-1);color:#0b0e1a;font-weight:800;'>👤 {st.session_state['user_email'].split('@')[0]}</span></div>"
        st.markdown(right, unsafe_allow_html=True)
        if st.button("Logout", use_container_width=True):
            st.session_state['user_email'] = None
            st.session_state['user_token'] = None
            st.rerun()

    st.markdown("<div class='card' style='padding:0.5rem'></div>", unsafe_allow_html=True)  # subtle separator

    with st.spinner("Loading platform stats..."):
        total_jobs, total_profiles = get_platform_stats()
        total_users = get_total_user_count()

    a,b,c = st.columns(3)
    with a: st.metric("👥 Registered Users", total_users)
    with b: st.metric("🎯 Available Jobs", total_jobs)
    with c: st.metric("💼 Active Profiles", total_profiles)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    user_id = auth_client.get_account_info(st.session_state['user_token'])['users'][0]['localId']

    tab1, tab2, tab3 = st.tabs(["🚀 Auto-Matcher", "📝 Job Management", "👤 My Profile"])

    with tab1:
        st.header("Smart Matching")
        st.caption("Uses your saved CV from My Profile; or paste below")

        saved_cv = get_user_cv(user_id)
        with st.container(border=True):
            cv_text = st.text_area("📄 Your CV Text", value=saved_cv, height=320, placeholder="Paste your CV...")

        CANDIDATE_POOL_SIZE = 10
        TOP_N_RESULTS = 5

        if st.button(f"Find Top {TOP_N_RESULTS} Matches", type="primary", use_container_width=True):
            if cv_text:
                start_time = time.time()
                with st.spinner(f"Step 1/3: Vector screening top {CANDIDATE_POOL_SIZE}..."):
                    all_jobs = get_job_postings_with_vectors()
                    if not all_jobs:
                        st.warning("No job postings found. Please add jobs first.")
                        st.stop()

                    cv_vector = get_embedding(cv_text)
                    if not cv_vector:
                        st.error("Could not generate fingerprint for your CV. Aborting.")
                        st.stop()

                    job_vectors = np.array([job['vector'] for job in all_jobs])
                    cv_vector_np = np.array(cv_vector)
                    similarities = np.dot(job_vectors, cv_vector_np)

                    pool_size = min(len(all_jobs), CANDIDATE_POOL_SIZE)
                    top_candidate_indices = np.argsort(similarities)[-pool_size:][::-1]

                analysis_results = []
                progress_bar = st.progress(0, text=f"Step 2/3: Analyzing {pool_size} candidates... (0%)")

                with concurrent.futures.ThreadPoolExecutor(max_workers=pool_size) as executor:
                    future_to_job = {}
                    for index in top_candidate_indices:
                        matched_job = all_jobs[index]
                        future = executor.submit(get_gemini_analysis, cv_text, matched_job['description'])
                        future_to_job[future] = matched_job

                    completed_count = 0
                    for future in concurrent.futures.as_completed(future_to_job):
                        matched_job = future_to_job[future]
                        try:
                            analysis_data = future.result()
                            if analysis_data and analysis_data.get("score") is not None:
                                analysis_results.append({
                                    "job": matched_job,
                                    "data": analysis_data,
                                    "score": int(analysis_data.get("score", 0))
                                })
                        except Exception as e:
                            st.error(f"Error analyzing job '{matched_job['title']}': {e}")

                        completed_count += 1
                        percent_complete = completed_count / pool_size
                        progress_bar.progress(percent_complete, text=f"Step 2/3: Analyzing... {int(percent_complete * 100)}% complete")

                progress_bar.empty()

                with st.spinner(f"Step 3/3: Ranking Top {TOP_N_RESULTS}..."):
                    if not analysis_results:
                        st.error("AI analysis failed for all candidates. Please try again.")
                        st.stop()

                    sorted_results = sorted(analysis_results, key=lambda x: x["score"], reverse=True)
                    end_time = time.time()
                    st.success(f"Done! Found Top {TOP_N_RESULTS} in {end_time - start_time:.2f}s.")

                    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

                    for i, result in enumerate(sorted_results[:TOP_N_RESULTS]):
                        rank = i + 1
                        job_title = result["job"]["title"]
                        score = result["score"]
                        analysis_data = result["data"]

                        st.markdown("<div class='rank-card fade-in'>", unsafe_allow_html=True)
                        c1, c2 = st.columns([0.22, 0.78])
                        with c1:
                            st.metric(label=f"Rank #{rank}", value=f"{score}%")
                        with c2:
                            st.subheader(job_title)
                            with st.expander("View AI Analysis"):
                                st.subheader("Summary")
                                st.write(analysis_data.get("summary", "N/A"))
                                colp, colc = st.columns(2)
                                with colp:
                                    st.markdown("#### Strengths")
                                    pros = analysis_data.get("pros", [])
                                    if pros:
                                        for pro in pros: st.markdown(f"• {pro}")
                                    else:
                                        st.write("N/A")
                                with colc:
                                    st.markdown("#### Weaknesses")
                                    cons = analysis_data.get("cons", [])
                                    if cons:
                                        for con in cons: st.markdown(f"• {con}")
                                    else:
                                        st.write("N/A")
                        st.markdown("</div>", unsafe_allow_html=True)
                        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
            else:
                st.warning("Please paste your CV text to find matches.")

    with tab2:
        st.header("Job Management")
        with st.form("new_job_form", clear_on_submit=True):
            st.subheader("Add a Single Job Posting")
            job_title = st.text_input("Job Title", placeholder="e.g., Senior Backend Engineer")
            job_description = st.text_area("Job Description", height=200, placeholder="Responsibilities, requirements, nice-to-haves...")
            submitted = st.form_submit_button("Save Single Job & Generate Vector", type="primary")
            if submitted:
                if job_title and job_description:
                    with st.spinner("Generating AI fingerprint (vector)..."):
                        job_vector = get_embedding(f"Title: {job_title}\n\nDescription: {job_description}")
                    if job_vector:
                        try:
                            db.collection("job_postings").document().set({
                                "title": job_title,
                                "description": job_description,
                                "created_at": firestore.SERVER_TIMESTAMP,
                                "vector": job_vector,
                                "added_by": st.session_state['user_email']
                            })
                            st.success(f"Successfully added '{job_title}'!")
                            st.cache_data.clear()
                        except Exception as e:
                            st.error(f"Error saving to Firebase: {e}")
                    else:
                        st.error("Could not generate AI fingerprint.")
                else:
                    st.warning("Please fill in both fields.")

        st.divider()

        st.subheader("Bulk Upload Jobs from CSV/Excel")
        st.caption("Upload a file with 'title' and 'description' columns.")
        uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)

                if 'title' not in df.columns or 'description' not in df.columns:
                    st.error("Error: File must contain 'title' and 'description' columns.")
                else:
                    st.success(f"File '{uploaded_file.name}' read successfully. Found {len(df)} jobs.")
                    st.dataframe(df.head(), use_container_width=True)
                    if st.button(f"Process and Upload {len(df)} Jobs", type="primary"):
                        st.info("Starting bulk upload... This may take several minutes.")
                        progress_bar_bulk = st.progress(0, text="Starting...")
                        success_count = 0
                        batch = db.batch()
                        for index, row in df.iterrows():
                            title = str(row['title'])
                            description = str(row['description'])
                            progress_text = f"Processing ({index + 1}/{len(df)}): {title[:30]}..."
                            progress_bar_bulk.progress((index + 1) / len(df), text=progress_text)
                            job_vector = get_embedding(f"Title: {title}\n\nDescription: {description}")
                            if job_vector:
                                doc_ref = db.collection("job_postings").document()
                                batch.set(doc_ref, {
                                    "title": title,
                                    "description": description,
                                    "created_at": firestore.SERVER_TIMESTAMP,
                                    "vector": job_vector,
                                    "added_by": f"bulk_upload_{st.session_state['user_email']}"
                                })
                                success_count += 1
                        batch.commit()
                        st.success(f"Done! Successfully processed and uploaded {success_count} out of {len(df)} jobs.")
                        st.cache_data.clear()
            except Exception as e:
                st.error(f"An error occurred while processing the file: {e}")

    with tab3:
        st.header("My Profile")
        st.caption("Save your CV here so you don’t paste it every time.")
        current_cv = get_user_cv(user_id)
        with st.form("profile_form"):
            new_cv_text = st.text_area("Your CV Text", value=current_cv, height=380, placeholder="Paste your complete CV...")
            submitted = st.form_submit_button("Save CV to Profile", type="primary")
            if submitted:
                try:
                    with st.spinner("Generating AI fingerprint for your CV..."):
                        cv_vector = get_embedding(new_cv_text)
                    if cv_vector:
                        db.collection("user_profiles").document(user_id).set({
                            "email": st.session_state['user_email'],
                            "cv_text": new_cv_text,
                            "cv_vector": cv_vector,
                            "updated_at": firestore.SERVER_TIMESTAMP
                        }, merge=True)
                        st.success("Your CV has been successfully saved to your profile!")
                    else:
                        st.error("Could not generate AI fingerprint for your CV. Not saved.")
                except Exception as e:
                    st.error(f"An error occurred while saving your profile: {e}")

# --- LOGIN PAGE ---
def login_page():
    st.markdown("<div style='text-align:center'>", unsafe_allow_html=True)
    st.title("🤖 AI CV Matching Platform")
    st.caption("Your next career move, powered by AI")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card' style='padding:.5rem'></div>", unsafe_allow_html=True)

    with st.spinner("Loading platform stats..."):
        total_jobs, total_profiles = get_platform_stats()
        total_users = get_total_user_count()

    c1,c2,c3 = st.columns(3)
    with c1: st.metric("👥 Registered Users", total_users)
    with c2: st.metric("🎯 Available Jobs", total_jobs)
    with c3: st.metric("💼 Active Profiles", total_profiles)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    login_tab, signup_tab = st.tabs(["Login", "Sign Up"])

    with login_tab:
        st.subheader("Welcome Back")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Login", type="primary"):
            if email and password:
                try:
                    user = auth_client.sign_in_with_email_and_password(email, password)
                    st.session_state['user_email'] = user['email']
                    st.session_state['user_token'] = user['idToken']
                    st.rerun()
                except Exception as e:
                    st.warning("Login failed. Please check your email and password.")
            else:
                st.warning("Please enter both email and password.")

    with signup_tab:
        st.subheader("Create a New Account")
        new_email = st.text_input("Email", key="signup_email")
        new_password = st.text_input("Password", type="password", key="signup_pass")
        if st.button("Sign Up", type="primary"):
            if new_email and new_password:
                try:
                    user = auth_client.create_user_with_email_and_password(new_email, new_password)
                    st.success("Account created successfully! Please go to the 'Login' tab to log in.")
                except Exception as e:
                    error_message = str(e)
                    if "WEAK_PASSWORD" in error_message:
                        st.warning("Password should be at least 6 characters.")
                    elif "EMAIL_EXISTS" in error_message:
                        st.warning("An account with this email already exists. Please log in.")
                    elif "INVALID_EMAIL" in error_message:
                        st.warning("Please enter a valid email address.")
                    else:
                        st.error("An unknown error occurred during sign up.")
            else:
                st.warning("Please enter both email and password.")

# --- ROUTER ---
if st.session_state['user_email']:
    main_app()
else:
    login_page()
