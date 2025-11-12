import streamlit as st
import google.generativeai as genai
import firebase_admin
from firebase_admin import credentials, firestore, auth
import json
import numpy as np
import re
import pyrebase 
import time
import concurrent.futures # (YENİ) Paralel API çağrıları için

# --- Sayfa Ayarları ---
st.set_page_config(
    page_title="AI Powered CV Matching",
    page_icon="🤖",
    layout="wide"
)

# --- 1. FIREBASE ADMIN BAĞLANTISI ---
@st.cache_resource
def init_firebase_admin():
    try:
        creds_dict = dict(st.secrets["firebase_credentials"])
        creds_dict["private_key"] = creds_dict["private_key"].replace(r'\n', '\n')
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

# --- OTURUM YÖNETİMİ ---
if 'user_email' not in st.session_state:
    st.session_state['user_email'] = None
if 'user_token' not in st.session_state:
    st.session_state['user_token'] = None

# --- YARDIMCI FONKSİYONLAR ---

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
        clean_json_text = re.sub(r"^```json\n", "", response.text)
        clean_json_text = re.sub(r"\n```$", "", clean_json_text).strip()
        analysis_data = json.loads(clean_json_text)
        return analysis_data
    except Exception as e:
        # Hata durumunda, AI'ın ne döndüğünü görmek için loglayabiliriz
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

# --- ANA UYGULAMA FONKSİYONU ---
def main_app():
    
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.title("🤖 AI CV Matching Platform")
    with col2:
        st.write(f"Logged in as: `{st.session_state['user_email']}`")
        if st.button("Logout", use_container_width=True):
            st.session_state['user_email'] = None
            st.session_state['user_token'] = None
            st.rerun() 
            
    st.markdown("---") 

    with st.spinner("Loading platform stats..."):
        total_jobs, total_profiles = get_platform_stats()
        total_users = get_total_user_count()
    
    stat_col1, stat_col2, stat_col3 = st.columns(3)
    with stat_col1:
        st.metric(label="👥 Total Registered Users", value=total_users)
    with stat_col2:
        st.metric(label="🎯 Total Jobs in Pool", value=total_jobs)
    with stat_col3:
        st.metric(label="👤 Saved CV Profiles", value=total_profiles, help="Number of users who have saved their CV.")

    st.markdown("---")
    
    user_id = auth_client.get_account_info(st.session_state['user_token'])['users'][0]['localId']

    tab1, tab2, tab3 = st.tabs(["🚀 Auto-Matcher", "📝 Job Management", "👤 My Profile"])

    # --- (GÜNCELLENDİ) Sekme 1: Auto-Matcher (Hızlandırıldı) ---
  # --- Sekme 1: Auto-Matcher ---
    with tab1:
        st.header("Find the Best Jobs for Your CV")
        st.markdown("We will use the CV saved in your 'My Profile' tab. If it's empty, please paste your CV below.")
        
        saved_cv = get_user_cv(user_id)
        
        with st.container(border=True):
            cv_text = st.text_area("📄 Your CV Text:", value=saved_cv, height=350)
        
        CANDIDATE_POOL_SIZE = 10 
        TOP_N_RESULTS = 5       
        
        if st.button(f"Find My Top {TOP_N_RESULTS} Matches", type="primary", use_container_width=True):
            if cv_text:
                start_time = time.time() 
                
                # --- Adım 1: Hızlı Filtreleme (Vektör Arama) ---
                with st.spinner(f"Step 1/3: Searching all jobs for the top {CANDIDATE_POOL_SIZE} candidates..."):
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

                # --- Adım 2: Paralel Analiz (Hızlı) ---
                analysis_results = []
                # (GÜNCELLENDİ) İlerleme çubuğunun başlangıç metni
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
                        # (GÜNCELLENDİ) İlerleme çubuğunun güncelleme metni (yüzde gösterir)
                        progress_bar.progress(percent_complete, text=f"Step 2/3: Analyzing... {int(percent_complete * 100)}% complete") 
                
                progress_bar.empty()

                # --- Adım 3: Yeniden Sırala ve Göster ---
                with st.spinner(f"Step 3/3: Ranking results and showing the Top {TOP_N_RESULTS}..."):
                    if not analysis_results:
                        st.error("AI analysis failed for all candidates. Please try again.")
                        st.stop()

                    sorted_results = sorted(analysis_results, key=lambda x: x["score"], reverse=True)
                    
                    end_time = time.time()
                    st.success(f"Done! Found and ranked your Top {TOP_N_RESULTS} matches in {end_time - start_time:.2f} seconds.")
                    
                    # (YENİ EKLEME) İşte animasyon burada!
                    st.balloons() 
                    
                    st.markdown("---")

                    for i, result in enumerate(sorted_results[:TOP_N_RESULTS]):
                        rank = i + 1
                        job_title = result["job"]["title"]
                        score = result["score"]
                        analysis_data = result["data"]
                        
                        with st.container(border=True):
                            col_metric, col_details = st.columns([0.2, 0.8])
                            with col_metric:
                                st.metric(label=f"Rank #{rank} Match", value=f"{score}%")
                            with col_details:
                                st.subheader(job_title)
                                with st.expander("Click to see detailed AI analysis"):
                                    st.subheader("Summary")
                                    st.write(analysis_data.get("summary", "N/A"))
                                    st.subheader("Strengths (Pros)")
                                    # (GÜNCELLENDİ) Eğer veri yoksa "N/A" göstermek için
                                    pros = analysis_data.get("pros", [])
                                    if pros:
                                        for pro in pros: st.markdown(f"* {pro}")
                                    else:
                                        st.write("N/A") 
                                    st.subheader("Weaknesses (Cons)")
                                    # (GÜNCELLENDİ) Eğer veri yoksa "N/A" göstermek için
                                    cons = analysis_data.get("cons", [])
                                    if cons:
                                        for con in cons: st.markdown(f"* {con}")
                                    else:
                                        st.write("N/A")
                        st.divider()
            else:
                st.warning("Please paste your CV text to find matches.")
                
    # --- Sekme 2: İlan Yönetimi (Toplu Yükleme dahil) ---
    with tab2:
        st.header("Job Management")
        
        # Tekli ilan formu
        with st.form("new_job_form", clear_on_submit=True):
            st.subheader("Add a Single Job Posting")
            job_title = st.text_input("Job Title")
            job_description = st.text_area("Job Description", height=200)
            submitted = st.form_submit_button("Save Single Job & Generate Vector")
            
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
                        except Exception as e: st.error(f"Error saving to Firebase: {e}")
                    else: st.error("Could not generate AI fingerprint.")
                else: st.warning("Please fill in both fields.")

        st.divider()
        
        # Toplu ilan yükleme
        st.subheader("OR... Bulk Upload Jobs from CSV/Excel")
        st.markdown("Upload a file with **'title'** and **'description'** columns.")
        
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
                    st.dataframe(df.head())
                    
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


    # --- Sekme 3: Profilim ---
    with tab3:
        st.header("My Profile")
        st.markdown("Save your CV here so you don't have to paste it every time.")
        
        current_cv = get_user_cv(user_id)
        
        with st.form("profile_form"):
            new_cv_text = st.text_area("Your CV Text", value=current_cv, height=400)
            submitted = st.form_submit_button("Save CV to Profile")
            
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

# --- LOGIN SAYFASI FONKSİYONU ---
def login_page():
    st.title("🤖 AI CV Matching Platform")
    
    st.markdown("Welcome! Log in or sign up to find your perfect job match.")
    st.markdown("---")

    with st.spinner("Loading platform stats..."):
        total_jobs, total_profiles = get_platform_stats()
        total_users = get_total_user_count()
    
    stat_col1, stat_col2, stat_col3 = st.columns(3)
    with stat_col1:
        st.metric(label="👥 Total Registered Users", value=total_users)
    with stat_col2:
        st.metric(label="🎯 Total Jobs in Pool", value=total_jobs)
    with stat_col3:
        st.metric(label="👤 Saved CV Profiles", value=total_profiles)

    st.markdown("---")
    
    login_tab, signup_tab = st.tabs(["Login", "Sign Up"])
    
    with login_tab:
        st.subheader("Login")
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_pass")
        
        if st.button("Login", type="primary", key="login_button"):
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
        
        if st.button("Sign Up", type="primary", key="signup_button"):
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

# --- ANA MANTIK ---
if st.session_state['user_email']:
    main_app()
else:
    login_page()
