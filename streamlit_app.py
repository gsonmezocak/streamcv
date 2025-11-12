import streamlit as st
import google.generativeai as genai
import firebase_admin
from firebase_admin import credentials, firestore
import json

# --- Sayfa Ayarları ---
st.set_page_config(
    page_title="AI Powered CV Matching",
    page_icon="🤖",
    layout="wide"
)

# --- 1. FIREBASE BAĞLANTISI (YENİ VE DOĞRU YÖNTEM) ---
@st.cache_resource
def init_firebase():
    """
    Streamlit Secrets'tan alınan TABLO verisiyle Firebase'i başlatır.
    Bu yöntem, JSON hatalarını %100 engeller.
    """
    try:
        # 1. Adım: Secret'ı bir SÖZLÜK (dict) olarak al
        # st.secrets["firebase_credentials"] bir sözlük döndürecek
        # çünkü Secrets bölümünde [firebase_credentials] kullandık.
        creds_dict = dict(st.secrets["firebase_credentials"])
        
        # 2. Adım: private_key'deki \n'leri düzelt (TOML'un bir cilvesi)
        creds_dict["private_key"] = creds_dict["private_key"].replace(r'\n', '\n')
        
        # 3. Adım: Sözlüğü kimlik bilgisi olarak Firebase'e ver
        creds = credentials.Certificate(creds_dict)
        
        # 4. Adım: Uygulamayı başlat
        firebase_admin.initialize_app(creds)
        
    except ValueError as e:
        # Uygulama zaten başlatılmışsa (bu bir hata değil, normal)
        if "The default Firebase app already exists" in str(e):
            pass # Görmezden gel, devam et
        else:
            # Ama başka bir Değer Hatasıysa göster
            st.error(f"🔥 FİREBASE DEĞER HATASI: {e}")
            st.error("Secrets bölümündeki 'firebase_credentials' tablonuzu kontrol edin.")
            st.stop()
            
    except Exception as e:
        # Diğer tüm hatalar
        st.error(f"🔥 FİREBASE GENEL HATA: {e}")
        st.error("Secrets bölümünüzü ve kodunuzu kontrol edin.")
        st.stop()
        
    # Her şey yolunda gittiyse, veritabanı istemcisini döndür
    return firestore.client()

# --- 2. GEMINI AI BAĞLANTISI ---
@st.cache_resource
def init_gemini():
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel('models/gemini-flash-latest')
        return model
    except Exception as e:
        st.error(f"💎 GEMİNİ BAŞLATMA HATASI: {e}")
        st.error("Lütfen Streamlit Secrets'taki 'GEMINI_API_KEY' anahtarınızı kontrol edin.")
        st.stop()

# --- UYGULAMA BAŞLANGICI ---
st.title("🤖 AI CV Matching Platform (v1 - Firebase)")

# Servisleri başlat
db = init_firebase()
gemini_model = init_gemini()


# --- YARDIMCI FONKSİYONLAR ---

@st.cache_data(ttl=300) 
def get_job_postings():
    jobs = []
    try:
        docs = db.collection("job_postings").stream()
        for doc in docs:
            job_data = doc.to_dict()
            jobs.append({
                "id": doc.id,
                "title": job_data.get("title", "No Title"),
                "description": job_data.get("description", "No Description")
            })
        return jobs
    except Exception as e:
        st.error(f"İş ilanları çekilirken hata oluştu: {e}")
        return []

def get_gemini_analysis(cv, job_post):
    prompt = f"""
    You are a senior Human Resources (HR) specialist, and your task is to compare a CV with a job posting.
    Analyze the following CV text and JOB POSTING text in detail.
    Follow these steps in your analysis:
    1.  **Overall Compatibility Score:** Rate the CV's suitability for the job posting on a scale of 100.
    2.  **Strengths (Pros):** List the top 3-4 strengths of the candidate that meet the job requirements.
    3.  **Weaknesses / Gaps (Cons):** List 3-4 key points mentioned in the job posting that are missing or weak in the CV.
    4.  **Evaluation Summary:** Write a brief 2-3 sentence overall evaluation summary.
    Please provide your answer in **Markdown format** using clear headings.
    ---[CV TEXT]----
    {cv}
    -----------------
    ---[JOB POSTING TEXT]---
    {job_post}
    -----------------
    """
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred during analysis: {e}"

# --- ANA UYGULAMA ARAYÜZÜ ---

tab1, tab2 = st.tabs(["🚀 CV Matcher", "📝 Add New Job Posting"])

# (Kodun geri kalanı aynı, buraya eklemiyorum... Ah, hayır, kodun TAMAMINI istedi, o yüzden devam etmeliyim)

# --- Sekme 1: CV EŞLEŞTİRİCİ ---
with tab1:
    st.header("Match Your CV Against Our Job Postings")
    
    job_list = get_job_postings()
    
    if not job_list:
        st.info("No job postings found in the database. Please add a job in the 'Add New Job Posting' tab.")
    else:
        job_dict = {job["title"]: job["description"] for job in job_list}
        
        selected_title = st.selectbox(
            "Select a job posting from the database:",
            options=job_dict.keys()
        )
        
        if selected_title:
            selected_description = job_dict[selected_title]
            
            col1, col2 = st.columns(2)
            with col1:
                with st.container(border=True):
                    st.subheader("📄 Paste CV Text Below")
                    cv_text = st.text_area("CV Text", height=350, label_visibility="collapsed", key="cv_text_tab1")
            
            with col2:
                with st.container(border=True):
                    st.subheader(f"🎯 Selected Job: {selected_title}")
                    st.text_area("Job Posting Description", value=selected_description, height=350, disabled=True, label_visibility="collapsed")
            
            if st.button("Run Compatibility Analysis", type="primary", use_container_width=True, key="analyze_button"):
                if cv_text:
                    with st.spinner("Gemini AI is analyzing... Please wait."):
                        analysis_result = get_gemini_analysis(cv_text, selected_description)
                        with st.expander("✨ Click to See Analysis Result", expanded=True):
                            st.markdown(analysis_result)
                else:
                    st.warning("Please paste your CV text to analyze.")

# --- Sekme 2: YENİ İLAN EKLEME ---
with tab2:
    st.header("Add a New Job Posting to the Database")
    
    with st.form("new_job_form", clear_on_submit=True):
        job_title = st.text_input("Job Title")
        job_description = st.text_area("Job Description (Paste the full text)", height=300)
        
        submitted = st.form_submit_button("Save Job Posting to Firebase")
        
        if submitted:
            if job_title and job_description:
                try:
                    doc_ref = db.collection("job_postings").document()
                    doc_ref.set({
                        "title": job_title,
                        "description": job_description,
                        "created_at": firestore.SERVER_TIMESTAMP
                    })
                    st.success(f"Successfully added job posting: '{job_title}'")
                    st.cache_data.clear()
                except Exception as e:
                    st.error(f"An error occurred while saving to Firebase: {e}")
            else:
                st.warning("Please fill in both the Job Title and Job Description.")
