import streamlit as st
import google.generativeai as genai
import firebase_admin
from firebase_admin import credentials, firestore
import json # JSON kütüphanesini içe aktardığımızdan emin olalım

# --- Sayfa Ayarları ---
st.set_page_config(
    page_title="AI Powered CV Matching",
    page_icon="🤖",
    layout="wide"
)

# --- 1. FIREBASE BAĞLANTISI (SAĞLAMLAŞTIRILDI) ---
@st.cache_resource
def init_firebase():
    """
    Streamlit Secrets'tan alınan kimlik bilgileriyle Firebase'i başlatır.
    """
    try:
        # 1. Adım: Secret'ı METİN (string) olarak al
        creds_json_str = st.secrets["FIREBASE_CREDENTIALS"]
        
        # 2. Adım: Bu metni bir Python SÖZLÜĞÜNE (dict) çevir
        creds_dict = json.loads(creds_json_str)
        
        # 3. Adım: Sözlüğü kimlik bilgisi olarak Firebase'e ver
        creds = credentials.Certificate(creds_dict)
        
        # 4. Adım: Uygulamayı başlat
        firebase_admin.initialize_app(creds)
        
    except json.JSONDecodeError as e:
        # Hata 1: JSON metni bozuksa
        st.error(f"🔥 FİREBASE JSON HATASI: {e}")
        st.error("FIREBASE_CREDENTIALS secret'ınızın içeriği geçerli bir JSON değil.")
        st.error("Lütfen Streamlit Secrets bölümündeki üçlü tırnakların (''') İÇİNDEKİ metnin tam bir JSON olduğundan emin olun.")
        st.stop()
        
    except ValueError as e:
        # Hata 2: Uygulama zaten başlatılmışsa (bu bir hata değil, normal)
        if "The default Firebase app already exists" in str(e):
            pass # Görmezden gel, devam et
        else:
            # Ama başka bir Değer Hatasıysa (örn: creds geçersizse) göster
            st.error(f"🔥 FİREBASE DEĞER HATASI: {e}")
            st.stop()
            
    except Exception as e:
        # Hata 3: Diğer tüm hatalar
        st.error(f"🔥 FİREBASE GENEL HATA: {e}")
        st.stop()
        
    # Her şey yolunda gittiyse, veritabanı istemcisini döndür
    return firestore.client()

# --- 2. GEMINI AI BAĞLANTISI ---
@st.cache_resource
def init_gemini():
    """
    Streamlit Secrets'tan alınan API anahtarıyla Gemini'yi başlatır.
    """
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

@st.cache_data(ttl=300) # Veritabanı sorgusunu 5 dakika önbelleğe al
def get_job_postings():
    """
    Firestore'dan tüm iş ilanlarını çeker.
    """
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
    """
    Gemini'ye analiz prompt'unu gönderir.
    """
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

# --- Sekme 1: CV EŞLEŞTİRİCİ ---
with tab1:
    st.header("Match Your CV Against Our Job Postings")
    
    # Firebase'den ilanları çek
    job_list = get_job_postings()
    
    if not job_list:
        st.info("No job postings found in the database. Please add a job in the 'Add New Job Posting' tab.")
    else:
        # İlanları bir sözlükte (dict) sakla: {Başlık: Açıklama}
        job_dict = {job["title"]: job["description"] for job in job_list}
        
        # Kullanıcıya seçtir
        selected_title = st.selectbox(
            "Select a job posting from the database:",
            options=job_dict.keys()
        )
        
        if selected_title:
            selected_description = job_dict[selected_title]
            
            # Seçilen ilanı ve CV giriş alanını göster
            col1, col2 = st.columns(2)
            with col1:
                with st.container(border=True):
                    st.subheader("📄 Paste CV Text Below")
                    cv_text = st.text_area("CV Text", height=350, label_visibility="collapsed", key="cv_text_tab1")
            
            with col2:
                with st.container(border=True):
                    st.subheader(f"🎯 Selected Job: {selected_title}")
                    st.text_area("Job Posting Description", value=selected_description, height=350, disabled=True, label_visibility="collapsed")
            
            # Analiz butonu
            if st.button("Run Compatibility Analysis", type="primary", use_container_width=True, key="analyze_button"):
                if cv_text:
                    with st.spinner("Gemini AI is analyzing... Please wait."):
                        analysis_result = get_gemini_analysis(cv_text, selected_description)
                        with st.expander("✨ Click to See Analysis Result", expanded=True):
                            st.markdown(analysis_result)
                else:
                    st.warning("Please paste your CV text to analyze.")

# --- Sekme 2: Y
