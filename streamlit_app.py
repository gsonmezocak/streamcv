import os
import re
import json
import time
import concurrent.futures
import io

# Flask ve Backend Kütüphaneleri
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from dotenv import load_dotenv # API anahtarlarını .env'den yüklemek için

# Sizin Orijinal Kütüphaneleriniz
import google.generativeai as genai
import firebase_admin
from firebase_admin import credentials, firestore, auth
import pyrebase
import numpy as np
import fitz  # PyMuPDF
from docx import Document

# --- 0. UYGULAMA VE KONFİGÜRASYON ---
load_dotenv() # .env dosyasındaki değişkenleri yükler

app = Flask(__name__)
# Flask 'session' (oturum) için gizli bir anahtar gereklidir
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "cok-gizli-bir-anahtar-yaz")

# --- 1. FIREBASE ADMIN BAĞLANTISI ---
try:
    # NOT: Firebase credentials'ı artık st.secrets'ten değil,
    # bir JSON dosyasından veya ortam değişkenlerinden almalısınız.
    # Bu örnekte, 'firebase-credentials.json' adında bir dosyanız olduğunu varsayıyorum.
    creds_path = os.environ.get("FIREBASE_CREDS_PATH", "firebase-credentials.json")
    creds = credentials.Certificate(creds_path)
    firebase_admin.initialize_app(creds)
except ValueError:
    pass # Zaten başlatılmışsa
db = firestore.client()

# --- 2. FIREBASE AUTH BAĞLANTISI ---
firebase_config = {
    "apiKey": os.environ.get("FIREBASE_WEB_API_KEY"),
    "authDomain": f"{os.environ.get('FIREBASE_PROJECT_ID')}.firebaseapp.com",
    "projectId": os.environ.get('FIREBASE_PROJECT_ID'),
    "storageBucket": f"{os.environ.get('FIREBASE_PROJECT_ID')}.appspot.com",
    "databaseURL": f"https://{os.environ.get('FIREBASE_PROJECT_ID')}-default-rtdb.firebaseio.com",
}
firebase = pyrebase.initialize_app(firebase_config)
auth_client = firebase.auth()

# --- 3. GEMINI AI BAĞLANTISI ---
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
generation_config = genai.types.GenerationConfig(response_mime_type="application/json")
gemini_model = genai.GenerativeModel('models/gemini-flash-latest', generation_config=generation_config)
embedding_model = genai.GenerativeModel('models/text-embedding-004')

# --- 4. YARDIMCI FONKSİYONLAR (Streamlit'ten bağımsız) ---

def get_gemini_analysis(cv, job_post):
    prompt = f"""
    You are a senior Human Resources (HR) specialist.
    Analyze the following CV and JOB POSTING.
    Your response MUST be a valid JSON object... (Prompt'unuzun geri kalanı)
    ---[CV TEXT]----
    {cv}
    ---[JOB POSTING TEXT]---
    {job_post}
    """
    try:
        response = gemini_model.generate_content(prompt)
        clean_json_text = re.sub(r"^```json\n", "", response.text)
        clean_json_text = re.sub(r"\n```$", "", clean_json_text).strip()
        return json.loads(clean_json_text)
    except Exception as e:
        print(f"JSON Parse Hatası: {e}")
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
        print(f"Embedding Hatası: {e}")
        return None

def get_user_profile(user_id):
    try:
        doc_ref = db.collection("user_profiles").document(user_id).get()
        if doc_ref.exists:
            return doc_ref.to_dict()
        return {}
    except Exception as e:
        print(f"Profil çekme hatası: {e}")
        return {}

def parse_cv_file(file_bytes, file_name):
    text = ""
    try:
        if file_name.endswith('.pdf'):
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                for page in doc: text += page.get_text()
        elif file_name.endswith('.docx'):
            doc = Document(io.BytesIO(file_bytes))
            for para in doc.paragraphs: text += para.text + "\n"
        else:
            text = file_bytes.decode('utf-8')
        return text
    except Exception as e:
        print(f"Dosya okuma hatası: {e}")
        return None

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
        print(f"İlanları çekme hatası: {e}")
        return []

# --- 5. FRONTEND SAYFA YOLLARI (HTML Sunma) ---

@app.route("/")
def index():
    """Kullanıcıyı sign_up.html'e yönlendirir."""
    return render_template("sign_up.html")

@app.route("/profile")
def profile_page():
    """Kullanıcı giriş yaptıysa code.html (profil) sayfasını gösterir."""
    if 'user_token' not in session:
        return redirect(url_for("index")) # Giriş yapmadıysa ana sayfaya at
    return render_template("code.html")

# --- 6. AUTH API ENDPOINT'LERİ (JavaScript'in çağıracağı) ---

@app.route("/api/signup", methods=["POST"])
def api_signup():
    data = request.json
    email = data.get("email")
    password = data.get("password")
    
    if not email or not password:
        return jsonify({"success": False, "error": "Email ve şifre gerekli"}), 400
        
    try:
        user = auth_client.create_user_with_email_and_password(email, password)
        return jsonify({"success": True, "message": "Kayıt başarılı!"})
    except Exception as e:
        error_json = e.args[1]
        error_message = json.loads(error_json).get("error", {}).get("message", "Bilinmeyen hata")
        return jsonify({"success": False, "error": error_message}), 400

@app.route("/api/login", methods=["POST"])
def api_login():
    data = request.json
    email = data.get("email")
    password = data.get("password")
    
    try:
        user = auth_client.sign_in_with_email_and_password(email, password)
        # Kullanıcı bilgilerini sunucu tarafında session'a kaydet
        session['user_token'] = user['idToken']
        session['user_id'] = user['localId']
        session['user_email'] = user['email']
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": "Geçersiz email veya şifre"}), 401

@app.route("/api/logout")
def api_logout():
    session.clear() # Oturumu temizle
    return redirect(url_for("index"))

# --- 7. ANA API ENDPOINT'LERİ (JavaScript'in çağıracağı) ---

@app.route("/api/profile", methods=["GET"])
def api_get_profile():
    """Kullanıcının mevcut profil bilgilerini döndürür."""
    if 'user_id' not in session:
        return jsonify({"error": "Yetkisiz"}), 401
    
    user_id = session['user_id']
    profile_data = get_user_profile(user_id)
    return jsonify(profile_data)

@app.route("/api/profile", methods=["POST"])
def api_update_profile():
    """Kullanıcının profilini (form ve CV dosyası ile) günceller."""
    if 'user_id' not in session:
        return jsonify({"error": "Yetkisiz"}), 401
        
    try:
        user_id = session['user_id']
        full_name = request.form.get("full-name")
        headline = request.form.get("headline")
        cv_text_from_area = request.form.get("experience-skills")
        cv_file = request.files.get("cv-file") # 'cv-file' HTML'deki input'un 'name'i olmalı

        final_cv_text = ""

        if cv_file and cv_file.filename != "":
            file_bytes = cv_file.read()
            final_cv_text = parse_cv_file(file_bytes, cv_file.filename)
            if final_cv_text is None:
                return jsonify({"success": False, "error": "CV dosyası okunamadı"}), 400
        else:
            final_cv_text = cv_text_from_area

        if not final_cv_text:
            return jsonify({"success": False, "error": "CV metni veya dosyası gerekli"}), 400

        cv_vector = get_embedding(final_cv_text)
        if cv_vector is None:
            return jsonify({"success": False, "error": "CV vektörü oluşturulamadı"}), 500

        # Veriyi Firestore'a kaydet
        profile_data = {
            "email": session['user_email'],
            "full_name": full_name,
            "headline": headline,
            "cv_text": final_cv_text,
            "cv_vector": cv_vector,
            "updated_at": firestore.SERVER_TIMESTAMP
        }
        db.collection("user_profiles").document(user_id).set(profile_data, merge=True)
        
        return jsonify({"success": True, "message": "Profil güncellendi!"})

    except Exception as e:
        print(f"Profil güncelleme hatası: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/auto-match")
def api_auto_match():
    """Kullanıcının kayıtlı CV'si ile iş ilanlarını eşleştirir."""
    if 'user_id' not in session:
        return jsonify({"error": "Yetkisiz"}), 401
        
    user_id = session['user_id']
    profile = get_user_profile(user_id)
    cv_text = profile.get("cv_text")
    cv_vector = profile.get("cv_vector")

    if not cv_text or not cv_vector:
        return jsonify({"error": "Lütfen önce profilinize bir CV ekleyin"}), 400

    all_jobs = get_job_postings_with_vectors()
    if not all_jobs:
        return jsonify({"error": "Sistemde hiç iş ilanı bulunamadı"}), 404
        
    # Adım 1: Vektör Benzerlik Araması
    job_vectors = np.array([job['vector'] for job in all_jobs])
    cv_vector_np = np.array(cv_vector)
    similarities = np.dot(job_vectors, cv_vector_np)
    
    CANDIDATE_POOL_SIZE = 10
    pool_size = min(len(all_jobs), CANDIDATE_POOL_SIZE)
    top_candidate_indices = np.argsort(similarities)[-pool_size:][::-1]

    # Adım 2: Paralel Gemini Analizi
    analysis_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=pool_size) as executor:
        future_to_job = {}
        for index in top_candidate_indices:
            matched_job = all_jobs[index]
            future = executor.submit(get_gemini_analysis, cv_text, matched_job['description'])
            future_to_job[future] = matched_job
        
        for future in concurrent.futures.as_completed(future_to_job):
            matched_job = future_to_job[future]
            analysis_data = future.result()
            if analysis_data and analysis_data.get("score") is not None:
                analysis_results.append({
                    "job": matched_job,
                    "data": analysis_data,
                    "score": int(analysis_data.get("score", 0))
                })

    # Adım 3: Sırala ve Döndür
    sorted_results = sorted(analysis_results, key=lambda x: x["score"], reverse=True)
    return jsonify(sorted_results[:5]) # Top 5 sonucu döndür

# --- 8. SUNUCUYU BAŞLAT ---
if __name__ == "__main__":
    app.run(debug=True, port=5000)
