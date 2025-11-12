import streamlit as st
import google.generativeai as genai
import time

# --- Sayfa Ayarları ---
st.set_page_config(
    page_title="AI Destekli CV Eşleştirme",
    page_icon="🤖"
)

# --- Başlık ve Açıklama ---
st.title("🤖 AI Destekli CV - İş İlanı Eşleştirme (MVP)")
st.markdown("Bu uygulama, Gemini AI kullanarak bir CV metni ile bir iş ilanı metni arasındaki uyumu analiz eder.")

# --- API Anahtarını Güvenli Yerden Alma ---
# Streamlit'in secrets özelliğini kullanarak anahtarı güvenle çekiyoruz.
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except Exception as e:
    st.error("API Anahtarı bulunamadı veya geçersiz. Lütfen secrets.toml dosyanızı kontrol edin.")
    st.stop() # Hata varsa uygulamayı durdur

# --- Gemini Modelini Ayarlama ---
# gemini-1.5-flash en hızlı ve maliyet-etkin modellerden biridir.
model = genai.GenerativeModel('gemini-pro')

# --- Prompt (AI'a Vereceğimiz Komut) Tasarımı ---
def create_prompt(cv, ilan):
    return f"""
    Sen kıdemli bir İnsan Kaynakları (İK) uzmanısın ve görevin bir CV ile bir iş ilanını karşılaştırmak.
    Aşağıdaki CV metni ile İŞ İLANI metnini detaylıca analiz et.

    Analizini yaparken şu adımları izle:
    1.  **Genel Uyum Skoru:** CV'nin ilana uygunluğunu 100 üzerinden puanla.
    2.  **Güçlü Yönler (Artılar):** Adayın ilandaki gereksinimleri karşılayan en güçlü 3-4 yönünü listele.
    3.  **Zayıf Yönler / Eksiklikler (Eksiler):** İlanda aranan ancak CV'de bulunmayan veya zayıf olan 3-4 noktayı listele.
    4.  **Değerlendirme Özeti:** 2-3 cümlelik kısa bir genel değerlendirme yazısı yaz.

    Lütfen cevabını net başlıklar kullanarak **Markdown formatında** ver.

    ---[CV METNİ]----
    {cv}
    -----------------

    ---[İŞ İLANI METNİ]---
    {ilan}
    -----------------
    """

# --- Kullanıcı Arayüzü (UI) ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 CV Metni")
    cv_text = st.text_area("Adayın CV'sini buraya yapıştırın", height=300, label_visibility="collapsed")

with col2:
    st.subheader("🎯 İş İlanı Metni")
    ilan_text = st.text_area("İş ilanını buraya yapıştırın", height=300, label_visibility="collapsed")

# --- Buton ve Çalıştırma Mantığı ---
if st.button("Uyum Analizi Yap", type="primary", use_container_width=True):
    if cv_text and ilan_text:
        # Butona basıldığında yükleniyor animasyonu göster
        with st.spinner("Gemini AI, CV ve ilanı analiz ediyor... Lütfen bekleyin."):
            try:
                # Prompt'u oluştur
                prompt = create_prompt(cv_text, ilan_text)
                
                # Gemini API'a isteği gönder
                response = model.generate_content(prompt)
                
                # Sonucu ekrana yazdır
                st.divider()
                st.subheader("✨ Analiz Sonucu")
                st.markdown(response.text)
                
            except Exception as e:
                st.error(f"Analiz sırasında bir hata oluştu: {e}")
    else:
        st.warning("Lütfen hem CV hem de iş ilanı alanlarını doldurun.")
