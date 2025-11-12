import streamlit as st
import google.generativeai as genai
import pandas as pd

st.set_page_config(page_title="Model Bulucu", page_icon="🔍")
st.title("🔍 Hangi Modellerim Var?")
st.markdown("Bu araç, Streamlit Secrets'taki API anahtarınızın kullanabildiği **çalışan** Gemini modellerini listeler.")

try:
    # API Anahtarını DOĞRU İSİMLE çağırıyoruz
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    st.success("API Anahtarı başarıyla yüklendi. Modeller aranıyor...")
except Exception as e:
    st.error(f"API Anahtarı ('GEMINI_API_KEY') Streamlit Secrets'ta bulunamadı veya geçersiz: {e}")
    st.stop()

# --- Model Listeleme ---
try:
    model_list = []
    # API'den tüm modelleri listelemesini istiyoruz
    for model in genai.list_models():
        # Bizim için önemli olan 'generateContent' metodunu desteklemesi
        # Çünkü bizim uygulamamız bu metodu kullanıyor.
        if 'generateContent' in model.supported_generation_methods:
            model_list.append({
                "Model Adı (Bunu Kopyalayın)": model.name,
                "Açıklama": model.description
            })

    if not model_list:
        st.warning("API anahtarınız 'generateContent' metodunu destekleyen hiçbir model bulamadı. Bu çok nadir bir durum. Lütfen Google AI Studio'da API anahtarınızı ve projenizi kontrol edin.")
    else:
        st.info("Aşağıdaki modellerden BİRİNİ kullanabilirsiniz. 'Model Adı' sütunundakini kopyalayın:")
        st.dataframe(pd.DataFrame(model_list), use_container_width=True)
        st.balloons()

except Exception as e:
    st.error(f"Modeller listelenirken bir hata oluştu: {e}")
    st.warning("API anahtarınızın Google AI Studio'da doğru projeye bağlı olduğundan ve 'Generative AI API'nin etkin olduğundan emin olun.")
