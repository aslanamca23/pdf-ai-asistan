from google import genai
from google.genai import types
import pypdf
import streamlit as st

# Sayfa Yapılandırması
st.set_page_config(
    page_title="PDF & Web Akıllı Asistan", page_icon="🤖", layout="wide"
)

st.title("🚀 Akıllı PDF & Web Araştırma Asistanı")
st.markdown(
    "Gemini altyapısıyla çalışan; hem yüklediğin PDF'leri okuyan hem de"
    " internette canlı arama yapabilen yapay zekan!"
)

# Güvenli API Anahtarı Kontrolü (Streamlit Secrets üzerinden alır)
try:
  api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
  # Eğer bulutta secrets ayarlanmadıysa local test için sidebar'dan alır
  with st.sidebar:
    st.header("⚙️ Ayarlar")
    api_key = st.text_input("Google Gemini API Key", type="password")
    st.markdown(
        "[Ücretsiz API Key Al](https://aistudio.google.com/app/apikey)"
    )

if not api_key:
  st.warning(
      "Lütfen Gemini API Key'ini tanımlayın (Bulutta secrets ayarlanmalı veya"
      " sidebar'dan girilmeli)."
  )
  st.stop()

# Sidebar: Dosya Yükleme
with st.sidebar:
  st.divider()
  st.markdown("### 📂 Dosya Yükleme")
  uploaded_file = st.file_uploader(
      "Bir PDF dosyası yükle", type=["pdf"], key="pdf_uploader"
  )

# Client Tanımlama
client = genai.Client(api_key=api_key)


# PDF Metnini Çıkarma Fonksiyonu
def extract_text_from_pdf(pdf_file):
  reader = pypdf.PdfReader(pdf_file)
  text = ""
  for page in reader.pages:
    text += page.extract_text() + "\n"
  return text


pdf_context = ""
if uploaded_file is not None:
  with st.spinner("PDF okunuyor ve analiz ediliyor..."):
    pdf_context = extract_text_from_pdf(uploaded_file)
  st.sidebar.success(
      f"'{uploaded_file.name}' başarıyla yüklendi ve hafızaya alındı!"
  )

# Sohbet Geçmişi Başlatma
if "messages" not in st.session_state:
  st.session_state.messages = []

# Geçmiş Mesajları Ekranda Gösterme
for message in st.session_state.messages:
  with st.chat_message(message["role"]):
    st.markdown(message["content"])

# Kullanıcı Girişi
if prompt := st.chat_input(
    "Bir şeyler sor (örn: 'Bu PDF'i özetle' veya 'Bugün dünyadaki son yapay"
    " zeka gelişmeleri neler?')"
):
  st.session_state.messages.append({"role": "user", "content": prompt})
  with st.chat_message("user"):
    st.markdown(prompt)

  with st.chat_message("assistant"):
    with st.spinner("Yapay zeka düşünüyor ve araştırıyor..."):
      try:
        system_instruction = (
            "Sen gelişmiş bir araştırma ve PDF analiz asistanısın. Kullanıcının"
            " sorularına hem yüklediği PDF belgesindeki bilgileri baz alarak"
            " hem de internet araması yaparak en doğru, güncel ve yapılandırılmış"
            " yanıtı ver. Türkçe yanıt ver."
        )

        full_prompt = prompt
        if pdf_context:
          full_prompt = (
              f"İşte yüklenen PDF belgesinin içeriği:\n---\n{pdf_context[:10000]}\n---\n\nKullanıcının"
              f" sorusu/isteği: {prompt}"
          )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=full_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=[{"google_search": {}}],
                temperature=0.3,
            ),
        )

        answer = response.text
        st.markdown(answer)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

      except Exception as e:
        st.error(f"Bir hata oluştu: {e}")