from google import genai
from google.genai import types
from gtts import gTTS
import os
import pypdf
import streamlit as st

# Sayfa Yapılandırması
st.set_page_config(
    page_title="Sesli PDF & Web Akıllı Asistan", page_icon="🎧", layout="wide"
)

st.title("🎧 Sesli PDF & Web Araştırma Asistanı")
st.markdown(
    "Gemini altyapısıyla çalışan; yüklediğin PDF'leri okuyan, internette arama"
    " yapan ve yanıtları **sesli okuyabilen** yapay zekan!"
)

# Güvenli API Anahtarı Kontrolü (Streamlit Secrets üzerinden)
try:
  api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
  with st.sidebar:
    st.header("⚙️ Ayarlar")
    api_key = st.text_input("Google Gemini API Key", type="password")
    st.markdown(
        "[Ücretsiz API Key Al](https://aistudio.google.com/app/apikey)"
    )

if not api_key:
  st.warning("Lütfen Gemini API Key'ini tanımlayın.")
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
for idx, message in enumerate(st.session_state.messages):
  with st.chat_message(message["role"]):
    st.markdown(message["content"])
    if message["role"] == "assistant" and "audio_path" in message:
      if os.path.exists(message["audio_path"]):
        st.audio(message["audio_path"], format="audio/mp3")

# Kullanıcı Girişi
if prompt := st.chat_input(
    "Bir şeyler sor (örn: 'Bu PDF'i sesli olarak özetle' veya güncel bir şey"
    " sor)"
):
  st.session_state.messages.append({"role": "user", "content": prompt})
  with st.chat_message("user"):
    st.markdown(prompt)

  with st.chat_message("assistant"):
    with st.spinner("Yapay zeka düşünüyor, araştırıyor ve ses hazırlıyor..."):
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

        # Güncel model adı kullanıldı: gemini-3.8-flash
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=full_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=[{"google_search": {}}],
                temperature=0.3,
            ),
        )

        answer = response.text
        st.markdown(answer)

        # Metni Ses Dosyasına Çevirme (gTTS)
        audio_file_path = f"temp_audio_{len(st.session_state.messages)}.mp3"
        tts = gTTS(text=answer, lang="tr", slow=False)
        tts.save(audio_file_path)

        # Ses oynatıcıyı arayüze ekle
        st.audio(audio_file_path, format="audio/mp3")

        # Mesajı ve ses dosyası yolunu hafızaya kaydet
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "audio_path": audio_file_path,
        })

      except Exception as e:
        st.error(f"Bir hata oluştu: {e}")
