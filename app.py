import streamlit as st
import numpy as np
import pickle
import os
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Set Streamlit Page Config
st.set_page_config(
    page_title="Agricultural AI Assistant 🌱",
    layout="wide"
)

load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
chat = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")

model_path = r"C:\Users\saipr\Crop_Recommendation\saved_models\RF_Model.pkl"
model = pickle.load(open(model_path, 'rb'))

st.markdown("""
    <style>
        .title { text-align: center; color: mediumseagreen; }
        .warning { color: red; font-weight: bold; text-align: center; }
        .container {
            background: #edf2f7; font-weight: bold;
            padding: 20px; border-radius: 15px; margin-top: 20px;
        }
        .stButton>button {
            background-color: #007bff; color: white;
            font-size: 16px; font-weight: bold; border: none;
            border-radius: 5px; padding: 10px 20px;
        }
        .stTextInput>div>input {
            border-radius: 5px; border: 1px solid #007bff; padding: 10px;
        }
    </style>
""", unsafe_allow_html=True)

if 'flow_messages' not in st.session_state:
    st.session_state['flow_messages'] = [
        SystemMessage(content="You are a highly intelligent and friendly agricultural assistant. Provide accurate and relevant answers about crops, farming, and agricultural practices.")
    ]

def get_response(question):
    st.session_state['flow_messages'].append(HumanMessage(content=question))
    answer = chat(st.session_state['flow_messages'])
    st.session_state['flow_messages'].append(AIMessage(content=answer.content))
    return answer.content

st.markdown('<h1 class="title">🌾 Agricultural AI Assistant</h1>', unsafe_allow_html=True)
st.sidebar.header("🔹 Features")
features = st.sidebar.radio("Choose a feature:", ("Crop Recommendation", "Crop Disease Diagnosis", "Conversational Q&A"))

if features == "Crop Recommendation":
    st.write("### 📊 Provide the necessary agricultural parameters:")

    N = st.number_input('Nitrogen', min_value=0, max_value=150, step=1)
    P = st.number_input('Phosphorus', min_value=0, max_value=100, step=1)
    K = st.number_input('Potassium', min_value=0, max_value=100, step=1)
    temp = st.number_input('Temperature (°C)', min_value=-10.0, max_value=60.0, step=0.1)
    humidity = st.number_input('Humidity (%)', min_value=0.0, max_value=100.0, step=0.1)
    ph = st.number_input('pH', min_value=0.0, max_value=14.0, step=0.1)
    rainfall = st.number_input('Rainfall (mm)', min_value=0.0, max_value=1000.0, step=1.0)

    if st.button('🌱 Get Recommendation'):
        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)
        
        prediction = model.predict(single_pred)[0]  

        crop = str(prediction).strip().title()

        st.success(f"🌾 **{crop}** is the best crop for the provided data!")

elif features == "Crop Disease Diagnosis":
    st.write("### 🦠 Diagnose Crop Diseases")

    symptoms = st.text_input("🔍 Enter Symptoms (e.g., yellow leaves, wilting):")
    crop = st.text_input("🌱 Enter Crop Name (e.g., Tomato, Wheat):")
    location = st.text_input("📍 Enter Location (e.g., Punjab, India):")
    season = st.selectbox("🗓 Select Season:", ["Summer", "Winter", "Rainy", "Spring", "Autumn"])

    disease_prompt = PromptTemplate(
        input_variables=["symptoms", "crop", "location", "season"],
        template=(
            "You are an expert plant pathologist assisting farmers in diagnosing crop diseases.\n\n"
            "📌 **Symptoms:** {symptoms}\n"
            "🌱 **Crop:** {crop}\n"
            "📍 **Location:** {location}\n"
            "🗓 **Season:** {season}\n\n"
            "### 🦠 Possible Disease(s) and Causes:\n"
            "- Analyze symptoms and list possible diseases.\n"
            "- Mention environmental and pest-related causes.\n\n"
            "### 💊 Treatment & Remedies:\n"
            "- Suggest **organic** and **chemical** treatments.\n"
            "- Recommend suitable pesticides or fungicides (if needed).\n\n"
            "### 🛡 Preventive Measures:\n"
            "- Guide the farmer on crop rotation, irrigation, and soil treatment.\n"
            "- Suggest resistant crop varieties if available."
        )
    )

    if st.button("🩺 Diagnose"):
        chain = LLMChain(llm=chat, prompt=disease_prompt)
        response = chain.run(symptoms=symptoms, crop=crop, location=location, season=season)
        st.write(response)

elif features == "Conversational Q&A":
    st.write("### 💬 Ask an Agriculture-related Question")
    user_input = st.text_input("Your Question:")
    if st.button("🤖 Ask AI"):
        if user_input.strip():
            response = get_response(user_input)
            st.subheader("AI Response:")
            st.write(response)
        else:
            st.warning("⚠️ Please enter a question!")

