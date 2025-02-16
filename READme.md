## 🌱 Agricultural AI Assistant & Crop Recommendation 🌾

An AI-powered Streamlit application that provides crop recommendations, crop disease diagnosis, and an intelligent conversational assistant to help farmers and agricultural enthusiasts.

🚀 Features

1️⃣ Crop Recommendation System

📊 Inputs: Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH, Rainfall

🌾 Predicts the most suitable crop based on soil and environmental conditions

🎯 Uses a Random Forest model trained on agricultural data

2️⃣ Crop Disease Diagnosis (AI-Powered)

🔍 Identifies potential diseases based on symptoms

🩺 Suggests organic & chemical treatments

🛡 Provides preventive measures for future protection

🤖 Powered by AI (Groq LLM) for expert analysis

3️⃣ Conversational AI for Agriculture

💬 Ask any agriculture-related questions (e.g., best farming practices, fertilizers, weather impact, etc.)

🔥 AI-driven chatbot provides instant, accurate responses


🛠 Tech Stack
Python 🐍

Streamlit 🎨 (for UI)

Scikit-Learn 🤖 (for Machine Learning)

LangChain & Groq API 🧠 (for AI-powered chatbot & disease diagnosis)

Pickle 📦 (for model persistence)

## 🎯 Project Structure

📂 Crop_Recommendation
│── 📂 Chat                    
│   ├── testing_prompts.ipynb 
    |---RAG.ipynb  
│
│── 📂 data                   
│   ├── Crop_recommendation.csv 
│   ├── farmerbook.pdf          
│
│── 📂 EDA                     
│   ├── explore.ipynb           
│
│── 📂 Research                 
│   ├── model_selection.ipynb  
│   ├── Predictions.ipynb      
│
│── 📂 saved_models            
│   ├── RF_Model.pkl   
|   |--gb_model.pkl         
│── .env                       
│── .gitignore                  
│── app.py                      
│── README.md                   
│── requirements.txt           

