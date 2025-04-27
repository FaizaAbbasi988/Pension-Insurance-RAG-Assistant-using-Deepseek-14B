import streamlit as st
import requests
from typing import Dict, List
import json

# Set page config FIRST (before any other Streamlit commands)
st.set_page_config(
    page_title="Pension Insurance RAG Assistant",  # Default title, will be updated later
    layout="wide"
)



# Configuration
BACKEND_URL = "http://localhost:7000"

# Enhanced document type descriptions
DOCUMENT_TYPES = {
    "en": {
        "all": "General Pension Information",
        "policy_advice": "Policy Advice (Expert-level guidance)",
        "business_consulting": "Business Procedures",
        "platform_operations": "Platform Operations",
        "procedures": "Claim Procedures",
        "phone_numbers": "Verified Contact Numbers"
    },
    "zh": {
        "all": "综合养老金信息",
        "policy_advice": "政策咨询 (专家级指导)",
        "business_consulting": "业务流程",
        "platform_operations": "平台运营",
        "procedures": "索赔流程",
        "phone_numbers": "已验证的联系电话"
    }
}

# More detailed translations
TRANSLATIONS = {
    "en": {
        "title": "Pension Insurance RAG Assistant",
        "description": "Ask questions about pension policies using our Retrieval-Augmented Generation system",
        "question_placeholder": "Type your pension-related question...",
        "submit_button": "Ask RAG System",
        "document_type_label": "Select knowledge domain:",
        "language_label": "Language:",
        "response_header": "RAG System Response",
        "debug_header": "System Information",
        "model_info": "Model Details",
        "api_status": "API Status",
        "loading": "Querying RAG pipeline...",
        "error": "API Error - Please try again"
    },
    "zh": {
        "title": "养老金保险RAG助手",
        "description": "使用我们的检索增强生成系统咨询养老金政策问题",
        "question_placeholder": "输入您关于养老金的问题...",
        "submit_button": "询问RAG系统",
        "document_type_label": "选择知识领域:",
        "language_label": "语言:",
        "response_header": "RAG系统回复",
        "debug_header": "系统信息",
        "model_info": "模型详情",
        "api_status": "API状态",
        "loading": "正在查询RAG管道...",
        "error": "API错误 - 请重试"
    }
}

def get_api_status():
    try:
        response = requests.get(f"{BACKEND_URL}/health")
        return response.status_code == 200, response.json()
    except:
        return False, {}

def get_model_info():
    try:
        response = requests.get(f"{BACKEND_URL}/model_details")
        return response.status_code == 200, response.json()
    except:
        return False, {}

def send_query(question: str, doc_type: str, language: str):
    try:
        response = requests.post(
            f"{BACKEND_URL}/query",
            json={
                "question": question,
                "doc_type": doc_type,
                "language": language
            }
        )
        return True, response.json()
    except Exception as e:
        return False, {"error": str(e)}

def main():
    # Language selection
    language = st.sidebar.selectbox(
        TRANSLATIONS["en"]["language_label"],
        options=["zh", "en"],
        format_func=lambda x: "中文" if x == "zh" else "English",
        key="lang_select"
    )
    
    # Set page config

    
    # Display title and description
    st.title(TRANSLATIONS[language]["title"])
    st.markdown(f"**{TRANSLATIONS[language]['description']}**")
    
    # API status indicator
    api_ok, api_status = get_api_status()
    model_ok, model_info = get_model_info()
    
    with st.expander(TRANSLATIONS[language]["debug_header"]):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(TRANSLATIONS[language]["api_status"])
            if api_ok:
                st.success("✅ API Connected")
                st.json(api_status)
            else:
                st.error("❌ API Not Available")
        
        with col2:
            st.subheader(TRANSLATIONS[language]["model_info"])
            if model_ok:
                st.success("✅ Model Loaded")
                st.json(model_info)
            else:
                st.error("❌ Model Info Unavailable")
    
    # Document type selection
    doc_type = st.selectbox(
        TRANSLATIONS[language]["document_type_label"],
        options=list(DOCUMENT_TYPES[language].keys()),
        format_func=lambda x: DOCUMENT_TYPES[language][x],
        key="doc_type_select"
    )
    
    # Question input
    question = st.text_area(
        "",
        placeholder=TRANSLATIONS[language]["question_placeholder"],
        height=150,
        key="question_input"
    )
    
    # Submit button
    if st.button(TRANSLATIONS[language]["submit_button"]):
        if question.strip():
            with st.spinner(TRANSLATIONS[language]["loading"]):
                success, response = send_query(question, doc_type, language)
                
                if success:
                    st.subheader(TRANSLATIONS[language]["response_header"])
                    st.markdown(f"**Question:** {response['question']}")
                    st.markdown(f"**Document Type:** {DOCUMENT_TYPES[language][response['doc_type']]}")
                    st.divider()
                    st.markdown(f"**Answer:**\n\n{response['answer']}")
                    
                    # Debug info (can be hidden in production)
                    with st.expander("RAG Debug Info"):
                        st.json(response)
                else:
                    st.error(TRANSLATIONS[language]["error"])
                    st.error(response.get("error", "Unknown error"))
        else:
            st.warning("Please enter a question")

if __name__ == "__main__":
    main()