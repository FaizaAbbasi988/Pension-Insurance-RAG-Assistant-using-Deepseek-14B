# Pension Insurance RAG Assistant usng Deepseek-R1-14B

![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-00A67E?style=for-the-badge)

A Retrieval-Augmented Generation (RAG) system for answering pension insurance queries, with multilingual support (English/中文).

## Features

- 🗂️ Document-based question answering across multiple knowledge domains
- 🌐 Bilingual interface (English/Chinese)
- 🔍 Context-aware responses using RAG pipeline
- 📊 Document type specialization:
  - Policy advice
  - Business procedures
  - Platform operations
  - Claim procedures
  - Verified contact numbers

## pre-requisities

Python 3.10+
Ollama server running locally
Required documents in specified paths

## Installation
Clone the repository:

git clone https://github.com/FaizaAbbasi988/pension-rag-assistant.git 
cd pension-rag-assistant

## Project Structure

pension-rag-assistant/
├── app.py                # Streamlit frontend
├── main.py               # FastAPI backend
├── InsuranceModel.py     # RAG pipeline implementation
├── requirements.txt      # Python dependencies
├── documents/            # Sample document storage
│   ├── policy_advice.xlsx
│   ├── phone_numbers.pdf
│   └── ...
└── README.md    

## Architecture

```mermaid
graph TD
    A[Streamlit UI] --> B[FastAPI Backend]
    B --> C[RAG Pipeline]
    C --> D[Document Vector Stores]
    C --> E[LLM (Ollama)]
    D --> F[Excel/PDF Documents]