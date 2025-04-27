from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from InsuranceModel import InsuranceModel
import uvicorn
import logging

# Import configuration
from config import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(**config.api_config)

# Initialize your actual model with RAG pipeline
try:
    logger.info("Initializing InsuranceModel with RAG pipeline...")
    model = InsuranceModel()
    logger.info("Model initialization completed successfully")
except Exception as e:
    logger.error(f"Failed to initialize model: {str(e)}")
    raise RuntimeError("Failed to initialize the InsuranceModel") from e

class QueryRequest(BaseModel):
    question: str
    doc_type: str = "all"
    language: str = "zh"  # Default to Chinese

class QueryResponse(BaseModel):
    answer: str
    question: str
    doc_type: str
    language: str
    context: Optional[str] = None  # For debugging
    model_type: str = "RAG"  # Indicates we're using RAG

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process user query through the RAG pipeline and return response.
    
    Parameters:
    - question: The user's question
    - doc_type: Type of document to query (policy_advice, business_consulting, etc.)
    - language: Language preference ('zh' for Chinese, 'en' for English)
    """
    try:
        logger.info(f"Processing query: {request.question} for doc_type: {request.doc_type}")
        
        # Process through your actual RAG pipeline
        answer = model.transcribe(request.question, request.doc_type)
        
        # For debugging/transparency, you could add context if needed
        context = ""  # You could extract this from your pipeline if needed
        
        logger.info(f"Successfully processed query")
        
        return {
            "answer": answer,
            "question": request.question,
            "doc_type": request.doc_type,
            "language": request.language,
            "context": context,
            "model_type": "RAG"
        }
    except KeyError as e:
        logger.error(f"Invalid document type: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid document type. Available types: {list(model.models.keys())}"
        )
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

@app.get("/document_types")
async def get_document_types():
    """Return available document types from the model"""
    try:
        return config.get_document_types()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model_details")
async def get_model_details():
    """Return information about the RAG model configuration"""
    try:
        return {
            "model_type": "RAG",
            **config.model_config,
            "vector_stores": list(model.vector_stores.keys()) if hasattr(model, 'vector_stores') else []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint that verifies model is loaded"""
    try:
        # Verify model is operational by checking one of the vector stores
        if not hasattr(model, 'vector_stores') or not model.vector_stores:
            raise HTTPException(status_code=503, detail="Model not initialized")
        
        return {
            "status": "healthy",
            "model_initialized": True,
            "vector_stores_loaded": len(model.vector_stores)
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=config.server_config["host"],
        port=config.server_config["port"]
    )