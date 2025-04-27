from langchain.document_loaders import HuggingFaceDatasetLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA
import torch
from langchain_community.document_loaders import PyPDFLoader
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import UnstructuredExcelLoader


from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_ollama.llms import OllamaLLM

from typing import Dict
import os
from typing import Dict, List, TypedDict
from langchain_core.documents import Document
import os

# Define state for application at the module level (before the class that uses it)
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    doc_type: str  # Add this to track document type
class InsuranceModel:
    def __init__(self):
        self.models = {}  # Dictionary to store models for different document types
        self.embeddings = self.text_embedding()
        self.vector_stores = {}  # Dictionary to store vector stores for different document types
        self.prompts = {}  # Dictionary to store prompts for different document types
        self.load_all_models()
    
    def load_all_models(self):
        # Define your document types and their paths
        document_types = {
            "policy_advice": r"D:\Backend_insurance\Algorithm\RAG\updateddata\policy_advice.xlsx",
            "business_consulting": r"D:\Backend_insurance\Algorithm\RAG\updateddata\business_consulting.xlsx",
            "platform_operations": r"D:\Backend_insurance\Algorithm\RAG\updateddata\platform_operations.xlsx",
            "procedures": r'D:\Backend_insurance\Algorithm\RAG\updateddata\procedures.xlsx',
            "phone_numbers": r'D:\Backend_insurance\Algorithm\RAG\updateddata\new_phone_numbers.pdf',
            "all": r'D:\Backend_insurance\Algorithm\RAG\pension_complaints_rewritten.xlsx'
            
        }

        
        # Initialize all models, vector stores, and prompts
        for doc_type, file_path in document_types.items():
            self.initialize_document_type(doc_type, file_path)
        
        # Load the Ollama model (shared across all document types)
        self.model = OllamaLLM(model="deepseek-r1:14b", base_url="http://localhost:11434")
    def text_embedding(self):
        return HuggingFaceEmbeddings(model_name=r"D:\jincheng_project\RAG\all-mpnet-base-v2")
    def initialize_document_type(self, doc_type: str, file_path: str):
        # Load and split documents
        all_splits = self.doc_splitting(file_path, doc_type)
        
        # Create vector store
        vector_store = self.create_vector_store()
        vector_store.add_documents(documents=all_splits)
        self.vector_stores[doc_type] = vector_store
        
        # Create prompt template (you can customize this per document type if needed)
        self.prompts[doc_type] = self.format_prompt(doc_type)
        
        # Create and compile graph for this document type
        graph_builder = StateGraph(State).add_sequence([self.retrieve, self.generate])
        graph_builder.add_edge(START, "retrieve")
        self.models[doc_type] = graph_builder.compile()
    
    def doc_splitting(self, file_path: str, doc_type: str):
        if doc_type == 'phone_numbers':
            loader = PyPDFLoader(file_path)
        else:
            loader = UnstructuredExcelLoader(file_path, mode="elements")
        docs = loader.load()

        all_splits = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        ).split_documents(docs)
        return all_splits
    
    def create_vector_store(self):
        embedding_dim = len(self.embeddings.embed_query("hello world"))
        index = faiss.IndexFlatL2(embedding_dim)

        vector_store = FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        return vector_store
    
    def retrieve(self, state: State):
        doc_type = state.get("doc_type", "pension")  # Default to pension if not specified
        retrieved_docs = self.vector_stores[doc_type].similarity_search(state["question"], k=1)
        if not retrieved_docs:
            return {"context": [Document(page_content="无相关信息")]}
        return {"context": retrieved_docs}

    def generate(self, state: State):
        doc_type = state.get("doc_type", "pension")  # Default to pension if not specified
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        formatted_prompt = self.prompts[doc_type].format(
            question=state["question"], 
            context=docs_content
        )
        
        response = self.model(formatted_prompt)
        
        if isinstance(response, dict):
            answer = response.get("generated_text", "").replace(formatted_prompt, "").strip()
        else:
            answer = response.strip()
        
        return {"answer": answer}

    def format_prompt(self, doc_type: str):
        # You can customize prompts per document type if needed
        tem = {
            "policy_advice": """您是一位专业的养老保险理赔助理，并且已经掌握了所有与养老金政策相关的信息。请根据您以往的经验，直接回答用户关于养老金政策的咨询。答案必须准确。
                如果有人询问某项政策，并且涉及与某个部门打交道，而您掌握了该部门的信息，请注明部门名称。如果您没有相关信息，请避免提供错误的部门名称。如果有人问“您好吗”之类的泛泛问题，您必须进行相应的回答。如果该政策与任何应用程序（例如民城山西应用程序）相关，您必须明确提及。

                您的回复应该非常专业，就像一位专业的养老金政策专家一样，并且回复中应该包含“作为养老金政策专家”的字样。避免提供电话号码，并提及，如果有人想要电话号码信息，请联系12345山西热线或到电话号码部分获取更多电话号码信息
                严格的规则:
                严格的规则:
                    在任何情况下，请勿在回复中提供 12345 以外的任何电话号码（如有必要）。如果有人询问电话号码，请回复他联系山西热线 12345 或访问我们网站的电话号码部分。我们网站的电话号码部分

                上下文：
                {context}

                问题：
                {question}

                答案：""",
            "business_consulting": """您是一位专业的养老保险理赔助理，并且已经掌握了所有与养老保险业务运营相关的信息。请根据您以往的经验，直接回答用户关于业务咨询的咨询。答案必须准确。
                如果有人咨询，并且涉及与某个部门打交道，而您掌握该部门的信息，请注明部门名称。如果您没有相关信息，请避免提供错误的部门名称。如果有人问“您好吗”之类的泛泛问题，您必须进行相应的回答。如果该政策与任何应用程序（例如民城山西应用程序）相关，您必须明确提及。

                您的回复应该非常专业，就像一位专业的养老金业务运营顾问一样，并在回复中包含“作为业务顾问”的字样。避免提供电话号码，并提及，如果有人想要电话号码信息，请联系12345山西热线或到电话号码部分获取更多电话号码信息
                严格的规则:
                严格的规则:
                    在任何情况下，请勿在回复中提供 12345 以外的任何电话号码（如有必要）。如果有人询问电话号码，请回复他联系山西热线 12345 或访问我们网站的电话号码部分。我们网站的电话号码部分

                上下文：
                {context}

                问题：
                {question}

                答案：""",
            "platform_operations": """您是一位专业的养老保险理赔助理，并且您已经了解所有与平台运营相关的信息。请根据您以往的经验，直接回答用户关于平台运营的咨询。答案必须准确。
                如果有人询问平台运营，并且涉及与某个部门打交道，而您掌握该部门的信息，请提及该部门。如果您没有相关信息，请避免提供错误的部门名称。如果有人问“您好吗”之类的泛泛问题，您必须进行相应的回答。如果该政策与任何应用程序（例如民城山西应用程序）相关，您必须明确提及。

                您的回复应该非常专业，就像一位专业的平台运营专家一样，并且回复中应该包含“作为平台运营专家”的字样。避免提供电话号码，并提及，如果有人想要电话号码信息，请联系12345山西热线或到电话号码部分获取更多电话号码信息
                严格的规则:
                严格的规则:
在任何情况下，请勿在回复中提供 12345 以外的任何电话号码（如有必要）。如果有人询问电话号码，请回复他联系山西热线 12345 或访问我们网站的电话号码部分。我们网站的电话号码部分                上下文：
                {context}

                问题：
                {question}

                答案：""",
            "procedures": """您是一位专业的养老保险理赔助理，并且您已经了解所有与养老金流程相关的信息。请根据您过去的经验，直接回答用户关于养老金流程的咨询。答案必须准确。
                如果有人询问政策，并且涉及与某个部门打交道，而您掌握该部门的信息，请提及该部门。如果您没有相关信息，请避免提供错误的姓名。如果有人问“您好吗”之类的一般性问题，您必须进行相应的回答。如果该流程与任何应用程序（例如民城山西应用程序）相关，您必须明确提及。

                您的回复应该非常专业，就像一位专业的养老金流程专家一样，并且回复中应包含“作为养老金流程专家”的字样。避免提供电话号码，并提及，如果有人想要电话号码信息，请联系12345山西热线或到电话号码部分获取更多电话号码信息
                严格的规则:
                严格的规则:
在任何情况下，请勿在回复中提供 12345 以外的任何电话号码（如有必要）。如果有人询问电话号码，请回复他联系山西热线 12345 或访问我们网站的电话号码部分。我们网站的电话号码部分                上下文：
                {context}

                问题：
                {question}

                答案：""",
            "phone_numbers": """你是一个专业的养老保险理赔助理，你已经知道所有与电话号码相关的信息。根据您过去的经验，请直接回答用户有关电话号码的查询。答案必须准确。
                                如果有人询问您不知道的电话号码，请不要犯错，但应该说您目前没有此信息，并告诉联系12345以获取确切的电话号码。如果有人问一个一般性的问题，如"你好吗"，你必须相应地回答它。如果该过程与任何应用程序（例如"闽城山西"应用程序）相关，则必须明确提及。

                上下文：
                {context}

                问题：
                {question}

                答案：""",
            "all": """你是一个专业的退休金索赔助理。 请根据您的专业知识直接回应用户的询问. 
                答案必须准确并使用专业语言（避免第一人称代词）。

                回应架构指引:
                1. 总是以适当的问候开始
                2. 提供最准确的资料
                3. 在已知时包括相关部门名称
                4. 概述必要的步骤（如适用）
                5. 在讨论法规时解释政策和要求采取的行动

                特殊情况:
                -如果被问及有关部门：命名部门并概述解决问题的步骤
                -如果被问及有关政策：解释政策及其规定的行动
                -如果信息不可用：清楚地说明你不知道什么
                -如果有人想要电话号码信息，请回复他联系12345山西热线或到电话号码部分获取更多电话号码信息
                严格的规则:
                严格的规则:
                    不要给电话号码。 如果有人问电话号码，回复他联系山西热线12345或到我们网站的电话号码部分。 我们网站的电话号码部分
                上下文环境:
                {context}

                问题:
                {question}

                答案：:"""
        }
        
        # Example of customizing for different document types
        if doc_type == "policy_advice":
            prompt = PromptTemplate.from_template(tem["policy_advice"])
        elif doc_type == "business_consulting":
            prompt = PromptTemplate.from_template(tem["business_consulting"])
        elif doc_type == "platform_operations":
            prompt = PromptTemplate.from_template(tem["platform_operations"])
        elif doc_type == "procedures":
            prompt = PromptTemplate.from_template(tem["procedures"])
        elif doc_type == "phone_numbers":
            prompt = PromptTemplate.from_template(tem["phone_numbers"])
        elif doc_type == "all":
            prompt = PromptTemplate.from_template(tem["all"])
        else:
            prompt = PromptTemplate.from_template(tem["all"])
        return prompt

    def transcribe(self, question: str, doc_type: str = "pension"):
        if doc_type not in self.models:
            return "Invalid document type specified"
            
        response = self.models[doc_type].invoke({
            "question": question,
            "doc_type": doc_type  # Pass doc_type through the state
        })    
        return str(response['answer']).split("</think>")[-1]