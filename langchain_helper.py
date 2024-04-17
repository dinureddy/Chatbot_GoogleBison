from langchain_google_genai import GoogleGenerativeAI
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
# from InstructorEmbedding import INSTRUCTOR
from dotenv import load_dotenv
import os

load_dotenv()
model_name = 'models/text-bison-001'
llm = GoogleGenerativeAI(model=model_name, google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0.7)

# Load the raw data from the CSV
faq_data_path = 'codebasics_faqs.csv'
vectordb_file_path = 'db'
instruct_embeds = HuggingFaceInstructEmbeddings()


# Create embeddings and Vector Store
def create_vector_db():
    # Load data from FAQ sheet
    loader = CSVLoader(file_path=faq_data_path, source_column='prompt', encoding='Windows-1252')
    docs = loader.load()

    # Create a FAISS instance for vector database from 'data'
    vector_db = FAISS.from_documents(documents=docs, embedding=instruct_embeds)

    # Save vector database locally
    vector_db.save_local(vectordb_file_path)


def get_qa_chain():
    # Load the vector database from the local folder
    vector_db = FAISS.load_local(vectordb_file_path, embeddings=instruct_embeds, allow_dangerous_deserialization=True)

    # Create a retriever for querying the vector database
    retriever = vector_db.as_retriever(score_threshold=0.7)

    prompt_template = """Given the following context and a question, generate an answer based on this context only. 
    In the answer try to provide as much text as possible from "response" section in the source document context 
    without making much changes. If the answer is not found in the context, kindly state "I don't know." Don't try to 
    make up an answer.

        CONTEXT: {context}

        QUESTION: {question}
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type='stuff',
                                        retriever=retriever,
                                        input_key="query",
                                        chain_type_kwargs={"prompt": prompt},
                                        return_source_documents=True)
    return chain
