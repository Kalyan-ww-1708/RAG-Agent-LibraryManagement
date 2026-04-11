import os
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# --- CHANGED IMPORTS ---
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# -----------------------
from dotenv import load_dotenv

load_dotenv()


# Note: Ensure GOOGLE_API_KEY is in your .env file

def document_loader():
    loader = CSVLoader(file_path="booksdata2.0.csv")
    document = loader.load()
    return document


def chunking_document(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    return chunks


def create_vector_store(chunks):
    # --- CHANGED TO GEMINI EMBEDDINGS ---
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_db = FAISS.from_documents(chunks, embeddings)
    # Changed index name to avoid dimension mismatch with old OpenAI index
    vector_db.save_local("faiss_index_gemini")
    return vector_db


def load_retriever():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_db = FAISS.load_local("faiss_index_gemini", embeddings, allow_dangerous_deserialization=True)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    return retriever


def query_retriever(query):
    retriever = load_retriever()
    results = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in results])
    return context


def agent(query):
    # --- CHANGED TO GEMINI LLM ---
    # Lowered temperature slightly; Gemini at 2.0 can be very unpredictable.
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)

    context = query_retriever(query)

    prompt = f"""
You are a helpful library assistant.
Use ONLY the below context to answer the question.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{query}
"""
    response = llm.invoke(prompt)
    print("\n📌 Answer:")
    print(response.content)


def main():
    # --- IMPORTANT ---
    # You MUST re-create the index because OpenAI and Google
    # embeddings have different vector dimensions.

    # documents = document_loader()
    # chunks = chunking_document(documents)
    # create_vector_store(chunks)

    query = "Suggest a legal Thriller."
    agent(query)


if __name__ == "__main__":
    main()