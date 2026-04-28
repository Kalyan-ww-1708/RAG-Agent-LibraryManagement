from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

def document_loader():
    loader = CSVLoader(file_path="booksdata2.0.csv")
    document = loader.load()
    return document

def chunking_document(documents):
    # Sentence chunking can be used for better chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    return chunks

def create_vector_store(chunks):
    embeddings = OpenAIEmbeddings()
    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local("faiss_index")
    return vector_db

def load_retriever():
    embeddings = OpenAIEmbeddings()
    vector_db = FAISS.load_local("faiss_index",embeddings,allow_dangerous_deserialization=True)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    return retriever

def query_retriever(query):
    retriever = load_retriever()
    results = retriever.invoke(query)
    context = ""
    for doc in results:
        context += doc.page_content + "\n"

    return context

def agent(query):
    llm = ChatOpenAI(temperature=2)

    context = query_retriever(query)

    prompt = f"""
You are a helpful library assistant.

Use ONLY the below context to answer the question.
Make sure your answer is relevant to the book.
Make sure you take context and formatise the answer
If the answer is not in the context, say "I don't know".
Dont add any extra information from your database or web.

Context:
{context}

Question:
{query}

Answer:
"""
    response = llm.invoke(prompt)
    print("\n Answer:")
    return response.content

    # print(response.content)


def main():
    # Run this ONLY first time to create index
    # documents = document_loader()
    # chunks = chunking_document(documents)
    # create_vector_store(chunks)

    # Ask question
    query = "Suggest a biography."
    agent(query)
