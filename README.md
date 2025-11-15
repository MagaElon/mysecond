# mysecond
second contribution
from typing import List, Tuple
import requests
from langchain_core.documents import Document

from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import PyPDFLoader, TextLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFacePipeline
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
def load_documents() -> List[Document]:
    """
    Load the three documents from the course Github raw folder.
    Returns a list of Document objects.
    """
    base_raw = "https://raw.githubusercontent.com/ntomuro/CSC380/main/LLM_RAG/data/"
    filenames = [
        "Human-Nutrition-2020-Edition-1598491699.txt",
        "dci190009_pdf.txt",
        "dci190014_pdf.txt"
    ]

    documents = []
    for fname in filenames:
        url = base_raw + fname
        print(f"Fetching {url} ...")
        r = requests.get(url)
        if r.status_code != 200:
            raise RuntimeError(f"Failed to fetch {url}: {r.status_code}")
        text = r.text

        # create a langchain Document (you can put metadata for traceability)
        doc = Document(page_content=text, metadata={"source": fname})
        documents.append(doc)

    return documents


def preprocess_documents(documents: List[Document]) -> List[Document]:
    """
    Split each Document into chunks. Returns a flat list of chunk Documents.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )

    chunks = []
    for doc in documents:
        # Get text and split
        pieces = splitter.split_text(doc.page_content)
        for i, chunk_text in enumerate(pieces):
            md = dict(doc.metadata) if doc.metadata else {}
            md.update({"chunk_id": f"{md.get('source','doc')}_chunk_{i}"})
            chunks.append(Document(page_content=chunk_text, metadata=md))

    return chunks
from langchain_huggingface import HuggingFaceEmbeddings


def create_vector_store(documents: List[Document], embedding_model_name: str, client):
    """
    Build a Chroma vector store from 'documents' using the specified sentence-transformer.
    Returns a retriever object.
    """
    print(f"Creating embeddings with model: {embedding_model_name}")
    # initialize HuggingFaceEmbeddings wrapper
    hf_emb = HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs={"device": "cpu"})

    # Create a Chroma vectorstore. We'll put it in the client's persistent path already set up.
    # The Chroma constructor for langchain usually accepts persist_directory OR a chromadb client.
    vectordb = Chroma.from_documents(
        documents,
        embedding=hf_emb,
        client=client,   # pass the persistent chromadb client from main()
        collection_name=f"collection_{embedding_model_name}"
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    return retriever
def create_test_queries() -> List[str]:
    """
    3 answerable and 3 unanswerable queries. At least one long question per type.
    """
    # Answerable (should be covered by the nutrition + diabetes corpus)
    answerable = [
        "What are the recommended macronutrient distribution ranges for adults in the Human Nutrition 2020 guidelines?",
        "Describe dietary recommendations specifically for adults with prediabetes to reduce progression to diabetes. (Be specific about carbs and fiber.)",
        # Long answerable:
        "Summarize how dietary carbohydrate quality (e.g., refined vs. whole grains, fiber content) affects glycemic control and long-term diabetes risk, and cite the consensus recommendations given across the documents."
    ]

    # Unanswerable (should *not* be present in the corpus)
    unanswerable = [
        "Is there a single pharmaceutical drug recommended by the Human Nutrition 2020 textbook for reversing type 2 diabetes?",
        # Long unanswerable:
        "Can you provide a step-by-step 30-day meal plan that guarantees diabetes remission, including exact calories and day-by-day recipes, based solely on the three corpus documents?",
        "Does the corpus recommend intermittent fasting as the primary treatment for type 1 diabetes?"
    ]

    return answerable + unanswerable
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
def run_rag_query(query: str, retriever) -> Tuple[str, List[Document]]:
    # retrieve
    retrieved_docs = retriever.get_relevant_documents(query)
    # build context string
    context = "\n\n".join([f"Source: {d.metadata.get('source','unknown')}\n{d.page_content[:1500]}" for d in retrieved_docs])

    prompt_text = """
You are an evidence-aware assistant. Use only the provided context to answer the user's question.
If the answer cannot be found in the context, respond clearly: "I couldn't find information in the provided sources."
Context:
{context}

Question:
{question}

Answer concisely and cite which source(s) you used (by source filename).
"""
    prompt = PromptTemplate(template=prompt_text, input_variables=["context", "question"])

    # LLM chain with OpenAI
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0.0)
    chain = LLMChain(llm=llm, prompt=prompt)
    answer = chain.run({"context": context, "question": query})

    return answer, retrieved_docs
