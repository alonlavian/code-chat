import os
from typing import Any, Dict, List

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import VertexAIEmbeddings
from langchain.chat_models import ChatOpenAI, ChatGooglePalm
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Pinecone, Chroma
import pinecone


pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT_REGION"],
)


# def run_g_llm(query: str, chat_history: List[Dict[str, Any]] = []):
#     embeddings = VertexAIEmbeddings()  # Dimention 768
#
#     vectorstore = Pinecone.from_existing_index(
#         embedding=embeddings,
#         index_name=os.environ["PINECONE_INDEX_NAME"],
#     )
#     chat = ChatGooglePalm(
#         verbose=True,
#         temperature=0,
#     )
#
#     qa = ConversationalRetrievalChain.from_llm(
#         llm=chat, retriever=vectorstore.as_retriever(), return_source_documents=True
#     )
#     return qa({"question": query, "chat_history": chat_history})


def run_llm(query: str, chat_history: List[Dict[str, Any]] = [], vectordb: str = "chroma"):
    global docsearch
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])

    if vectordb == "pinecone":
        docsearch = Pinecone.from_existing_index(
            embedding=embeddings,
            index_name=os.environ["PINECONE_INDEX_NAME"],
        )
    elif vectordb == "chroma":
        docsearch = Chroma(persist_directory="./db", embedding_function=embeddings)

    chat = ChatOpenAI(
        verbose=True,
        temperature=0,
    )

    qa = ConversationalRetrievalChain.from_llm(
        llm=chat, retriever=docsearch.as_retriever(), return_source_documents=True
    )
    return qa({"question": query, "chat_history": chat_history})
