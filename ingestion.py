import os
import shutil

from langchain.document_loaders import GitLoader
from git import Repo
from langchain.embeddings import OpenAIEmbeddings, VertexAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone, Chroma
import pinecone

pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT_REGION"],
)

def delete_folder(folder_path):
    if os.path.exists(folder_path):
        # Remove the folder and its contents
        shutil.rmtree(folder_path)
        print(f"Folder '{folder_path}' and its contents have been deleted.")
    else:
        print(f"Folder '{folder_path}' does not exist.")

def ingest_docs(repo_url: str, vendor: str = "openai", vectordb: str = "chroma") -> None:
    global embeddings
    to_path = "./repo_to_embed"
    delete_folder(to_path)

    repo = Repo.clone_from(repo_url, to_path=to_path, branch="main")
    loader = GitLoader(repo_path=to_path)
    raw_documents = loader.load()

    print(f"loaded {len(raw_documents)} documents")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
    )

    for doc in raw_documents:
        source = doc.metadata["source"]
        cleaned_source = "/".join(source.split("/")[1:])
        doc.page_content = (
                "FILE NAME: "
                + cleaned_source
                + "\n###\n"
                + doc.page_content.replace("\u0000", "")
        )

    documents = text_splitter.split_documents(raw_documents)

    print(f"Going to add {len(documents)} to the vector store")
    if vendor == "google":
        embeddings = VertexAIEmbeddings()
    elif vendor == "openai":
        embeddings = OpenAIEmbeddings()

    chunk_size = 5
    for i in range(0, len(documents), chunk_size):
        print(f"iteration {i}/{len(documents) / chunk_size}...")
        chunked_documents = documents[i: i + chunk_size]
        if vectordb == "pinecone":
            Pinecone.from_documents(
                chunked_documents, embeddings, index_name=os.environ["PINECONE_INDEX_NAME"]
            )
        elif vectordb == "chroma":
            # Embed and store the texts
            # Supplying a persist_directory will store the embeddings on disk
            persist_directory = './db'
            vectors = Chroma.from_documents(documents=chunked_documents,
                                             embedding=embeddings,
                                             persist_directory=persist_directory)
    print("****Loading to vectorestore done ***")


if __name__ == "__main__":
    ingest_docs(repo_url="https://github.com/jasonnovichRunAI/gh-pages-test")
