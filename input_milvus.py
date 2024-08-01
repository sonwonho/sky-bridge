from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import Milvus
from langchain_text_splitters import CharacterTextSplitter

from utils.clova_embedding import ClovaEmbedding

ce = ClovaEmbedding()
embeddings = ce.get_embedding()

text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=200)

# 대학 모집요강
loader = PyPDFDirectoryLoader("data/university/", extract_images=True)
docs = loader.load()

splited_docs = text_splitter.split_documents(docs)

vector_db = Milvus.from_documents(
    splited_docs,
    embeddings,
    collection_name="rag_cosine",
    connection_args={"host": "127.0.0.1", "port": "19530"},
    index_params={
        "metric_type": "COSINE",
        "index_type": "FLAT",
    },
    search_params={"metric_type": "COSINE"},
    # drop_old=True,
)

# 대입 일정
loader = CSVLoader("data/plan/2025년 대입일정.csv")
docs = loader.load()
new_docs = []
for i in docs:
    i.metadata["page"] = i.metadata["row"]
    del i.metadata["row"]
    new_docs.append(i)

splited_docs = text_splitter.split_documents(new_docs)

vector_db = Milvus.from_documents(
    splited_docs,
    embeddings,
    collection_name="rag_cosine",
    connection_args={"host": "127.0.0.1", "port": "19530"},
    index_params={
        "metric_type": "COSINE",
        "index_type": "FLAT",
    },
    search_params={"metric_type": "COSINE"},
    # drop_old=True,
)

# 선생님 리뷰
loader = CSVLoader(
    "data/teacher_review/[입시혁명단] 스카이브릿지 RAG 데이터 - (1번 탭)강사 리뷰.csv"
)
docs = loader.load()
new_docs = []
for i in docs:
    i.metadata["page"] = i.metadata["row"]
    del i.metadata["row"]
    new_docs.append(i)

splited_docs = text_splitter.split_documents(new_docs)

vector_db = Milvus.from_documents(
    splited_docs,
    embeddings,
    collection_name="rag_cosine",
    connection_args={"host": "127.0.0.1", "port": "19530"},
    index_params={
        "metric_type": "COSINE",
        "index_type": "FLAT",
    },
    search_params={"metric_type": "COSINE"},
    # drop_old=True,
)

# 문제집
loader = CSVLoader(
    "data/workbook/[입시혁명단] 스카이브릿지 RAG 데이터 - (1번 탭) 문제집(전체).csv"
)
docs = loader.load()
new_docs = []
for i in docs:
    i.metadata["page"] = i.metadata["row"]
    del i.metadata["row"]
    new_docs.append(i)

splited_docs = text_splitter.split_documents(new_docs)

vector_db = Milvus.from_documents(
    splited_docs,
    embeddings,
    collection_name="rag_cosine",
    connection_args={"host": "127.0.0.1", "port": "19530"},
    index_params={
        "metric_type": "COSINE",
        "index_type": "FLAT",
    },
    search_params={"metric_type": "COSINE"},
    # drop_old=True,
)
