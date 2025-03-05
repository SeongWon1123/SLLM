import os
import uuid
import csv
from tqdm import tqdm
from langchain.docstore.document import Document
from langchain_community.document_loaders import TextLoader  # CSVLoader 대체로 직접 구현
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb import HttpClient
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
import ollama

# CSV 파일에서 Document 객체 리스트를 생성하는 함수 (텍스트 컬럼 이름이 "text"라고 가정)
def load_csv_documents(file_path, text_column="text"):
    documents = []
    with open(file_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 지정한 컬럼이 있으면 사용, 없으면 모든 열을 합침
            if text_column in row and row[text_column]:
                text = row[text_column]
            else:
                text = " ".join(row.values())
            doc = Document(page_content=text, metadata=row)
            documents.append(doc)
    return documents

# 파일 확장자에 따라 문서를 불러오는 함수 (현재 CSV 파일만 사용)
def load_documents(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".csv":
        return load_csv_documents(file_path, text_column="text")
    else:
        loader = TextLoader(file_path)
        return loader.load()

# data 폴더 내의 모든 CSV 파일의 문서를 로드하는 함수
def load_all_csv_documents(data_folder):
    documents = []
    for filename in os.listdir(data_folder):
        if filename.lower().endswith(".csv"):
            file_path = os.path.join(data_folder, filename)
            docs = load_documents(file_path)
            documents.extend(docs)
    return documents

# 데이터를 ChromaDB에 삽입하는 함수
def insert_data(data_folder):
    # data 폴더 내의 모든 CSV 파일에서 문서를 로드
    documents = load_all_csv_documents(data_folder)
    if not documents:
        print("CSV 문서가 없습니다.")
        return

    # ChromaDB 클라이언트 설정 (호스트: localhost, 포트: 8000)
    client = HttpClient(host="localhost", port=8000)
    
    # 임베딩 함수 설정: OllamaEmbeddingFunction을 사용하여 gemma2:2b 모델을 호출
    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="gemma2:2b"
    )
    
    # 컬렉션 "collection1"을 가져오거나 없으면 생성
    try:
        collection = client.get_collection(name="collection1", embedding_function=ollama_ef)
    except Exception:
        collection = client.create_collection(name="collection1", embedding_function=ollama_ef)
    
    # 문서 청크 분할: RecursiveCharacterTextSplitter 사용 (문맥을 최대한 보존하도록)
    # 최적의 파라미터는 데이터 특성에 따라 다르므로 여기서는 예시로 chunk_size=300, chunk_overlap=50 사용
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    
    # 모든 Document 객체의 청크를 모으기
    all_chunks = []
    for doc in documents:
        chunks = text_splitter.split_documents([doc])
        all_chunks.extend(chunks)
    
    # 각 청크에 대해 고유 ID를 생성하고 컬렉션에 저장
    for chunk in tqdm(all_chunks, desc="Inserting documents", total=len(all_chunks), leave=True):
        unique_id = str(uuid.uuid4())
        collection.add(ids=[unique_id], documents=[chunk.page_content])
    
    print("Stored Data:", collection.peek())

# Retrieval 및 최종 모델 호출을 통한 답변 생성 함수
def query_and_generate(query):
    # ChromaDB 클라이언트 설정
    client = HttpClient(host="localhost", port=8000)
    
    # 임베딩 함수 설정: 동일하게 gemma2:2b 모델 사용
    from langchain_community.embeddings import OllamaEmbeddings
    embedding_function = OllamaEmbeddings(model="gemma2:2b")
    
    # 최신 버전의 langchain_community.vectorstores.Chroma를 사용하여 벡터 스토어 생성
    from langchain_community.vectorstores import Chroma
    db = Chroma(client=client, collection_name="collection1", embedding_function=embedding_function)
    
    # Query 임베딩 후, 유사한 문서와 유사도 점수를 검색 (유사도 점수는 거리 기반으로 계산)
    docs = db.similarity_search_with_score(query)
    if not docs:
        return "데이터에서 관련 정보를 찾을 수 없습니다."
    
    # 가장 유사한 문서를 선택 (가장 작은 거리값)
    best_doc = min(docs, key=lambda x: x[1])[0].page_content
    
    # 최종 프롬프트 구성: 검색된 문서를 기반으로 모델에게 구체적인 질문 및 가이드 제공
    prompt = (
        "다음 데이터는 경기 기록 등 다양한 정보를 포함하고 있습니다. "
        "각 데이터 항목에는 날짜, 구장, 원정팀, 홈팀, 점수, 승/패 결과, 비고 등 세부 정보가 포함되어 있습니다.\n\n"
        "데이터:\n"
        f"{best_doc}\n\n"
        "위 데이터를 참고하여, 아래 질문에 대해 구체적이고 상세하게 답변해 주세요. "
        "만약 데이터에 질문에 필요한 정보가 충분하지 않다면, '답변할 수 없습니다.'라고 명시해 주세요.\n\n"
        "질문:\n"
        f"{query}\n\n"
        "답변은 반드시 한국어로 작성해 주세요."
    )

    
    # Ollama 모델 호출 (gemma2:2b 모델 사용)
    output = ollama.generate(
        model="gemma2:2b",
        prompt=prompt
    )
    return output['response']

# 전체 파이프라인 실행 예제
def main():
    data_folder = "C:/Users/user/Desktop/webui/chroma/data"
    
    # 1. 데이터 삽입: data 폴더 내의 모든 CSV 파일을 로드해 ChromaDB에 저장
    insert_data(data_folder)
    
    # 2. Query 및 답변 생성: 예시 질문
    query = "연령별 성별 통계는 어떻게 되나요?"
    answer = query_and_generate(query)
    print("Final Answer:", answer)

if __name__ == "__main__":
    main()
