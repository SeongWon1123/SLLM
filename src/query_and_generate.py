# src/query_and_generate.py
import re
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from chromadb import HttpClient
import ollama
from langchain.docstore.document import Document

def remove_markdown(text):
    """
    입력된 텍스트에서 마크다운 관련 문법(헤더, 굵게, 기울임 등)을 제거합니다.
    """
    # 헤더(예: ##) 제거
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    # 굵게, 기울임 등 마크다운 기호 제거
    text = text.replace('**', '')
    text = text.replace('*', '')
    return text.strip()

def query_and_generate(query):
    """
    사용자가 입력한 쿼리에 대해 답변을 생성합니다.
    ChromaDB에서 관련 문서를 검색하고, Ollama 모델을 통해 답변을 생성합니다.
    """
    # 1. Chroma DB와 Ollama Embeddings 설정
    client = HttpClient(host="localhost", port=8000)  # ChromaDB 서버 연결
    ollama_ef = OllamaEmbeddings(model="gemma2:2b")  # Ollama 모델을 사용한 임베딩 함수

    # 2. 벡터 스토어 초기화: 예시 문서를 컬렉션에 추가
    docs_example = [
        Document(page_content="문서 내용 1", metadata={"source": "file"}),
        Document(page_content="문서 내용 2", metadata={"source": "file"})
    ]
    db = Chroma.from_documents(
        docs_example,
        ollama_ef,
        client=client,
        collection_name="collection1"
    )

    # 3. ChromaDB에서 쿼리에 관련된 문서 검색
    search_results = db.similarity_search(query)  # 쿼리와 유사한 문서 찾기

    if not search_results:
        return "데이터에서 관련 정보를 찾을 수 없습니다."

    # 4. 최적의 문서에서 답변 생성 (Document 객체이므로 인덱싱 없이 사용)
    best_doc = search_results[0].page_content

    # 5. LLM을 사용해 답변 생성 (여기서 Ollama를 사용)
    prompt = (
        f"주어진 데이터는 다음과 같습니다:\n{best_doc}\n\n"
        f"위 데이터를 바탕으로 아래 질문에 대한 답을 작성해주세요: {query}\n\n답변:"
    )

    try:
        output = ollama.generate(model="gemma2:2b", prompt=prompt)  # 모델을 통해 답변 생성
        answer = output['response']
        # 마크다운 포맷팅 제거
        answer = remove_markdown(answer)
        return answer
    except Exception as e:
        return f"답변 생성 중 오류 발생: {str(e)}"
