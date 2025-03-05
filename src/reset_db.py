# src/reset_db.py
from chromadb import HttpClient

def reset_chroma_db():
    """기존 컬렉션 삭제 및 새 컬렉션 생성"""
    client = HttpClient(host="localhost", port=8001)
    
    # 기존 컬렉션 삭제
    try:
        client.delete_collection(name="collection1")
        print("기존 컬렉션 'collection1'이 삭제되었습니다.")
    except Exception as e:
        print(f"컬렉션 삭제 실패: {e}")
    
    # 새 컬렉션 생성
    try:
        collection = client.create_collection(name="collection1")
        print("새 컬렉션 'collection1'이 생성되었습니다.")
    except Exception as e:
        print(f"컬렉션 생성 실패: {e}")
        return None

    return collection

if __name__ == "__main__":
    reset_chroma_db()
