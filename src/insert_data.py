import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from src.query_and_generate import query_and_generate
import ollama

app = Flask(__name__)
CORS(app)  # 모든 도메인에 대해 CORS 허용

app.secret_key = "your_secret_key"
UPLOAD_FOLDER = os.path.join(os.getcwd(), "data")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {"csv", "txt", "pdf"}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def insert_data(folder_path):
    """
    지정된 폴더 내의 CSV 파일들을 읽어 데이터를 처리합니다.
    (실제 데이터베이스 삽입 등의 로직으로 수정할 수 있습니다.)
    """
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    reader = csv.reader(file)
                    for row in reader:
                        # CSV 데이터 처리 (예: 데이터베이스 삽입 로직 등)
                        print(row)
            except Exception as e:
                print(f"파일 처리 중 오류 발생 ({file_path}): {e}")
    return "Data inserted successfully!"

@app.route("/", methods=["GET", "POST"])
def index():
    answer = None
    if request.method == "POST":
        if "upload" in request.form:
            if "file" not in request.files:
                flash("파일이 없습니다.")
                return redirect(request.url)
            file = request.files["file"]
            if file.filename == "":
                flash("선택된 파일이 없습니다.")
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(file_path)
                # 업로드된 파일을 기반으로 데이터 삽입 실행
                insert_data(app.config["UPLOAD_FOLDER"])
                flash("파일 업로드 및 데이터 삽입 완료!")
                return redirect(url_for("index"))
        elif "query" in request.form:
            query = request.form.get("query")
            if query:
                answer = query_and_generate(query)
    return render_template("index.html", answer=answer)

@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        print("파일 파트가 없음")
        return jsonify({"error": "파일 파트가 없습니다."}), 400
    file = request.files["file"]
    if file.filename == "":
        print("선택된 파일이 없음")
        return jsonify({"error": "선택된 파일이 없습니다."}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        upload_folder = app.config["UPLOAD_FOLDER"]
        os.makedirs(upload_folder, exist_ok=True)
        file_path = os.path.join(upload_folder, filename)
        try:
            file.save(file_path)
            print(f"파일 저장됨: {file_path}")
            insert_data(upload_folder)
            return jsonify({"message": "파일 업로드 및 데이터 삽입 완료!"})
        except Exception as e:
            print("파일 저장 에러:", e)
            return jsonify({"error": f"파일 저장 실패: {str(e)}"}), 500
    else:
        print("허용되지 않은 파일 형식")
        return jsonify({"error": "허용되지 않은 파일 형식입니다."}), 400

@app.route("/api/answer", methods=["POST"])
def answer():
    data = request.get_json()
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "Query is empty"}), 400
    response = query_and_generate(query)
    return jsonify({"answer": response})

@app.route("/api/normal", methods=["POST"])
def normal_answer():
    data = request.get_json()
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "Query is empty"}), 400
    # 간단한 프롬프트 구성: retrieval 없이 바로 gemma2:2b 모델에 질문 전달
    prompt = (
        f"아래 질문에 대해 답변해 주세요:\n{query}\n\n"
        "답변은 반드시 한국어로 작성해 주세요."
    )
    try:
        output = ollama.generate(
            model="gemma2:2b",
            prompt=prompt
        )
        return jsonify({"answer": output['response']})
    except Exception as e:
        return jsonify({"error": f"일반 답변 생성 실패: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=5000)