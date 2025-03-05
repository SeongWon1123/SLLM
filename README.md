# SLLM
SLLM project

.\ui\Scripts\activate => 내가 편할려고 적어놓은 가상환경임.

시작하는 방법
docker에 chroma DB 를 올려야함
https://velog.io/@cathx618/Docker%EC%97%90%EC%84%9C-ChromaDB-%EC%82%AC%EC%9A%A9%ED%95%98%EA%B8%B0 > 이 사이트 참고

docker desktop 을 설치한 뒤, 구동시킨 다음
chroma DB를 올리면 chroma 폴더가 생길텐데 cd chroma 를 들어간 뒤,
visual 터미널에서 docker-compose up -d 를 진행
docker 서버가 정상적으로 진행이 된다면
cd .. 로 프로젝트 폴더로 나와 python app.py 를 진행 하면 정상적으로 백엔드는 실행된 걸로 알 수 있음
react를 사용하는 방법은, cmd나 터미널에 들어간 뒤 cd frontend -> npm install -> npm start 진행
