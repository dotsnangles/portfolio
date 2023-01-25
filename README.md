## 직무 경험

- 데이터 수집 및 정제
- 전처리 및 증식
- 라벨 검수 및 수정
- 훈련 및 검증
- 추론 및 후처리
- 백엔드 API 작성 및 데이터베이스 연동
- 분석 및 시각화

## 개발 환경 및 도구

- Ubuntu, Windows
- AWS EC2, WSL2, Colab
- Git/Github, Papermill, WandB
- PyTorch, Tensorflow, Huggingface, Sentence-Transformers, Scikit-learn, Gensim
- Pandas, OpenCV, NumPy, Matplotlib
- BeautifulSoup, Selenium
- Flask, MySQL, SQLAlchemy, Marshmallow

## 수행 프로젝트

### NPL 기반 항만 안전사고 인과 모델 생성을 위한 학습 DB구축 참여 - 한국철도기술원 위임 용역

개체명 인식

- 논문 탐색 및 요약 정리
- 전처리 및 라벨 오류 수정
- 사고 환경 정보 취합을 위한 BERT-LSTM-CRF 개체명 인식 모델 개발
- 추론 및 후처리를 위한 스크립트 작성
- 후속 연구 제안 및 보고서 작성

[auxiliary-sequence-tagger-for-causal-relation-extraction](https://github.com/dotsnangles/auxiliary-sequence-tagger-for-causal-relation-extraction)  
[causal-event-extractor](https://github.com/dotsnangles/causal-event-extractor)

---

### 신문기사 수집 및 분석을 통한 민관 맞춤 알선 시스템 개발 - 외부 업체 의뢰 용역

문서 수집 및 정제, 규칙 기반 문서 선별, 유사도 기반 문서 추천

- 웹크롤링을 통한 신문기사 수집 및 정제
- 단어 출현 빈도 스레시홀딩을 적용한 문서 유효성 검증
- SBERT를 활용한 유사도 기반 문서 추천 및 시각화
- 프로토타입 스크립트 작성

[targeted-news-crawling-and-document-retrieval-based-on-similarity-scoring](https://github.com/dotsnangles/targeted-news-crawling-and-document-retrieval-based-on-similarity-scoring)  
[dimensionality-reduction-in-python-with-scikit-learn](https://github.com/dotsnangles/dimensionality-reduction-in-python-with-scikit-learn)

---

### 교육 자료 개발 - 용역

- 논문 탐색 및 요약 정리
- 구현 코드 탐색 및 Tensorflow-PyTorch 번역
- 폐렴 및 부정맥 진단 보조 의료 인공지능
- YOLOV4/5/7을 활용한 객체 인식
- StyleGAN / CycleGAN 데모

[from-keras-to-pytorch-and-more](https://github.com/dotsnangles/from-keras-to-pytorch-and-more)

---

### 감성 대화 챗봇 개발 - 부트캠프 프로젝트 평가 1위

유사도 기반 문서 추천, 챗봇 구현, 백엔드 구축

- Poly Encoder를 활용한 문서 유사도 기반 챗봇 시스템 구상 및 구현
- 효과적인 유사도 훈련을 위한 데이터 구성 방법 고안
- 모델 훈련 및 챗봇 모듈/스크립트 작성
- 모델 활용을 위한 데이터베이스 구축 및 REST API 작성

[backend-api-for-chatbot-with-Poly-Encoder](https://github.com/dotsnangles/backend-api-for-chatbot-with-Poly-Encoder)  
[retrieval-based-chatbot-with-Poly-Encoder](https://github.com/dotsnangles/retrieval-based-chatbot-with-Poly-Encoder)  
[Poly-Encoder](https://github.com/dotsnangles/Poly-Encoder)

---

### 온라인 쇼핑몰 상품평 속성 기반 감성 분석

문서 분류, 개체명 인식, 문서 자료 증식, 모델 파이프라인, 모델 앙상블

- 문서쌍을 입력으로 하는 분류 모델 개발
- 훈련 데이터 수정을 통한 개체명 인식 모델 개발
- 모델 파이프라인 구축 및 시험
- 각종 문서 자료 증식 기법 구현 및 정리

[aspect-based-sentiment-analysis](https://github.com/dotsnangles/aspect-based-sentiment-analysis)

---

### BART를 활용한 회의록 요약

문서 요약, 문서 자료 증식, Decoding Methods

- Encoder-Decoder 계열 사전학습모델을 파인튜닝하여 문서 요약 모델 개발
- Easy Data Augmentation 기법을 구현하여 증식 수행
- Dacon AI 기반 회의 녹취록 요약 경진대회 Public 7위 (전체 489팀/연습참가)
- Decoding Methods 정리 (Greedy Search, Beam Search, Sampling)

[text-summarisation-with-BART](https://github.com/dotsnangles/text-summarisation-with-BART)

---

### 신문기사 주제 분류

문서 분류, 문서 자료 증식, 모델 앙상블

- BERT 계열 사전학습모델을 파인튜닝하여 문서 분류 모델 개발
- 백트랜슬레이션 증식 수행
- 모델 앙상블 구현
- Dacon 뉴스 토픽 분류 AI 경진대회 Public 19위 (전체 418팀/연습참가)

[text-classification-with-BERT](https://github.com/dotsnangles/text-classification-with-BERT)

---

### T5를 활용한 한영/영한 번역

기계 번역, 벤치마크

- 다중 언어 T5를 파인튜닝하여 한영 번역기와 영한 번역기를 개발
- 토크나이저의 사전 크기를 고려해 10일 정도에 걸쳐 훈련을 진행
- 수렴 이후 충분한 성능을 발휘하는 것을 BLEU 스코어를 통해 검증

[NMT-with-transformers](https://github.com/dotsnangles/NMT-with-transformers)

---

### GPT를 활용한 도메인 특정 문서 생성

문서 생성, 문서 수집 및 정제

- 여러 장르의 시 문서를 웹 크롤링을 통해 수집 후 정제
- GPT-2의 성능을 검증하기 위해 파인튜닝 후 샘플링 기법을 적용하여 문서를 생성
- Perplexity를 측정하여 파인튜닝 전후 모델 성능 차이를 검증

[poetry-generator-with-GPT2](https://github.com/dotsnangles/poetry-generator-with-GPT2)

---

### Music VAE를 활용한 드럼 비트 생성

- Magenta 라이브러리를 활용한 미디 데이터 생성
- 논문에서 제한하는 구조의 모델 정의
- 훈련 후 샘플링 및 Inerpolation 진행

[beat-generation-with-mvae](https://github.com/dotsnangles/beat-generation-with-mvae)

---

[Link to Bootcamp Retrospective](https://github.com/dotsnangles/bootcamp-retrospective)