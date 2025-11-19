# 필요한 라이브러리들과 그 역할
# pip install langchain langchain-community faiss-cpu sentence-transformers

# === LangChain 관련 라이브러리들 ===
from langchain_community.llms import Ollama  # Ollama LLM과 연결하기 위한 라이브러리
from langchain.text_splitter import RecursiveCharacterTextSplitter  # 긴 문서를 작은 청크로 나누는 도구
from langchain_community.vectorstores import FAISS  # 벡터 데이터베이스 (Facebook AI Similarity Search)
from langchain_community.embeddings import HuggingFaceEmbeddings  # 텍스트를 벡터로 변환하는 임베딩 모델
from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader  # 다양한 파일 형식 로더
from langchain.chains import RetrievalQA  # 검색 기반 질의응답 체인
from langchain.prompts import PromptTemplate  # LLM에게 전달할 프롬프트 템플릿

# === 기본 Python 라이브러리들 ===
import os  # 파일시스템 작업 (경로 확인, 디렉토리 생성 등)
import pickle  # 파이썬 객체 직렬화/역직렬화 (저장/로드)
import time  # 시간 측정 및 지연 처리
import threading  # 멀티스레딩 (비동기 입력 처리)
import queue  # 스레드 간 안전한 데이터 전달
from tqdm import tqdm  # 진행률 표시바 (시각적 피드백)

class RAGSystem:
    """
        RAG (Retrieval-Augmented Generation) 시스템 클래스

        작동 원리:
            1. 문서 로드 → 2. 텍스트 분할 → 3. 벡터화 → 4. 벡터 DB 저장
            5. 사용자 질문 → 6. 유사 문서 검색 → 7. LLM에게 컨텍스트 제공 → 8. 답변 생성
    """

    def __init__(self, embedding_model_name="jhgan/ko-sroberta-multitask", device=None):
        """
            RAG 시스템 초기화

            매개변수:
                - embedding_model_name: 텍스트를 벡터로 변환할 때 사용할 모델
                                       (jhgan/ko-sroberta-multitask는 한국어에 특화된 모델)
                - device: 사용할 디바이스 ('cpu', 'cuda', None=자동감지)

            작동 과정:
                1. 클래스 인스턴스 생성
                2. GPU 사용 가능 여부 자동 감지
                3. 임베딩 모델 미리 로드 (중복 로드 방지)
                4. 필요한 변수들을 None으로 초기화 (나중에 할당될 예정)
        """
        self.embedding_model_name = embedding_model_name
        self.device = self._detect_device(device)  # 최적 디바이스 자동 감지
        self.embeddings = self._load_embeddings()  # 임베딩 모델 미리 로드
        self.vector_store = None  # FAISS 벡터 저장소가 저장될 변수
        self.qa_chain = None  # 질의응답 체인이 저장될 변수

    def _load_embeddings(self):
        """
            임베딩 모델 로드 메서드 (중복 로드 방지)

            작동 원리:
                1. GPU 메모리 부족 시 자동으로 CPU로 전환
                2. 모델 로드 실패 시 재시도 메커니즘
                3. 한 번만 로드해서 메모리 효율성 증대
        """
        try:
            print(f"🔄 임베딩 모델 로드 중... ({self.embedding_model_name})")
            embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={'device': self.device},
                encode_kwargs={'normalize_embeddings': True}
            )
            print(f"✅ 임베딩 모델 로드 완료 ({self.device})")
            return embeddings

        except RuntimeError as e:
            if "CUDA out of memory" in str(e) and self.device == 'cuda':
                print("⚠️ GPU 메모리 부족 - CPU로 자동 전환")
                self.device = 'cpu'
                return self._load_embeddings()  # CPU로 재시도
            else:
                raise e
        except Exception as e:
            print(f"❌ 임베딩 모델 로드 실패: {e}")
            raise e

    def _detect_device(self, device=None):
        """
            최적의 디바이스 자동 감지 메서드

            작동 원리:
                1. 사용자가 명시적으로 지정했으면 그것 사용
                2. 자동 감지 시 GPU 사용 가능성 체크
                3. GPU 메모리 충분한지 확인
                4. 안전한 디바이스 선택해서 반환
        """
        if device is not None:
            print(f"🎯 사용자 지정 디바이스: {device}")
            return device

        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                print(f"🎮 GPU 감지: {gpu_name}")
                print(f"💾 GPU 메모리: {gpu_memory:.1f}GB")

                if gpu_memory >= 4.0:
                    print("✅ GPU(FP16 최적화) 사용으로 설정")
                    self.torch_dtype = torch.float16  # FP16 고정
                    return 'cuda'
                else:
                    print("⚠️ GPU 메모리 부족 - CPU 사용")
                    self.torch_dtype = torch.float32
                    return 'cpu'
            else:
                print("💻 GPU 없음 - CPU 사용")
                self.torch_dtype = torch.float32
                return 'cpu'

        except ImportError:
            print("⚠️ PyTorch 없음 - CPU 사용")
            self.torch_dtype = torch.float32
            return 'cpu'

    def load_docs(self, directory):
        """
            문서 로드 및 청크 분할 메서드

            작동 원리:
                1. 지정된 디렉토리에서 여러 형식의 파일을 스캔
                2. 각 파일 형식에 맞는 로더로 텍스트 추출
                3. 긴 문서를 작은 청크(덩어리)로 분할
                4. 분할된 청크들을 리스트로 반환

            왜 청크로 나누나?
                - LLM의 토큰 제한 때문에 긴 문서를 통째로 처리할 수 없음
                - 작은 단위로 나누면 더 정확한 검색이 가능
                - 메모리 효율성도 향상
        """
        print(f"📁 {directory}에서 문서 로딩 중...")

        # 다양한 파일 형식을 처리할 수 있는 로더들 설정
        # DirectoryLoader: 디렉토리 전체를 스캔해서 특정 확장자 파일들을 찾음
        # glob 패턴: **/*.pdf는 하위 디렉토리까지 모든 pdf 파일을 의미
        loaders = [
            # PDF 파일 로더
            DirectoryLoader(directory, glob="**/*.pdf", loader_cls=PyPDFLoader),

            # 텍스트 파일 로더
            DirectoryLoader(directory, glob="**/*.txt", loader_cls=TextLoader),

            # 마크다운 파일 로더
            DirectoryLoader(directory, glob="**/*.md", loader_cls=TextLoader)
        ]

        documents = []  # 로드된 모든 문서를 저장할 리스트

        # 각 로더를 순회하면서 문서 로드 시도
        for loader in loaders:
            try:
                # 실제 파일 로드 실행
                docs = loader.load()

                # 로드된 문서들을 전체 리스트에 추가
                documents.extend(docs)
                print(f"✅ {len(docs)}개 문서 로드 완료")
            except Exception as e:
                # 특정 파일 형식에서 오류가 발생해도 다른 형식은 계속 처리
                print(f"⚠️ 로더 오류: {e}")
                continue

        # 로드된 문서가 없으면 오류 발생
        if not documents:
            raise ValueError("로드할 문서가 없습니다. 디렉토리 경로를 확인해주세요.")

        # 문서를 적절한 크기로 분할하는 과정
        # RecursiveCharacterTextSplitter: 문서의 구조를 고려해서 자연스럽게 분할
        text_splitter = RecursiveCharacterTextSplitter(
            # 각 청크의 최대 문자 수 (한국어는 영어보다 정보 밀도가 높아 작게 설정)
            chunk_size=800,

            # 청크 간 겹치는 문자 수 (맥락 유지를 위해)
            chunk_overlap=100,

            # 분할 우선순위 (문단 → 문장 → 단어 순)
            separators=["\n\n", "\n", ".", "!", "?", " ", ""]
        )

        # 실제 문서 분할 실행
        chunks = text_splitter.split_documents(documents)
        print(f"📊 총 {len(chunks)}개의 문서 청크 생성 완료")
        return chunks

    def create_vector_db(self, chunks, save_path="faiss_index"):
        """
            벡터 저장소 생성 및 저장 메서드

            작동 원리:
                1. 임베딩 모델 로드 (텍스트를 숫자 벡터로 변환하는 모델)
                2. 각 문서 청크를 벡터로 변환
                3. FAISS 라이브러리를 사용해 벡터 데이터베이스 생성
                4. 생성된 벡터 DB를 디스크에 저장 (다음에 재사용 가능)

            벡터 DB가 필요한 이유:
                - 텍스트 유사도를 빠르게 계산하기 위해
                - 수백만 개의 벡터 중에서 가장 유사한 것들을 빠르게 찾기 위해
                - FAISS는 Facebook에서 개발한 고성능 벡터 검색 라이브러리
        """
        print("🔄 임베딩 모델 로드 중...")
        start_time = time.time()  # 처리 시간 측정 시작

        # HuggingFace의 사전 훈련된 임베딩 모델 로드
        # 이 모델은 텍스트를 512차원 또는 768차원 벡터로 변환함
        self.embeddings = HuggingFaceEmbeddings(
            # 한국어 특화 모델 사용
            model_name=self.embedding_model_name,

            # 자동 감지된 최적 디바이스 사용
            model_kwargs={'device': self.device},

            # 벡터 정규화로 검색 성능 향상
            encode_kwargs={'normalize_embeddings': True}
        )

        print("🔄 벡터 저장소 생성 중...")

        # FAISS 벡터 저장소 생성 과정:
            # 1. 각 문서 청크를 임베딩 모델로 벡터화
            # 2. 모든 벡터를 FAISS 인덱스에 저장
            # 3. 빠른 유사도 검색이 가능한 구조로 최적화
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)

        # 벡터 DB를 디스크에 저장 (다음 실행 시 빠른 로드를 위해)
        # save_local은 인덱스 파일과 메타데이터를 함께 저장
        self.vector_store.save_local(save_path)
        print(f"💾 벡터 DB를 {save_path}에 저장 완료")

        elapsed_time = time.time() - start_time
        print(f"⏱️ 벡터 DB 생성 시간: {elapsed_time:.2f}초")

        return self.vector_store

    def load_existing_vector_db(self, save_path="faiss_index"):
        """
            기존에 저장된 벡터 DB를 로드하는 메서드 (속도 향상의 핵심!)

            작동 원리:
                1. 이미 저장된 벡터 DB 파일이 있는지 확인
                2. 임베딩 모델을 다시 로드 (벡터 DB와 같은 모델이어야 함)
                3. 저장된 FAISS 인덱스를 메모리로 로드
                4. 성공하면 True, 실패하면 False 반환

            속도 향상 효과:
                - 새로 생성: 수분~수십분 소요
                - 기존 로드: 수초만 소요
        """
        try:
            print(f"🔄 기존 벡터 DB 로드 중... ({save_path})")

            # 이미 로드된 임베딩 모델 사용 (중복 로드 방지)
            # 벡터 DB와 임베딩 모델의 차원과 특성이 일치해야 함
            if not self.embeddings:
                print("⚠️ 임베딩 모델이 로드되지 않음 - 다시 로드")
                self.embeddings = self._load_embeddings()

            # 저장된 FAISS 인덱스를 로드
            # allow_dangerous_deserialization=True: pickle 파일 로드 허용
            self.vector_store = FAISS.load_local(save_path, self.embeddings, allow_dangerous_deserialization=True)
            print("✅ 기존 벡터 DB 로드 완료!")
            return True

        except Exception as e:
            print(f"⚠️ 기존 벡터 DB 로드 실패: {e}")
            return False

    def setup_rag_chain(self, model_name="gpt-oss:20b", search_mode="balanced"):
        """
            RAG 질의응답 체인 설정 메서드

            매개변수:
                - model_name: 사용할 Ollama 모델명
                - search_mode: 검색 모드 ("fast", "balanced", "comprehensive", "all")

            작동 원리:
                1. Ollama LLM 모델 연결 (연결 실패 시 재시도)
                2. 한국어 답변을 위한 프롬프트 템플릿 설정
                3. 검색 기반 질의응답 체인 구성
                4. 검색 파라미터를 모드에 따라 최적화 설정

            RAG 체인의 동작 과정:
                사용자 질문 → 벡터 검색 → 관련 문서 추출 → 프롬프트 생성 → LLM 답변 → 결과 반환
        """
        print(f"🤖 Ollama 모델 ({model_name}) 연결 중...")

        try:
            # Ollama 모델 설정 및 연결 테스트
            # temperature: 답변의 창의성 조절 (0에 가까울수록 일관된 답변)
            # top_p: 다음 토큰 선택 시 고려할 확률 범위
            llm = Ollama(
                model=model_name,

                # 낮은 값으로 설정해서 정확한 답변 유도
                temperature=0.1,
                top_p=0.9
            )

            # Ollama 연결 테스트
            try:
                test_response = llm.invoke("테스트")
                print("✅ Ollama 모델 연결 성공!")
            except Exception as e:
                print(f"⚠️ Ollama 연결 테스트 실패: {e}")
                print("💡 Ollama가 실행 중인지 확인해주세요: ollama serve")
                raise e

        except Exception as e:
            print(f"❌ Ollama 모델 연결 실패: {e}")
            raise e

        # 전체 문서 개수 확인 (all 모드용)
        total_docs = self.vector_store.index.ntotal if self.vector_store else 100

        # 검색 모드에 따른 파라미터 설정
        search_configs = {
            "fast": {
                "search_type": "similarity",
                "k": 5,
                "description": "빠른 답변 (5-15초, 핵심 정보만)"
            },
            "balanced": {
                "search_type": "similarity_score_threshold",
                "k": 15,
                "score_threshold": 0.6,
                "description": "균형 잡힌 품질과 속도 (30초-1분)"
            },
            "comprehensive": {
                "search_type": "mmr",  # Maximum Marginal Relevance
                "k": 30,
                "fetch_k": 50,
                "lambda_mult": 0.7,
                "description": "포괄적 검색 (1-3분, 다양한 정보)"
            },
            "all": {
                "search_type": "similarity",
                "k": min(total_docs, 200),  # 최대 200개로 제한 (안전장치)
                "description": f"전체 검색 (5-20분, 모든 문서 {total_docs}개)"
            }
        }

        config = search_configs.get(search_mode, search_configs["balanced"])
        print(f"🔍 검색 모드: {search_mode} - {config['description']}")

        # all 모드일 때 경고 메시지
        if search_mode == "all":
            print(f"⚠️ 전체 검색 모드: {config['k']}개 문서를 처리합니다")
            print("💡 고성능 GPU 없이는 매우 느릴 수 있습니다")
            confirm = input("계속하시겠습니까? (y/n): ").lower().strip()
            if confirm != 'y':
                print("🔄 balanced 모드로 변경합니다")
                config = search_configs["balanced"]
                search_mode = "balanced"

        # 한국어 답변을 위한 프롬프트 템플릿 정의
        # 이 템플릿이 LLM에게 어떻게 답변할지 지시함
        prompt_template = """
            다음 문서들을 바탕으로 질문에 정확하게 답변해주세요.
            답변할 때는 반드시 제공된 문서의 내용만을 사용하고, 없는 정보는 추측하지 마세요.

            문서 내용:
            {context}

            질문: {question}

            답변 (한국어로, 정확하고 구체적으로):"""

        # PromptTemplate 객체 생성
        # input_variables: 템플릿에서 사용할 변수들 정의
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        # 검색 설정 구성
        search_kwargs = {key: value for key, value in config.items()
                         if key not in ["search_type", "description"]}

        # RetrievalQA 체인 구성 - RAG의 핵심 부분!
        # 작동 과정:
            # 1. retriever가 질문과 유사한 문서들을 벡터 DB에서 검색
            # 2. 검색된 문서들을 context로 프롬프트에 삽입
            # 3. LLM이 context를 바탕으로 답변 생성
        self.qa_chain = RetrievalQA.from_chain_type(
            # 사용할 언어 모델
            llm=llm,

            # 검색된 문서들을 어떻게 처리할지 (stuff: 모든 문서를 하나로 합침)
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_type=config["search_type"],
                search_kwargs=search_kwargs
            ),

            # 참조한 문서들도 함께 반환 (출처 확인용)
            return_source_documents=True,

            # 사용할 프롬프트 템플릿
            chain_type_kwargs={"prompt": PROMPT}
        )

        print("✅ RAG 체인 설정 완료!")
        return self.qa_chain

    def answer_question(self, question):
        """
            사용자 질문에 답변하는 메서드

            작동 과정:
                1. 질문을 받아서 벡터로 변환
                2. 벡터 DB에서 유사한 문서들 검색
                3. 검색된 문서들과 질문을 프롬프트에 결합
                4. LLM에게 전달해서 답변 생성
                5. 답변과 참조 문서를 함께 반환
        """
        if not self.qa_chain:
            raise ValueError("RAG 체인이 설정되지 않았습니다. setup_rag_chain()을 먼저 실행하세요.")

        print(f"🔍 질문 처리 중: {question}")
        print("💭 답변 생성 중입니다... (잠시만 기다려주세요)")
        start_time = time.time()

        try:
            # qa_chain 실행 - 실제 RAG 동작이 일어나는 부분
            # 내부적으로 다음 과정이 진행됨:
            # 1. 질문 벡터화 → 2. 유사 문서 검색 → 3. 프롬프트 생성 → 4. LLM 호출 → 5. 결과 반환
            response = self.qa_chain({"query": question})

            answer = response["result"]  # LLM이 생성한 답변
            source_docs = response["source_documents"]  # 참조한 문서들

            elapsed_time = time.time() - start_time
            print(f"⏱️ 답변 생성 시간: {elapsed_time:.2f}초")

            # 결과를 딕셔너리 형태로 정리해서 반환
            return {
                "answer": answer,
                "sources": [doc.page_content[:200] + "..." for doc in source_docs],  # 출처는 200자만 표시
                "response_time": elapsed_time
            }

        except Exception as e:
            print(f"❌ 답변 생성 오류: {e}")
            return {
                "answer": "죄송합니다. 답변 생성 중 오류가 발생했습니다.",
                "sources": [],
                "response_time": 0
            }

def main():
    """
        메인 실행 함수 - 전체 RAG 시스템의 진입점

        프로그램 실행 흐름:
            1. RAG 시스템 초기화
            2. 벡터 DB 로드 또는 생성
            3. LLM 체인 설정
            4. 사용자 입력 처리 루프 시작
            5. 중복 요청 방지 메커니즘 동작
    """
    # RAG 시스템 인스턴스 생성
    rag = RAGSystem()

    # 파일 경로 설정
    # 원본 문서들이 있는 디렉토리
    documents_dir = "./uploads"

    # 벡터 DB가 저장될 경로
    vector_db_path = "faiss_index"

    # 기존 벡터 DB 확인 및 로드/생성 결정
    # 이 부분이 속도 향상의 핵심!
    if os.path.exists(vector_db_path):
        print("📋 기존 벡터 DB를 사용하시겠습니까? (y/n): ", end="")
        use_existing = input().lower().strip()

        if use_existing == 'y':
            # 기존 벡터 DB 로드 시도 (빠른 시작)
            if rag.load_existing_vector_db(vector_db_path):
                print("🚀 기존 벡터 DB 로드 완료 - 빠른 시작!")
            else:
                # 로드 실패 시 새로 생성
                print("⚠️ 새로운 벡터 DB를 생성합니다...")
                chunks = rag.load_docs(documents_dir)
                rag.create_vector_db(chunks, vector_db_path)
        else:
            # 사용자가 새로 생성하기를 원함
            chunks = rag.load_docs(documents_dir)
            rag.create_vector_db(chunks, vector_db_path)
    else:
        # 처음 실행하는 경우 - 반드시 새로 생성
        print("🆕 처음 실행입니다. 벡터 DB를 생성합니다...")
        chunks = rag.load_docs(documents_dir)
        rag.create_vector_db(chunks, vector_db_path)

    # RAG 체인 설정 (Ollama 모델 연결 및 프롬프트 설정)
    print("\n📋 검색 모드를 선택하세요:")
    print("1. fast - 빠른 답변 (5-15초, 상위 5개 문서)")
    print("2. balanced - 균형잡힌 검색 (30초-1분, 상위 15개 문서)")
    print("3. comprehensive - 포괄적 검색 (1-3분, 상위 30개 문서)")
    print("4. all - 전체 검색 (5-20분, 모든 문서) ⚠️ 고성능 GPU 필요")

    while True:
        mode_choice = input("모드 선택 (1/2/3/4 또는 fast/balanced/comprehensive/all): ").strip().lower()
        if mode_choice in ['1', 'fast']:
            search_mode = "fast"
            break
        elif mode_choice in ['2', 'balanced', '']:  # 기본값
            search_mode = "balanced"
            break
        elif mode_choice in ['3', 'comprehensive']:
            search_mode = "comprehensive"
            break
        elif mode_choice in ['4', 'all']:
            search_mode = "all"
            break
        else:
            print("⚠️ 올바른 모드를 선택하세요. (기본값: balanced)")
            search_mode = "balanced"
            break

    rag.setup_rag_chain("gpt-oss:20b", search_mode)

    print("\n🎉 RAG 시스템 준비 완료!")
    print("=" * 50)

    # 중복 요청 방지를 위한 변수들
    # 이 부분이 엔터 여러 번 누르는 문제를 해결!
    is_processing = False  # 현재 답변 생성 중인지 확인하는 플래그
    input_queue = queue.Queue()  # 스레드 간 안전한 입력 전달을 위한 큐

    def input_handler():
        """
            비동기 입력 처리 함수

            작동 원리:
                - 별도 스레드에서 실행되어 사용자 입력을 계속 받음
                - 받은 입력을 큐에 저장
                - 메인 스레드는 큐에서 입력을 순차적으로 처리
                - 이렇게 하면 입력과 처리를 분리할 수 있음
        """
        while True:
            try:
                # 사용자 입력 대기
                user_input = input()

                # 큐에 입력 저장
                input_queue.put(user_input)
            except (EOFError, KeyboardInterrupt):
                break

    # 입력 핸들러를 별도 스레드로 시작
    # daemon=True: 메인 프로그램 종료 시 이 스레드도 자동 종료
    input_thread = threading.Thread(target=input_handler, daemon=True)
    input_thread.start()

    # 메인 대화 루프
    while True:
        print("\n" + "=" * 50)
        print("🤔 질문을 입력하세요 (종료하려면 'exit' 입력): ", end="", flush=True)

        # 큐에서 입력 대기 (블로킹 방식)
        # get()은 큐에 뭔가 들어올 때까지 기다림
        question = input_queue.get()

        # 종료 명령 체크
        if question.lower() in ['exit', 'quit', '종료']:
            print("👋 RAG 시스템을 종료합니다. 안녕히 가세요!")
            break

        # 빈 입력 체크
        if not question.strip():
            print("⚠️ 질문을 입력해주세요.")
            continue

        # 중복 요청 방지 로직 - 핵심 부분!
        if is_processing:
            print("⚠️ 현재 답변 생성 중입니다. 잠시만 기다려주세요.")
            # 큐에 쌓인 중복 입력들을 모두 제거
            # get_nowait(): 큐가 비어있으면 즉시 예외 발생 (블로킹 안 함)
            while not input_queue.empty():
                try:
                    input_queue.get_nowait()
                except queue.Empty:
                    break
            continue

        # 처리 시작 플래그 설정
        is_processing = True

        try:
            # 실제 질문 처리 및 답변 생성
            result = rag.answer_question(question)

            # 결과 출력
            print("\n" + "=" * 50)
            print("🤖 답변:")
            print(result["answer"])

            print(f"\n📚 참조 문서 ({len(result['sources'])}개):")
            for i, source in enumerate(result["sources"]):
                print(f"  {i+1}. {source}")

            print(f"\n⏱️ 응답 시간: {result['response_time']:.2f}초")

        finally:
            # 처리 완료 플래그 해제 (try-finally로 예외 발생해도 반드시 실행)
            is_processing = False
            # 처리 중에 쌓인 중복 입력들을 모두 제거
            while not input_queue.empty():
                try:
                    input_queue.get_nowait()
                except queue.Empty:
                    break

# 프로그램 진입점
# 이 파일이 직접 실행될 때만 main() 함수 호출
# 다른 파일에서 import할 때는 실행되지 않음
if __name__ == "__main__":
    main()