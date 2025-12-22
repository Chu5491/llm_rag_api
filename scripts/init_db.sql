-- 1. pgvector 확장 활성화
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. 시스템 전역 설정 관리 테이블 (configurations)
-- DROP TABLE IF EXISTS public.configurations;
CREATE TABLE public.configurations (
    id serial4 NOT NULL,
    embedding_model varchar(255) DEFAULT 'bona/bge-m3-korean'::character varying NULL, -- 사용할 임베딩 모델명
    llm_model varchar(255) NULL,                                                       -- 사용할 LLM 모델명
    rag_tc_count int4 DEFAULT 20 NULL,                                                -- 생성할 테스트케이스 기본 개수
    rag_batch_size int4 DEFAULT 3 NULL,                                               -- RAG 배치 처리 크기
    rag_tc_id_prefix varchar(50) DEFAULT 'REQ_TC'::character varying NULL,            -- 테스트케이스 ID 접두어
    figma_enabled bool DEFAULT true NULL,                                             -- 피그마 기능 활성화 여부
    created_at timestamptz DEFAULT CURRENT_TIMESTAMP NULL,
    updated_at timestamptz DEFAULT CURRENT_TIMESTAMP NULL,
    CONSTRAINT configurations_pkey PRIMARY KEY (id)
);

COMMENT ON TABLE public.configurations IS '시스템 운영 및 RAG 관련 전역 설정 저장';

-- 3. RAG 통합 임베딩 테이블 (rag_embeddings)
-- DROP TABLE IF EXISTS public.rag_embeddings;
CREATE TABLE public.rag_embeddings (
    id serial4 NOT NULL,
    project_id int4 DEFAULT 1 NOT NULL,      -- 프로젝트 구분용 ID (기본값 1)
    source_type varchar(50) NOT NULL,        -- 데이터 소스 타입 ('file', 'figma' 등)
    "text" text NOT NULL,                    -- 원문 텍스트 청크
    embedding public.vector NULL,            -- 벡터 데이터 (차원은 모델에 따라 자동 결정)
    metadata_json json NULL,                 -- 소스 파일명, 페이지 정보 등 메타데이터
    created_at timestamp DEFAULT CURRENT_TIMESTAMP NULL,
    CONSTRAINT rag_embeddings_pkey PRIMARY KEY (id)
);

-- 검색 성능 향상을 위한 인덱스 설정
CREATE INDEX ix_rag_embeddings_project_id ON public.rag_embeddings USING btree (project_id);
CREATE INDEX ix_rag_embeddings_source_type ON public.rag_embeddings USING btree (source_type);

-- 벡터 유사도 검색(코사인 거리)을 위한 HNSW 인덱스 (대규모 데이터 검색 성능 최적화)
CREATE INDEX rag_embeddings_embedding_idx ON public.rag_embeddings USING hnsw (embedding vector_cosine_ops);

COMMENT ON TABLE public.rag_embeddings IS '파일 및 피그마 소스에서 추출된 텍스트와 벡터 데이터';

-- 4. 초기 환경 설정값 주입
INSERT INTO configurations (
    id,
    embedding_model,
    llm_model,
    rag_tc_count,
    rag_batch_size,
    rag_tc_id_prefix,
    figma_enabled
)
VALUES (
    1,
    'bona/bge-m3-korean',
    'exaone3.5:2.4b',
    20,
    3,
    'REQ_TC',
    TRUE
)
ON CONFLICT (id) DO NOTHING;

-- 5. 테스트케이스 자동 생성 이력 테이블 (generate_history)
CREATE TABLE public.generate_history (
    id serial4 NOT NULL,
    project_id int4 DEFAULT 1 NOT NULL,
    title varchar(255) NOT NULL,                    -- 작업 제목
    source_type varchar(50) NOT NULL,               -- 'file', 'figma'
    status varchar(20) NOT NULL DEFAULT 'running',  -- 'running', 'success', 'failed'
    summary text NULL,                              -- 작업 요약

    total_batches int4 DEFAULT 0 NULL,              -- 총 배치 수
    current_batch int4 DEFAULT 0 NULL,              -- 현재 처리 중인 배치 번호
    progress int4 DEFAULT 0 NULL,                   -- 진행률 (0~100)

    started_at timestamptz DEFAULT CURRENT_TIMESTAMP NULL,
    finished_at timestamptz NULL,
    duration interval NULL,                         -- 소요 시간

    logs text NULL,                                 -- 실행 로그 (상세 내역)
    result_data jsonb NULL,                         -- 생성된 결과물 (JSON)
    model_name varchar(100) NULL,                  -- 사용된 모델

    CONSTRAINT generate_history_pkey PRIMARY KEY (id)
);

CREATE INDEX ix_generate_history_status ON public.generate_history (status);
CREATE INDEX ix_generate_history_started_at ON public.generate_history (started_at DESC);

COMMENT ON TABLE public.generate_history IS '테스트케이스 자동 생성 실행 이력 및 실시간 진행 상태';
