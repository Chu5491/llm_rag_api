import logging
import sys

def setup_logging():
    """애플리케이션 로깅 설정 (터미널에만 출력)"""
    
    # 포맷터 정의
    formatter = logging.Formatter(
        fmt='[%(asctime)s] %(levelname)-8s [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 루트 로거 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # 콘솔 핸들러만 추가
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # 써드파티 라이브러리 로그 레벨 조정
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("faiss").setLevel(logging.WARNING)
    logging.getLogger("ollama").setLevel(logging.INFO)

def get_logger(name: str) -> logging.Logger:
    """모듈별 로거 획득"""
    return logging.getLogger(name)