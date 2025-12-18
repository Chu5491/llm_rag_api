from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime


# Figma 프로젝트 정보 스키마
class FigmaProjectInfo(BaseModel):
    team_id: str
    project_id: str


# 프로젝트 내 파일 정보 스키마
class FigmaFile(BaseModel):
    key: str
    name: str
    thumbnail_url: Optional[str] = None
    last_modified: datetime
    branch_data: Optional[dict] = None  # branch_data가 true일 경우 포함


# 프로젝트 파일 목록 응답 스키마
class FigmaProjectFilesResponse(BaseModel):
    name: str
    files: List[FigmaFile]
    cursor: Optional[dict] = None  # 페이지네이션을 위한 커서


# Figma 문서 트리의 노드 (Frame, Component, Group, Text 등 공통)
class FigmaNode(BaseModel):
    """
    Figma API 스펙 전체를 다 넣으면 너무 커지니까
    RAG에 필요한 최소 필드만 잡아둔 버전.
    """

    id: str
    name: str
    type: str
    visible: Optional[bool] = None
    children: Optional[List["FigmaNode"]] = None  # 재귀 참조

    # 필요하면 아래 같은 필드를 점점 추가해가면 됨
    # characters: Optional[str] = None   # TEXT 노드일 때 실제 텍스트
    # description: Optional[str] = None
    # etc...


# 컴포넌트 메타데이터 스키마
class FigmaComponentMeta(BaseModel):
    key: str
    name: str
    description: Optional[str] = None
    remote: Optional[bool] = None


# 스타일 메타데이터 스키마
class FigmaStyleMeta(BaseModel):
    key: str
    name: str
    style_type: str = Field(alias="styleType")
    description: Optional[str] = None


# 파일 상세 정보 스키마
class FigmaFileDetail(BaseModel):
    """
    GET /v1/files/{file_key} 응답 매핑
    - 문서 전체 트리(document)
    - 컴포넌트/스타일 메타
    """

    name: str
    last_modified: datetime = Field(alias="lastModified")
    thumbnail_url: Optional[str] = Field(default=None, alias="thumbnailUrl")
    version: Optional[str] = None
    schema_version: Optional[int] = Field(default=None, alias="schemaVersion")

    document: FigmaNode  # 전체 문서 트리 루트
    components: Dict[str, FigmaComponentMeta] = {}
    styles: Dict[str, FigmaStyleMeta] = {}


# LLM이 이해하기 위한 최소 정보
class FigmaControlSummary(BaseModel):
    """
    LLM이 '무슨 인터랙션 요소가 있는지' 이해하기 위한 최소 정보.
    - 버튼, 네비게이션 링크 등
    """

    type: str  # "button", "nav_link", "input" 등 추론한 역할
    label: Optional[str] = (
        None  # 사용자가 보게 되는 텍스트 (ex. "reserve your spot", "Home")
    )
    node_id: Optional[str] = None  # 원본 FigmaNode.id
    path: Optional[str] = None  # "홈 / Desktop / Main / Button A" 같은 계층 경로
    component_name: Optional[str] = None  # Button A, Nav 등 컴포넌트 이름(있으면)


class FigmaSectionSummary(BaseModel):
    """
    한 화면(Screen) 안의 논리적 섹션 단위 요약.
    - Hero, Main, Call to action section, Footer, Nav 등
    - 섹션에 포함된 텍스트와 인터랙션 요소를 묶어서 전달.
    """

    name: str  # 섹션 이름 (ex. "Hero headline section")
    node_id: Optional[str] = None  # 섹션의 FigmaNode.id
    path: str  # "홈 / Desktop / Hero headline section" 같은 전체 경로
    texts: List[str] = []  # 섹션 내부 TEXT 노드의 characters만 모은 리스트
    controls: List[FigmaControlSummary] = []  # 섹션 안 버튼/링크 등


class FigmaScreenSummary(BaseModel):
    """
    LLM에게 넘길 '한 화면(페이지+디바이스)' 단위 요약.
    - RAG 인덱스에서 보통 이 단위로 쪼개서 저장/검색.
    """

    file_name: str  # FigmaFileDetail.name
    page_name: str  # SECTION 이름 ("/reserve", "/about", "홈" 등)
    variant: str  # "Desktop", "Tablet", "Mobile" 등
    screen_id: str  # 해당 FRAME의 FigmaNode.id
    screen_path: str  # "홈 / Desktop", "/reserve / Mobile" 등 상위 계층 경로
    sections: List[FigmaSectionSummary]  # 이 화면에 포함된 섹션들의 요약


class FigmaFileLLMSummary(BaseModel):
    """
    하나의 Figma 파일(FigmaFileDetail)에서 LLM/RAG에 필요한 부분만 추린 최종 요약 구조.
    - 실제 RAG 인덱싱 시에는 보통 screens 단위로 다시 쪼개서 저장.
    """

    file_name: str  # FigmaFileDetail.name
    last_modified: datetime  # FigmaFileDetail.last_modified
    screens: List[FigmaScreenSummary]


# 재귀 참조 해소
FigmaNode.model_rebuild()
