import httpx
from pydantic import ValidationError

from app.core.config import get_settings
from app.schemas.figma import (
    FigmaProjectFilesResponse,
    FigmaFileDetail,
    FigmaFileLLMSummary,
    FigmaScreenSummary,
    FigmaSectionSummary,
)

from app.core.logging import get_logger

logger = get_logger(__name__)


class FigmaClient:
    """Figma API 클라이언트"""

    def __init__(self):
        self.settings = get_settings()
        self.base_url = "https://api.figma.com/v1"

    def _get_token(self) -> str:
        """DB에서 최신 Figma API 토큰을 가져옵니다."""
        from app.db.database import VectorSessionLocal
        from app.crud.config import get_app_config

        try:
            with VectorSessionLocal() as db:
                config = get_app_config(db)
                if config and config.figma_api_token:
                    return config.figma_api_token
        except Exception as e:
            logger.error(f"DB에서 Figma 토큰을 가져오는데 실패했습니다: {e}")
        return ""

    @property
    def headers(self) -> dict:
        """실시간 토큰을 포함한 헤더를 반환합니다."""
        return {"X-Figma-Token": self._get_token()}

    async def get_project_files(
        self, project_id: str, branch_data: bool = False
    ) -> FigmaProjectFilesResponse:
        """프로젝트의 파일 목록을 가져옵니다."""
        url = f"{self.base_url}/projects/{project_id}/files"
        params = {"branch_data": str(branch_data).lower()}

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, headers=self.headers, params=params)
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                status = e.response.status_code

                # 에러 바디 파싱 시도 (JSON / 텍스트 둘 다 대비)
                try:
                    err_body = e.response.json()
                except ValueError:
                    err_body = e.response.text

                if status in (401, 403):
                    logger.error(
                        "Figma 인증/권한 오류: status=%s, body=%s",
                        status,
                        err_body,
                    )
                elif status == 404:
                    logger.error(
                        "Figma 프로젝트/팀을 찾을 수 없음: project_id=%s, body=%s",
                        project_id,
                        err_body,
                    )
                else:
                    logger.error(
                        "Figma API 오류: status=%s, body=%s",
                        status,
                        err_body,
                    )

                # 여기서는 일단 그대로 다시 던져서 router 쪽에서 HTTPException 으로 변환
                raise

            except httpx.RequestError as e:
                # DNS / 타임아웃 / 네트워크 계열 오류
                logger.error("Figma API 네트워크 오류: %s", repr(e))
                # 필요하면 커스텀 예외로 감싸서 올려도 됨
                raise

        # 여기까지 왔으면 status 2xx
        try:
            data = response.json()
            return FigmaProjectFilesResponse(**data)
        except ValidationError as e:
            # Figma 응답 스키마가 우리가 기대한 것과 다를 때
            logger.error("Figma 프로젝트 파일 응답 파싱 실패: %s", e)
            logger.error("원본 응답: %s", response.text)
            raise

    async def get_file(
        self,
        file_key: str,
        depth: int | None = None,
        ids: list[str] | None = None,
        geometry: str | None = None,
    ) -> FigmaFileDetail:
        """
        단일 파일 내용(document 트리)을 가져옵니다.
        - file_key: Figma 파일 키
        - depth: (선택) 트리 깊이 제한
        - ids: (선택) 특정 노드만 가져오고 싶을 때 노드 id 리스트
        - geometry: (선택) 'paths' 등 geometry 옵션
        """
        url = f"{self.base_url}/files/{file_key}"

        params: dict = {}
        if depth is not None:
            params["depth"] = depth
        if ids:
            # Figma는 ids=ID1,ID2,ID3 형태로 받음
            params["ids"] = ",".join(ids)
        if geometry:
            params["geometry"] = geometry

        async with httpx.AsyncClient() as client:
            response = await client.get(
                url, headers=self.headers, params=params or None
            )
            response.raise_for_status()
            return FigmaFileDetail(**response.json())

    def get_file_llm_summary(self, detail: FigmaFileDetail) -> FigmaFileLLMSummary:
        """
        FigmaFileDetail(document 트리 전체)에서
        LLM/RAG에 필요한 화면/섹션/텍스트/컨트롤 정보만 뽑아서
        FigmaFileLLMSummary로 변환한다.

        ** 조건은 Figma 시스템 속성(type, children)만 사용 **
        ** 사용자 입력(name)은 조건에 사용하지 않음 **
        """

        screens: list[FigmaScreenSummary] = []

        doc = detail.document
        if not doc.children:
            return FigmaFileLLMSummary(
                file_name=detail.name,
                last_modified=detail.last_modified,
                screens=[],
            )

        # ========================================
        # 조건에 사용하는 것: Figma 시스템 속성만!
        # ========================================
        # - type: CANVAS, FRAME, SECTION, TEXT, INSTANCE, GROUP 등
        # - children: 자식 노드 존재 여부
        # - id: 노드 고유 ID
        #
        # 조건에 사용하지 않는 것: 사용자 입력!
        # - name: 사용자가 직접 입력한 레이어 이름
        # ========================================

        def collect_all_texts(node, path: str) -> list[str]:
            """노드와 모든 하위에서 TEXT 타입 노드의 name(내용)을 수집"""
            texts: list[str] = []

            if node.type == "TEXT":
                # TEXT의 name은 실제 텍스트 내용 (사용자가 입력한 "글자")
                if node.name:
                    texts.append(node.name)

            for child in node.children or []:
                child_path = f"{path} / {child.name}"
                texts.extend(collect_all_texts(child, child_path))

            return texts

        def find_screens_recursive(node, page_name: str, depth: int = 0):
            """
            재귀적으로 모든 FRAME을 탐색하여 화면 후보 수집
            조건: type == FRAME (name은 조건에 사용하지 않음)
            """
            found_screens = []

            # FRAME 타입이고, 자식이 있으면 화면 후보로 간주
            if node.type == "FRAME" and node.children:
                screen_path = f"{page_name} / {node.name}"
                sections: list[FigmaSectionSummary] = []

                # 화면 바로 아래의 FRAME/INSTANCE/GROUP을 "섹션"으로 취급
                # 조건: type만 체크
                for section_node in node.children or []:
                    if section_node.type not in {"FRAME", "INSTANCE", "GROUP"}:
                        continue

                    section_path = f"{screen_path} / {section_node.name}"

                    # 섹션 내 모든 TEXT 수집
                    texts = collect_all_texts(section_node, section_path)

                    # 텍스트가 있는 섹션만 저장
                    if texts:
                        sections.append(
                            FigmaSectionSummary(
                                name=section_node.name,
                                node_id=section_node.id,
                                path=section_path,
                                texts=texts,
                                controls=[],  # 컨트롤 분류는 name 기반이므로 제거
                            )
                        )

                # 섹션이 있으면 화면으로 저장
                if sections:
                    found_screens.append(
                        FigmaScreenSummary(
                            file_name=detail.name,
                            page_name=page_name,
                            variant=node.name,  # name은 저장용으로만 사용 (조건X)
                            screen_id=node.id,
                            screen_path=screen_path,
                            sections=sections,
                        )
                    )
            else:
                # FRAME이 아니면 자식들 계속 탐색
                for child in node.children or []:
                    found_screens.extend(
                        find_screens_recursive(child, page_name, depth + 1)
                    )

            return found_screens

        # 1) 최상위에서 CANVAS 찾기 (조건: type만)
        for top in doc.children or []:
            if top.type == "CANVAS":
                page_name = top.name  # name은 저장용으로만 사용

                # CANVAS 하위에서 화면 찾기
                for child in top.children or []:
                    # SECTION이나 FRAME 타입의 자식들 탐색
                    if child.type in {"SECTION", "FRAME"}:
                        # SECTION인 경우: 그 안에서 FRAME(화면) 찾기
                        if child.type == "SECTION":
                            section_page_name = f"{page_name} / {child.name}"
                            for section_child in child.children or []:
                                found = find_screens_recursive(
                                    section_child, section_page_name
                                )
                                screens.extend(found)
                        else:
                            # FRAME인 경우: 바로 화면 후보
                            found = find_screens_recursive(child, page_name)
                            screens.extend(found)

        # 2) 화면을 못 찾았으면 최상위 FRAME들을 화면으로 간주 (Fallback)
        if not screens:
            logger.info("화면을 찾지 못해 최상위 노드에서 직접 텍스트를 수집합니다.")
            for top in doc.children or []:
                if top.type == "CANVAS":
                    page_name = top.name
                    # CANVAS의 모든 텍스트 수집
                    all_texts = collect_all_texts(top, page_name)

                    if all_texts:
                        screens.append(
                            FigmaScreenSummary(
                                file_name=detail.name,
                                page_name=page_name,
                                variant="default",
                                screen_id=top.id,
                                screen_path=page_name,
                                sections=[
                                    FigmaSectionSummary(
                                        name="all_content",
                                        node_id=top.id,
                                        path=page_name,
                                        texts=all_texts,
                                        controls=[],
                                    )
                                ],
                            )
                        )

        return FigmaFileLLMSummary(
            file_name=detail.name,
            last_modified=detail.last_modified,
            screens=screens,
        )
