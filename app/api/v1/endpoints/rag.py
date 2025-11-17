# FastAPI 관련 임포트
from fastapi import APIRouter, Body, Depends, HTTPException
from starlette import status
from starlette.concurrency import run_in_threadpool

import time
import httpx

from app.api.deps import get_app_settings
from app.core.config import Settings
from app.schemas.rag import (
	RagBuildRequest,
	RagBuildResponse,
	RagQueryRequest,
	RagQueryResponse,
	RagStatusResponse,
)
from app.services.rag import rag_manager
from app.utils.ollama_client import OllamaClient

router = APIRouter(prefix="/rag", tags=["rag"])


@router.get("/status", response_model=RagStatusResponse)
async def rag_status():
	# 인덱스가 없으면 가능하면 로드
	if not rag_manager.vector_store:
		try:
			await run_in_threadpool(rag_manager.ensure_vector_store)
		except Exception:
			pass

	status_payload = rag_manager.status()
	return RagStatusResponse(**status_payload)


@router.post("/build", response_model=RagBuildResponse)
async def build_rag(body: RagBuildRequest = Body(...)):
	start_time = time.monotonic()
	try:
		chunk_count = await run_in_threadpool(
			rag_manager.ensure_vector_store,
			body.force,
			body.embedding_model,
		)
	except (FileNotFoundError, ValueError) as exc:
		raise HTTPException(
			status_code=status.HTTP_400_BAD_REQUEST,
			detail=str(exc)
		)
	except Exception as exc:
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail=str(exc)
		)
	duration = time.monotonic() - start_time
	return RagBuildResponse(
		success=True,
		chunk_count=chunk_count,
		message="벡터 색인 생성 완료",
		duration=duration,
	)


@router.post("/query", response_model=RagQueryResponse)
async def query_rag(
	body: RagQueryRequest,
	settings: Settings = Depends(get_app_settings),
):
	question = (body.question or "").strip()
	if not question:
		raise HTTPException(
			status_code=status.HTTP_400_BAD_REQUEST,
			detail="질문을 입력해주세요."
		)

	start_time = time.monotonic()

	try:
		await run_in_threadpool(rag_manager.ensure_vector_store)
	except Exception as exc:
		raise HTTPException(
			status_code=status.HTTP_400_BAD_REQUEST,
			detail=f"벡터 DB 준비 실패: {exc}"
		)

	try:
		docs, mode_config = await run_in_threadpool(
			rag_manager.retrieve_documents,
			question,
			body.search_mode,
		)
	except RuntimeError as exc:
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail=str(exc)
		)

	if not docs:
		raise HTTPException(
			status_code=status.HTTP_404_NOT_FOUND,
			detail="동일한 문서에서 내용을 가져올 수 없습니다."
		)

	context, source_previews = rag_manager.compose_context(docs)
	if not context:
		raise HTTPException(
			status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
			detail="문서에서 추출할 수 있는 텍스트가 부족합니다."
		)

	prompt = rag_manager.build_prompt(context, question)
	client = OllamaClient(settings)

	try:
		generated = await client.generate(
			prompt=prompt,
			model=body.model or rag_manager.default_ollama_model,
			stream=False,
			options={"temperature": 0.1, "top_p": 0.9, "max_new_tokens": 1024},
		)
	except httpx.HTTPError as exc:
		raise HTTPException(
			status_code=status.HTTP_502_BAD_GATEWAY,
			detail=f"Ollama 통신 오류: {exc}"
		)

	output = generated.get("response") or generated.get("output") or generated.get("result") or ""
	elapsed = time.monotonic() - start_time
	model_used = body.model or rag_manager.default_ollama_model

	return RagQueryResponse(
		success=True,
		answer=output,
		sources=source_previews,
		search_mode=mode_config.get("mode", "balanced"),
		model=model_used,
		response_time=elapsed,
		raw_response=generated,
		context=context,
	)
