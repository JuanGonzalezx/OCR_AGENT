import asyncio
import logging
import time
import uuid
from functools import partial

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .ocr_service import extract_invoice
from .schemas import ExtractResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

MAGIC_BYTES = {
    "jpeg": b"\xff\xd8\xff",
    "png": b"\x89PNG\r\n\x1a\n",
}

app = FastAPI(
    title="Iguana Café OCR Service",
    description="Extracción de datos de facturas colombianas con Gemini Vision",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _validate_magic_bytes(data: bytes, content_type: str) -> bool:
    if content_type in ("image/jpeg", "image/jpg"):
        return data[:3] == MAGIC_BYTES["jpeg"]
    if content_type == "image/png":
        return data[:8] == MAGIC_BYTES["png"]
    if content_type == "image/webp":
        return data[:4] == b"RIFF" and data[8:12] == b"WEBP"
    return False


@app.get("/")
def root():
    return {"service": "iguana-cafe-ocr", "version": "1.0.0", "status": "running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ocr/extract", response_model=ExtractResponse)
async def ocr_extract(file: UploadFile = File(...)):
    request_id = str(uuid.uuid4())
    start_time = time.time()

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de archivo no soportado: {file.content_type}. Use JPG, PNG o WEBP.",
        )

    image_bytes = await file.read()

    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Imagen excede el tamaño máximo de 10MB ({len(image_bytes)} bytes recibidos).",
        )

    if not _validate_magic_bytes(image_bytes, file.content_type):
        raise HTTPException(
            status_code=400,
            detail="El archivo no corresponde al tipo de imagen declarado (magic bytes inválidos).",
        )

    logger.info("[%s] Procesando %s — %d bytes", request_id, file.filename, len(image_bytes))

    try:
        loop = asyncio.get_event_loop()
        extract_func = partial(extract_invoice, image_bytes, file.content_type)
        data = await asyncio.wait_for(
            loop.run_in_executor(None, extract_func),
            timeout=30.0,
        )
    except asyncio.TimeoutError:
        elapsed = int((time.time() - start_time) * 1000)
        logger.error("[%s] TIMEOUT %dms — %s", request_id, elapsed, file.filename)
        raise HTTPException(status_code=504, detail="Timeout: Gemini no respondió en 30 segundos.")
    except ValueError as exc:
        elapsed = int((time.time() - start_time) * 1000)
        logger.warning("[%s] UNPROCESSABLE %dms — %s: %s", request_id, elapsed, file.filename, exc)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        elapsed = int((time.time() - start_time) * 1000)
        logger.error("[%s] ERROR %dms — %s: %s", request_id, elapsed, file.filename, exc)
        raise HTTPException(status_code=502, detail=f"Error al procesar con Gemini: {exc}")

    processing_time_ms = int((time.time() - start_time) * 1000)
    logger.info(
        "[%s] OK %dms — %s (%d bytes) confidence=%.2f",
        request_id,
        processing_time_ms,
        file.filename,
        len(image_bytes),
        data.confidence or 0.0,
    )

    return ExtractResponse(
        request_id=request_id,
        processing_time_ms=processing_time_ms,
        data=data,
    )
