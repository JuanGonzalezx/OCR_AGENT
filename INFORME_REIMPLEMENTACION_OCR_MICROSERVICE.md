# Guía de réplica fiel — Instancia Gemini + lectura imagen/documento

Este documento está pensado para que **otra IA o un desarrollador** reproduzca el comportamiento del repo **sin adivinar**: cada módulo incluye **rol**, **contrato de uso**, **dependencias** y **código literal** tomado del proyecto.

**Alcance:** `gemini_api_key_context.py`, `llm_factory.py`, `extract_invoice_info.py` (tool imagen + structured output), fragmento de `schemas.py` (request/response del analizador), `dynamic_analyzer.py` completo (PDF/imagen vía SDK nativo + parseo y normalización). Sin voz/TTS.

**Modelo fijo en código:** `gemini-2.5-flash`.

**Grafo de dependencias (imports):**

```
tools/extract_invoice_info.py  -->  core/llm_factory.py  -->  utils/gemini_api_key_context.py
tools/dynamic_analyzer.py      -->  utils/gemini_api_key_context.py
tools/dynamic_analyzer.py      -->  services/schemas.py (DynamicInvoiceAnalyzer*)
tools/dynamic_analyzer.py      -->  google.genai
```

**Layout:** los imports asumen que el **working directory / PYTHONPATH** incluye la carpeta `src/` (donde viven `core`, `utils`, `tools`, `services`). Si mueves archivos, ajusta `from utils...`, `from core...`, `from services...`.

---

## Dependencias pip

Mínimo para réplica (alineado a `requirements-heavy.txt`):

- `langchain-google-genai>=2.0.0,<3`
- `langchain-core>=0.3.17,<0.4`
- `google-genai>=1.0.0`
- `pydantic>=2.0,<3`

---

## Réplica mínima sin gateway MCP

El repo exige `bind_gemini_api_key` antes del LLM. Para pruebas:

1. Llama `bind_gemini_api_key(os.environ["GOOGLE_API_KEY"])` al inicio del handler, **o**
2. Sustituye `get_effective_gemini_api_key_for_client()` por lectura directa de variable de entorno (cambio **no** presente en el repo original).

---

## Módulo 1 — `src/utils/gemini_api_key_context.py`

### Rol
Guardar la API key de Gemini **por contexto de ejecución** (`ContextVar`), típicamente una petición HTTP.

### Contrato
- Antes de cualquier `ChatGoogleGenerativeAI` o `genai.Client`: `bind_gemini_api_key(key)`.
- Tras la petición: `reset_gemini_api_key_context(token)` con el `Token` devuelto por `bind`.
- Si no hay key en contexto: `get_effective_gemini_api_key_for_client()` lanza `ValueError` con el mensaje documentado en código.

### Código

```python
"""Contexto por petición: API key de Gemini resuelta vía MCP Auth (integración Gemini)."""
from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Optional

_gemini_api_key_ctx: ContextVar[Optional[str]] = ContextVar("mcp_gemini_api_key", default=None)


def get_request_gemini_api_key() -> Optional[str]:
    return _gemini_api_key_ctx.get()


def bind_gemini_api_key(key: str) -> Token:
    """Asocia la API key a la petición actual; devuelve token para reset()."""
    return _gemini_api_key_ctx.set((key or "").strip() or None)


def reset_gemini_api_key_context(ctx_token: Token) -> None:
    _gemini_api_key_ctx.reset(ctx_token)


def get_effective_gemini_api_key_for_client() -> str:
    """
    API key de Gemini solo desde el contexto de petición (TokenGemini vía MCP Auth).
    Debe haberse llamado a bind_gemini_api_key en la ruta antes de invocar al LLM.
    """
    k = get_request_gemini_api_key()
    if k:
        return k
    raise ValueError(
        "No hay API key de Gemini en contexto. La petición debe incluir autenticación MCP "
        "(JWT + company_database_id) y la integración Gemini debe exponer TokenGemini."
    )
```

## Módulo 2 — `src/core/llm_factory.py`

### Rol
Fabricar **`ChatGoogleGenerativeAI`** para visión multimodal (imagen en mensaje), con la misma key que el resto del servicio.

### Parámetros exactos del constructor (no inventar)
- `model="gemini-2.5-flash"`
- `google_api_key=get_effective_gemini_api_key_for_client()`
- `temperature=temperature` (argumento de función, default `0` en firma)
- `max_retries=2`

### Código

```python
from langchain_google_genai import ChatGoogleGenerativeAI

from utils.gemini_api_key_context import get_effective_gemini_api_key_for_client


def get_vision_model(temperature: float = 0):
    """Retorna una instancia configurada de Gemini para visión."""
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  # Estándar actual
        google_api_key=get_effective_gemini_api_key_for_client(),
        temperature=temperature,
        max_retries=2,
    )
```

## Módulo 3 — `src/tools/extract_invoice_info.py`

### Rol
Tool LangChain **`ocr_extract_invoice`**: entrada `image_base64`, salida dict con `status` y `data` o `message`.

### Detalles críticos para réplica fiel
1. `get_vision_model(temperature=0)` → `with_structured_output(InvoiceData)`.
2. `HumanMessage` con dos partes: texto de instrucciones (literal en código) + `image_url` con **data-URI** `data:image/jpeg;base64,{image_base64}` (el MIME está **fijado a jpeg** en el string).
3. Post-procesamiento de `items`: si elementos son `str`, regex `r'(.+?)\s*[-:]\s*(\d+[.,]?\d*)$'` y `float` del precio con `replace(',', '').replace('.', '')`.
4. Imports no usados en el archivo original: `List`, `Union`, `validator` — se dejan como en el repo si la réplica es byte-a-byte.

### Código

```python
from typing import List, Union, Any
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field, validator
from core.llm_factory import get_vision_model
import re

# ==========================================
# 1. ESQUEMAS AJUSTADOS (TOLERANTES A FALLOS)
# ==========================================

class InvoiceLineItem(BaseModel):
    description: str = Field(description="Descripcion del producto o servicio")
    quantity: float = Field(description="Cantidad", default=1.0)
    total_price: float = Field(description="Precio total del item")

class InvoiceData(BaseModel):
    """Informacion extraida de la factura."""
    merchant_name: str = Field(description="Nombre del proveedor o comercio.")
    invoice_number: str = Field(description="Folio o numero de factura.", default="N/A")
    date: str = Field(description="Fecha YYYY-MM-DD.")
    currency: str = Field(description="Codigo de moneda", default="COP")
    total_amount: float = Field(description="Total final a pagar.")
    
    # CAMBIO CLAVE: Aceptamos Any para procesarlo manualmente si el LLM falla
    items: Any = Field(description="Lista de items detallados.")

# ==========================================
# 2. TOOL CON LÓGICA DE REPARACIÓN
# ==========================================

@tool
def ocr_extract_invoice(image_base64: str) -> dict:
    """
    [OCR] Extrae datos estructurados de una FACTURA/RECIBO en imagen.
    """
    if not image_base64:
        return {"status": "error", "message": "Imagen vacía."}

    try:
        llm = get_vision_model(temperature=0)
        structured_llm = llm.with_structured_output(InvoiceData)

        # Prompt ajustado para evitar la ambigüedad de strings
        system_instructions = """
        Eres un experto en extracción de facturas. Extrae los datos en formato JSON estricto.
        
        SOBRE LOS ITEMS:
        Debes extraer una LISTA DE OBJETOS. NO extraigas strings simples.
        CORRECTO: [{"description": "Coca Cola", "total_price": 5000, "quantity": 1}]
        INCORRECTO: ["Coca Cola - 5000"]

        Si la imagen es borrosa o difícil, haz tu mejor esfuerzo.
        """

        msg = HumanMessage(
            content=[
                {"type": "text", "text": system_instructions},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
            ]
        )

        # Invocamos al LLM
        result: InvoiceData = structured_llm.invoke([msg])
        
        # --- LÓGICA DE AUTO-REPARACIÓN DE ITEMS ---
        # Si el LLM devolvió strings en lugar de objetos (el error que tienes), lo arreglamos aquí.
        raw_items = result.items
        fixed_items = []

        if isinstance(raw_items, list):
            for item in raw_items:
                if isinstance(item, str):
                    # El LLM mandó "LIMONADA - 25000". Intentamos separarlo.
                    # Buscamos el último número en la cadena
                    match = re.search(r'(.+?)\s*[-:]\s*(\d+[.,]?\d*)$', item)
                    if match:
                        desc = match.group(1).strip()
                        price = float(match.group(2).replace(',', '').replace('.', '')) # Ajuste simple
                        fixed_items.append({"description": desc, "total_price": price, "quantity": 1.0})
                    else:
                        # Fallback: Ponemos todo en descripción
                        fixed_items.append({"description": item, "total_price": 0.0, "quantity": 1.0})
                elif isinstance(item, dict):
                     fixed_items.append(item)
                elif hasattr(item, 'model_dump'): # Es un objeto Pydantic
                     fixed_items.append(item.model_dump())
        
        # Reemplazamos los items reparados en el resultado final
        final_data = result.model_dump()
        final_data['items'] = fixed_items

        return {
            "status": "success",
            "data": final_data
        }

    except Exception as e:
        return {"status": "error", "message": f"Error procesando factura: {str(e)}"}
```

## Módulo 4 — `src/services/schemas.py` (fragmento requerido por `dynamic_analyzer.py`)

### Rol
- `DynamicInvoiceAnalyzerRequest`: valida `pdf_base64` (Base64, máx **10 MiB** decodificado, magic bytes PDF/JPEG/PNG, PDF debe contener `%%EOF`).
- `DynamicInvoiceAnalyzerResponse`: sobre `success`, `result`, `error`, `metadata`.

### Código
El bloque siguiente antecede explícitamente los imports mínimos que faltan respecto al archivo monolítico del repo (`schemas.py` mezcla muchas clases). Para un módulo nuevo solo con el analizador dinámico, basta:

```python
from typing import Optional
from pydantic import BaseModel, Field, field_validator

import base64
import binascii
from pydantic import field_validator

class DynamicInvoiceAnalyzerRequest(BaseModel):
    """
    Schema para la tool de análisis dinámico de PDFs.
    Permite inyectar system prompts y queries personalizadas.
    """
    pdf_base64: str = Field(
        ...,
        description="PDF o imagen (JPEG/PNG) codificada en Base64"
    )
    custom_system_prompt: str = Field(
        ...,
        description="Instrucciones de comportamiento para el LLM"
    )
    user_query: str = Field(
        ...,
        description="Pregunta específica sobre el documento"
    )
    temperature: float = Field(
        default=0,
        ge=0,
        le=1,
        description="Temperatura del modelo (0=determinista, 1=creativo)"
    )
    token: Optional[str] = Field(
        None,
        description="JWT obligatorio (body o header); la clave de Gemini es solo TokenGemini (integración Gemini).",
    )
    company_database_id: Optional[str] = Field(
        None,
        description="ID de base de datos de compañía obligatorio (body o header X-Company-Database-Id).",
    )

    @field_validator('pdf_base64')
    @classmethod
    def validate_base64_document(cls, v: str) -> str:
        """
        Validación: Base64 válido; contenido debe ser PDF, JPEG o PNG (magic bytes).
        Tamaño máximo: 10MB.
        """
        if v is None or (isinstance(v, str) and not v.strip()):
            raise ValueError(
                "pdf_base64 es obligatorio y no puede estar vacío. "
                "Incluya el PDF/imagen en Base64 dentro de payload o, en POST /execute, "
                "use el campo pdf_base64 en la raíz del JSON (junto a tool_name y payload)."
            )
        try:
            decoded = base64.b64decode(v, validate=True)
            max_size = 10 * 1024 * 1024
            if len(decoded) == 0:
                raise ValueError(
                    "pdf_base64 decodifica a 0 bytes (cadena vacía o Base64 inválido). "
                    "Envíe el documento en Base64 o use pdf_base64 en la raíz del JSON de /execute."
                )
            if len(decoded) > max_size:
                raise ValueError(f"El archivo excede {max_size / 1024 / 1024}MB")

            # PDF
            if decoded.startswith(b'%PDF-'):
                if b'%%EOF' not in decoded:
                    raise ValueError("PDF corrupto: falta marcador EOF")
                return v
            # JPEG
            if decoded.startswith(b'\xff\xd8\xff'):
                return v
            # PNG
            if decoded.startswith(b'\x89PNG\r\n\x1a\n'):
                return v

            raise ValueError(
                "El archivo no es un PDF ni una imagen JPEG/PNG válida (magic bytes incorrectos)"
            )

        except ValueError:
            raise
        except binascii.Error:
            raise ValueError("El string no es Base64 válido")
        except Exception as e:
            raise ValueError(f"Error validando documento: {str(e)}")


class DynamicInvoiceAnalyzerResponse(BaseModel):
    """Schema de respuesta con manejo de errores estructurado."""
    success: bool
    result: dict | str | None = Field(
        None,
        description="Objeto JSON canónico parseado para el orquestador (document_type, fields, line_items, tables, confidence)."
    )
    error: str | None = Field(
        None,
        description="Mensaje de error si success=False (validación PDF, etc.)"
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Info adicional: tokens, tamaño PDF, modelo usado"
    )
```

## Módulo 5 — `src/tools/dynamic_analyzer.py` (archivo completo)

### Rol
1. **Validar** documento Base64 (`_decode_and_validate_document` + magic bytes + EOF PDF).
2. **Instanciar** `genai.Client(api_key=get_effective_gemini_api_key_for_client())`.
3. **Llamar** `client.models.generate_content` con `model="gemini-2.5-flash"`, `Part.from_bytes` + `Part.from_text`, `GenerateContentConfig(temperature=...)`.
4. **Parsear** texto del modelo → JSON string limpio (`_clean_json_response` y auxiliares).
5. **Normalizar** a contrato canónico (`_normalize_canonical_result` y toda la cadena de helpers en la clase).

### Detalles que otra IA no debe cambiar sin motivo
- Orden en `contents`: primero bytes del documento, luego texto del prompt.
- Texto del prompt: `_build_prompt_text` usa el separador de líneas `═` tal como está en el código fuente.
- `_clean_json_response` devuelve **string JSON**; `execute` hace `json.loads` y pasa el dict a `_normalize_canonical_result`.
- `_is_rate_limit_error` anota `BaseException` (built-in).
- `GenAIClientError` puede quedar `None` si el SDK no define el símbolo.
- `@tool` al final: nombre de función **`dynamic_invoice_analyzer`** (nombre de tool LangChain).

### Código

```python
"""
Dynamic Invoice Analyzer Tool
==============================
Tool completamente stateless para análisis de PDFs con prompts personalizados.

GARANTÍAS DE DISEÑO:
- ✅ Cero persistencia de memoria entre llamadas
- ✅ Instanciación on-demand del LLM
- ✅ Validación robusta de PDFs
- ✅ Manejo de errores estructurado
- ✅ result SIEMPRE es JSON válido o mensaje de error; nunca volcado del archivo
"""

import base64
import hashlib
import json
import logging
import re
import threading
import time
from datetime import datetime
from typing import Dict, Any

from google import genai
from google.genai import types

# Excepciones del SDK (pueden no existir en todas las versiones)
try:
    from google.genai.errors import ClientError as GenAIClientError
except ImportError:
    GenAIClientError = None

from utils.gemini_api_key_context import get_effective_gemini_api_key_for_client
from services.schemas import (
    DynamicInvoiceAnalyzerRequest,
    DynamicInvoiceAnalyzerResponse
)

logger = logging.getLogger("uvicorn")

_CANONICAL_FIELD_KEYS = (
    "invoice_number",
    "supplier_name",
    "supplier_nit",
    "supplier_address",
    "issue_date",
    "issue_time",
    "currency",
    "subtotal",
    "tax_total",
    "discount_total",
    "total",
    "payment_method",
    "payment_brand",
    "authorization_code",
    "customer_id",
    "customer_name",
    "customer_address",
    "customer_phone",
    "salesperson",
    "pos_number",
    "notes",
)

_CANONICAL_LINE_ITEM_KEYS = (
    "description",
    "quantity",
    "unit_price",
    "line_total",
)

# Campos adicionales por línea que pasan tal cual al cliente (Legalisapp / contrato ocrFactura).
_LINE_ITEM_PASSTHROUGH = (
    "impuesto",
    "expense_category",
    "EntryTipoGto",
    "Campo1",
    "Campo2",
    "Campo3",
)

# Contador de peticiones en vuelo (saturación / concurrencia)
_in_flight = 0
_in_flight_lock = threading.Lock()


def _request_id(pdf_base64: str | None) -> str:
    """Identificador de petición para correlacionar logs."""
    raw = (pdf_base64[:200] if pdf_base64 else "") + str(time.time())
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def _truncate_for_log(text: str | None, max_len: int = 1500) -> str:
    """Texto seguro para una línea de log (sin saltos largos)."""
    if text is None:
        return ""
    s = str(text).replace("\r", " ").replace("\n", "↵ ")
    if len(s) > max_len:
        return s[:max_len] + f"... [truncado total_chars={len(text)}]"
    return s


def _log_response_structure(request_id: str, response: DynamicInvoiceAnalyzerResponse) -> None:
    """Loguea la estructura de la respuesta y un preview de lo que devuelve la IA al cliente."""
    try:
        d = response.model_dump() if hasattr(response, "model_dump") else {}
        top_keys = list(d.keys()) if isinstance(d, dict) else []
        logger.info(
            "dynamic_invoice_analyzer respuesta request_id=%s type=dict top_keys=%s success=%s",
            request_id,
            top_keys,
            d.get("success"),
        )
        result = d.get("result")
        if result is None:
            logger.warning("dynamic_invoice_analyzer request_id=%s [NO HAY result]", request_id)
            return
        if isinstance(result, dict):
            inner_keys = list(result.keys())
            logger.info("dynamic_invoice_analyzer request_id=%s result_keys=%s", request_id, inner_keys)
            # Errores de parseo / JSON no válido (no confundir con error=null del contrato data/error)
            err = result.get("error")
            raw = result.get("raw")
            if (isinstance(err, str) and err) or (raw is not None and raw != ""):
                logger.info(
                    "dynamic_invoice_analyzer IA devolvió error/fragmento request_id=%s error=%s raw_preview=%s",
                    request_id,
                    _truncate_for_log(str(err) if err else "", 400),
                    _truncate_for_log(str(raw) if raw is not None else "", 1200),
                )
            for k in ("numAtCard", "docDate", "date", "itemCode", "cufe", "price", "finalCustom", "comentarios"):
                if k in result and result[k] is not None:
                    val = str(result[k])[:50]
                    logger.info("dynamic_invoice_analyzer request_id=%s %s=%s", request_id, k, val)
            # Preview genérico del JSON result (útil cuando hay data anidada)
            if not ((isinstance(err, str) and err) or (raw is not None and raw != "")):
                try:
                    preview = json.dumps(result, ensure_ascii=False)[:2000]
                    logger.info(
                        "dynamic_invoice_analyzer result_preview request_id=%s %s",
                        request_id,
                        _truncate_for_log(preview, 2000),
                    )
                except Exception:
                    pass
        else:
            logger.info(
                "dynamic_invoice_analyzer request_id=%s result_type=%s preview=%s",
                request_id,
                type(result).__name__,
                _truncate_for_log(str(result), 1200),
            )
    except Exception as e:
        logger.warning("dynamic_invoice_analyzer log_response_structure request_id=%s error=%s", request_id, e)


def _is_rate_limit_error(err: BaseException) -> bool:
    """True si el error parece 429 / quota / resource exhausted."""
    s = (str(err) + getattr(err, "message", "")).lower()
    return any(
        x in s for x in ("429", "rate limit", "quota", "resource_exhausted")
    )


def _log_gemini_error(request_id: str, err: BaseException, duration_ms: float) -> None:
    """Centraliza el log de errores de Gemini (429 vs otros)."""
    err_str = str(err)[:500]
    err_type = type(err).__name__
    if _is_rate_limit_error(err):
        logger.warning(
            "dynamic_invoice_analyzer Gemini 429/rate_limit request_id=%s duration_ms=%.0f body_truncated=%s",
            request_id,
            duration_ms,
            err_str,
        )
    else:
        logger.error(
            "dynamic_invoice_analyzer Gemini error request_id=%s duration_ms=%.0f type=%s message=%s",
            request_id,
            duration_ms,
            err_type,
            err_str,
            exc_info=True,
        )


def _response_has_data_result(response: DynamicInvoiceAnalyzerResponse | None) -> bool:
    """True si la respuesta tiene data.result (para el orquestador: data.result presente y no vacío)."""
    if response is None:
        return False
    try:
        d = response.model_dump() if hasattr(response, "model_dump") else {}
        result = d.get("result")
        if result is None:
            return False
        if isinstance(result, dict) and result.get("error"):
            return False
        return True
    except Exception:
        return False


def _looks_like_base64_or_binary(s: str) -> bool:
    """True si el string parece base64/documento binario, no JSON."""
    if not s or len(s) < 80:
        return False
    s_stripped = s.strip()
    if s_stripped.startswith("{"):
        return False
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=\n\r\t ")
    return sum(1 for c in s_stripped[:500] if c in allowed) > 450


class DynamicInvoiceAnalyzer:
    """
    Analizador dinámico de PDFs sin formato fijo.
    Cada ejecución es completamente aislada.
    """

    @staticmethod
    def _try_json_loads(candidate: str) -> Dict[str, Any] | None:
        """Intenta parsear un string a dict JSON."""
        if not candidate or not isinstance(candidate, str):
            return None
        s = candidate.strip()
        if not s:
            return None
        try:
            out = json.loads(s)
            return out if isinstance(out, dict) else {"_parsed": out}
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _extract_fenced_blocks(content: str) -> list[str]:
        """Extrae cuerpos de bloques ```json ... ``` o ``` ... ```."""
        blocks: list[str] = []
        for m in re.finditer(r"```(?:json)?\s*([\s\S]*?)```", content, re.IGNORECASE):
            b = (m.group(1) or "").strip()
            if b:
                blocks.append(b)
        return blocks

    @staticmethod
    def _extract_balanced_json_candidates(content: str) -> list[str]:
        """
        Extrae substrings con llaves balanceadas { ... } respetando strings JSON.
        Útil cuando el modelo agrega texto antes/después del objeto JSON.
        """
        candidates: list[str] = []
        start_idx: int | None = None
        depth = 0
        in_string = False
        escaped = False

        for i, ch in enumerate(content):
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
                continue

            if ch == "{":
                if depth == 0:
                    start_idx = i
                depth += 1
            elif ch == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start_idx is not None:
                        candidate = content[start_idx : i + 1].strip()
                        if candidate:
                            candidates.append(candidate)
                        start_idx = None
        return candidates

    @staticmethod
    def _score_json_candidate(data: Dict[str, Any]) -> int:
        """Prioriza candidatos con estructura esperada del OCR."""
        score = 0
        if "data" in data:
            score += 5
        if "document_type" in data:
            score += 4
        if "fields" in data:
            score += 4
        if "line_items" in data or "items" in data or "products" in data:
            score += 3
        if any(k in data for k in ("dynamic_analyzer", "dynamic_analizer")):
            score += 3
        if isinstance(data.get("data"), dict):
            inner = data["data"]
            if "document_type" in inner:
                score += 4
            if "fields" in inner:
                score += 4
            if "line_items" in inner or "items" in inner or "products" in inner:
                score += 3
            if any(k in inner for k in ("dynamic_analyzer", "dynamic_analizer")):
                score += 3
        return score

    @staticmethod
    def _build_canonical_result_template() -> Dict[str, Any]:
        return {
            "document_type": "other",
            "fields": {k: None for k in _CANONICAL_FIELD_KEYS},
            "line_items": [],
            "tables": [],
            "confidence": None,
        }

    @staticmethod
    def _normalize_number(value: Any) -> float | int | None:
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return value
        if not isinstance(value, str):
            return None
        s = value.strip()
        if not s:
            return None
        s = re.sub(r"[^\d,.\-]", "", s)
        if not s:
            return None
        if "," in s and "." in s:
            if s.rfind(",") > s.rfind("."):
                s = s.replace(".", "").replace(",", ".")
            else:
                s = s.replace(",", "")
        elif "," in s:
            s = s.replace(".", "").replace(",", ".")
        try:
            n = float(s)
            return int(n) if n.is_integer() else n
        except Exception:
            return None

    @staticmethod
    def _normalize_money_number(value: Any, currency_hint: Any = None) -> float | int | None:
        """
        Normaliza montos monetarios con heurísticas LATAM (COP/USD).
        Evita interpretar miles con punto como decimales (ej: 28.400 -> 28400).
        """
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value) if value.is_integer() else value
        if not isinstance(value, str):
            return None

        s = value.strip()
        if not s:
            return None
        s = re.sub(r"[^\d,.\-]", "", s)
        if not s:
            return None

        # Caso típico COP con miles por punto: 28.400 / 1.234.567
        if "." in s and "," not in s and re.match(r"^-?\d{1,3}(?:\.\d{3})+$", s):
            try:
                return int(s.replace(".", ""))
            except Exception:
                return None

        # Caso miles por coma: 1,234,567
        if "," in s and "." not in s and re.match(r"^-?\d{1,3}(?:,\d{3})+$", s):
            try:
                return int(s.replace(",", ""))
            except Exception:
                return None

        n = DynamicInvoiceAnalyzer._normalize_number(s)
        if n is None:
            return None

        # Ajuste leve: si parece COP y viene float entero, devolver int.
        cur = str(currency_hint or "").upper()
        if cur == "COP" and isinstance(n, float) and n.is_integer():
            return int(n)
        return n

    @staticmethod
    def _normalize_date(value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        s = value.strip()
        if not s:
            return None
        s = s.replace("/", "-").replace(".", "-")
        for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d-%m-%y", "%Y%m%d"):
            try:
                return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
            except Exception:
                pass
        m = re.match(r"^(\d{1,2})-(\d{1,2})-(\d{4})$", s)
        if m:
            d, mo, y = m.groups()
            try:
                return datetime(int(y), int(mo), int(d)).strftime("%Y-%m-%d")
            except Exception:
                return None
        return None

    @staticmethod
    def _normalize_time(value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        s = value.strip()
        if not s:
            return None
        m = re.search(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", s)
        if not m:
            return None
        h = int(m.group(1))
        mi = int(m.group(2))
        sec = int(m.group(3) or "0")
        if h > 23 or mi > 59 or sec > 59:
            return None
        return f"{h:02d}:{mi:02d}:{sec:02d}"

    @staticmethod
    def _normalize_document_type(value: Any) -> str:
        if not isinstance(value, str):
            return "other"
        s = value.strip().lower()
        if s in ("invoice", "receipt", "other"):
            return s
        return "invoice" if "invoice" in s or "factura" in s else ("receipt" if "receipt" in s or "recibo" in s or "ticket" in s else "other")

    @staticmethod
    def _pick_first(source: Dict[str, Any], keys: tuple[str, ...]) -> Any:
        for k in keys:
            if k in source and source[k] not in ("", None):
                return source[k]
        return None

    @staticmethod
    def _pick_first_casefold(source: Dict[str, Any], keys_lower: tuple[str, ...]) -> Any:
        """Busca clave ignorando mayúsculas (p. ej. Producto vs producto)."""
        if not source:
            return None
        lowered = {str(k).casefold(): v for k, v in source.items()}
        for want in keys_lower:
            if want in lowered and lowered[want] not in ("", None):
                return lowered[want]
        return None

    @staticmethod
    def _flatten_nested_invoice_blob(src_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Muchos prompts devuelven `{ "factura": { "documento", "gastos", ... } }`.
        Aplana esas claves al mismo nivel que el resto para que el mapeo canónico las encuentre.
        """
        if not isinstance(src_data, dict):
            return src_data
        out = dict(src_data)
        for nest_key in ("factura", "invoice", "Factura"):
            blob = out.get(nest_key)
            if isinstance(blob, dict) and blob:
                out = {**blob, **out}
                break
        return out

    @staticmethod
    def _resolve_line_items_list(src_data: Dict[str, Any]) -> list[Any] | None:
        """
        El modelo a veces devuelve ítems bajo otras claves (typos o español).
        """
        if not isinstance(src_data, dict):
            return None
        for key in (
            "line_items",
            "items",
            "products",
            "gastos",
            "conceptos",
            "dynamic_analyzer",
            "dynamic_analizer",  # typo frecuente del modelo
            "detalle_productos",
            "productos",
            "lineas",
        ):
            val = src_data.get(key)
            if isinstance(val, list) and val:
                return val
        # Primer array de objetos que parezca detalle de factura
        for _k, val in src_data.items():
            if not isinstance(val, list) or len(val) < 1:
                continue
            if not isinstance(val[0], dict):
                continue
            row0 = val[0]
            if DynamicInvoiceAnalyzer._pick_first(row0, ("Producto", "product_name", "description")) is not None:
                return val
            if DynamicInvoiceAnalyzer._pick_first_casefold(
                row0, ("producto", "descripcion", "cant.", "cantidad", "valor unit.", "valor total")
            ) is not None:
                return val
        return None

    @staticmethod
    def _normalize_line_item(raw: Dict[str, Any]) -> Dict[str, Any]:
        out = {k: None for k in _CANONICAL_LINE_ITEM_KEYS}
        out["description"] = DynamicInvoiceAnalyzer._pick_first(
            raw,
            ("description", "name", "item", "detalle", "producto", "Producto", "product_name", "Notas", "notas"),
        ) or DynamicInvoiceAnalyzer._pick_first_casefold(raw, ("producto", "descripcion", "articulo", "notas"))
        out["quantity"] = DynamicInvoiceAnalyzer._normalize_number(
            DynamicInvoiceAnalyzer._pick_first(raw, ("quantity", "qty", "cantidad", "Cant.", "Cant"))
            or DynamicInvoiceAnalyzer._pick_first_casefold(raw, ("cant.", "cant", "cantidad"))
        )
        out["unit_price"] = DynamicInvoiceAnalyzer._normalize_money_number(
            DynamicInvoiceAnalyzer._pick_first(
                raw,
                ("unit_price", "price", "precio_unitario", "valor_unitario", "unit_value", "Valor Unit."),
            )
            or DynamicInvoiceAnalyzer._pick_first_casefold(raw, ("valor unit.", "valor unit", "precio unitario"))
        )
        out["line_total"] = DynamicInvoiceAnalyzer._normalize_money_number(
            DynamicInvoiceAnalyzer._pick_first(
                raw,
                ("line_total", "total", "importe", "subtotal", "total_value", "Valor Total", "Valor", "valor"),
            )
            or DynamicInvoiceAnalyzer._pick_first_casefold(raw, ("valor total", "importe total", "valor"))
        )
        # Inferencia suave para completar faltantes sin romper estructura.
        if out["line_total"] is None and out["quantity"] is not None and out["unit_price"] is not None:
            out["line_total"] = DynamicInvoiceAnalyzer._normalize_money_number(out["quantity"] * out["unit_price"])
        if out["unit_price"] is None and out["line_total"] is not None and out["quantity"] not in (None, 0):
            out["unit_price"] = DynamicInvoiceAnalyzer._normalize_money_number(out["line_total"] / out["quantity"])
        DynamicInvoiceAnalyzer._merge_line_item_passthrough(out, raw)
        return out

    @staticmethod
    def _merge_line_item_passthrough(out: Dict[str, Any], raw: Dict[str, Any]) -> None:
        """Copia impuesto, categoría, tipo SAP y campos libres sin reescribir el núcleo canónico."""
        if not isinstance(raw, dict):
            return
        tax_val = DynamicInvoiceAnalyzer._pick_first(
            raw, ("Impuesto", "impuesto", "iva", "IVA", "tax_rate", "tax_percentage")
        )
        if tax_val is not None and tax_val != "":
            out["impuesto"] = str(tax_val).strip() if not isinstance(tax_val, (int, float)) else str(tax_val)
        cat = DynamicInvoiceAnalyzer._pick_first(
            raw, ("expense_category", "tipo_gasto_categoria", "categoria_gasto", "category")
        )
        if cat is not None and str(cat).strip() != "":
            out["expense_category"] = str(cat).strip()
        tipo = DynamicInvoiceAnalyzer._pick_first(raw, ("EntryTipoGto", "entry_tipo_gto", "expense_type_id"))
        if tipo is not None and tipo != "":
            if isinstance(tipo, (int, float)):
                out["EntryTipoGto"] = int(tipo)
            else:
                try:
                    out["EntryTipoGto"] = int(str(tipo).strip())
                except ValueError:
                    out["EntryTipoGto"] = tipo
        for ck in ("Campo1", "Campo2", "Campo3"):
            if ck in raw and raw[ck] not in (None, ""):
                out[ck] = raw[ck]

    @staticmethod
    def _build_header_fields_table(fields: Dict[str, Any]) -> Dict[str, Any]:
        rows = []
        for key, value in fields.items():
            if value is not None:
                rows.append({"field": key, "value": value})
        return {
            "name": "header_fields",
            "columns": ["field", "value"],
            "rows": rows,
        }

    @staticmethod
    def _build_line_items_table(line_items: list[Dict[str, Any]]) -> Dict[str, Any]:
        base_cols = ["description", "quantity", "unit_price", "line_total"]
        extra_cols = [k for k in _LINE_ITEM_PASSTHROUGH if any(isinstance(it, dict) and it.get(k) not in (None, "") for it in line_items)]
        columns = base_cols + extra_cols
        rows = []
        for item in line_items:
            if not isinstance(item, dict):
                continue
            row = {c: item.get(c) for c in columns}
            rows.append(row)
        return {
            "name": "line_items",
            "columns": columns,
            "rows": rows,
        }

    @staticmethod
    def _build_canonical_tables(
        fields: Dict[str, Any],
        line_items: list[Dict[str, Any]],
        existing_tables: Any,
    ) -> list[Dict[str, Any]]:
        out_tables: list[Dict[str, Any]] = []
        out_tables.append(DynamicInvoiceAnalyzer._build_header_fields_table(fields))
        out_tables.append(DynamicInvoiceAnalyzer._build_line_items_table(line_items))

        if isinstance(existing_tables, list):
            for t in existing_tables:
                if not isinstance(t, dict):
                    continue
                name = str(t.get("name") or "").strip().lower()
                if name in ("header_fields", "line_items"):
                    continue
                out_tables.append(t)
        return out_tables

    @staticmethod
    def _normalize_canonical_result(parsed: Dict[str, Any]) -> Dict[str, Any]:
        canonical = DynamicInvoiceAnalyzer._build_canonical_result_template()
        src = parsed if isinstance(parsed, dict) else {}

        # Soporta respuesta ya en contrato data/error o contrato plano
        if isinstance(src.get("data"), dict):
            src_data = DynamicInvoiceAnalyzer._flatten_nested_invoice_blob(src.get("data") or {})
            canonical["document_type"] = DynamicInvoiceAnalyzer._normalize_document_type(src_data.get("document_type"))
            fields_src = src_data.get("fields") if isinstance(src_data.get("fields"), dict) else {}
            line_items_src = src_data.get("line_items")
            if not isinstance(line_items_src, list) or not line_items_src:
                line_items_src = DynamicInvoiceAnalyzer._resolve_line_items_list(src_data)
            tables_src = src_data.get("tables")
            confidence_src = src_data.get("confidence")
        else:
            src_data = DynamicInvoiceAnalyzer._flatten_nested_invoice_blob(src)
            canonical["document_type"] = DynamicInvoiceAnalyzer._normalize_document_type(src.get("document_type"))
            fields_src = src.get("fields") if isinstance(src.get("fields"), dict) else src_data
            line_items_src = src.get("line_items") or src.get("items") or src.get("products")
            if not isinstance(line_items_src, list) or not line_items_src:
                line_items_src = DynamicInvoiceAnalyzer._resolve_line_items_list(src_data)
            tables_src = src.get("tables")
            confidence_src = src.get("confidence")

        seller = src_data.get("informacion_vendedor") if isinstance(src_data.get("informacion_vendedor"), dict) else {}
        customer = src_data.get("informacion_cliente") if isinstance(src_data.get("informacion_cliente"), dict) else {}
        txn = src_data.get("detalles_transaccion") if isinstance(src_data.get("detalles_transaccion"), dict) else {}
        totals = src_data.get("resumen_totales") if isinstance(src_data.get("resumen_totales"), dict) else {}
        payment = src_data.get("metodo_pago") if isinstance(src_data.get("metodo_pago"), dict) else {}
        company_info = src_data.get("company_info") if isinstance(src_data.get("company_info"), dict) else {}
        invoice_details = src_data.get("invoice_details") if isinstance(src_data.get("invoice_details"), dict) else {}
        client_info = src_data.get("client_info") if isinstance(src_data.get("client_info"), dict) else {}
        invoice_header = src_data.get("invoice_header") if isinstance(src_data.get("invoice_header"), dict) else {}
        client_information = src_data.get("client_information") if isinstance(src_data.get("client_information"), dict) else {}
        totals_info = src_data.get("totals") if isinstance(src_data.get("totals"), dict) else {}
        payment_info = src_data.get("payment_info") if isinstance(src_data.get("payment_info"), dict) else {}
        summary = src_data.get("summary") if isinstance(src_data.get("summary"), dict) else {}

        def pick(*values: Any) -> Any:
            for v in values:
                if v not in ("", None):
                    return v
            return None

        _gastos = src_data.get("gastos")
        _looks_like_invoice = isinstance(_gastos, list) and len(_gastos) > 0
        # No usar canonical["document_type"] aquí (suele ser "other" por defecto y bloquea "invoice" con gastos).
        canonical["document_type"] = DynamicInvoiceAnalyzer._normalize_document_type(
            pick(
                src_data.get("document_type"),
                "invoice" if _looks_like_invoice else None,
                invoice_details.get("type"),
                invoice_header.get("invoice_type"),
            )
        )

        fields = canonical["fields"]
        # Esquema frecuente CO / Legalisapp: documento = NIT emisor, referencia = número de factura.
        fields["invoice_number"] = pick(
            fields_src.get("invoice_number"),
            src_data.get("invoice_number"),
            src_data.get("referencia"),
            src_data.get("numero_factura"),
            txn.get("invoice_number"),
            invoice_details.get("invoice_number"),
            invoice_header.get("invoice_number"),
            src_data.get("numAtCard"),
            src_data.get("documento"),
        )
        fields["supplier_name"] = pick(fields_src.get("supplier_name"), src_data.get("supplier_name"), seller.get("name"), company_info.get("name"), invoice_header.get("company_name"), src_data.get("merchant_name"))
        fields["supplier_nit"] = pick(
            fields_src.get("supplier_nit"),
            src_data.get("supplier_nit"),
            src_data.get("documento"),
            seller.get("nit"),
            company_info.get("nit"),
            invoice_header.get("nit"),
            src_data.get("nit"),
            src_data.get("cufe"),
        )
        fields["supplier_address"] = pick(fields_src.get("supplier_address"), src_data.get("supplier_address"), seller.get("address"), seller.get("direccion"), company_info.get("address"), invoice_header.get("store_address"))
        fields["issue_date"] = DynamicInvoiceAnalyzer._normalize_date(
            pick(
                fields_src.get("issue_date"),
                src_data.get("issue_date"),
                src_data.get("fecha"),
                txn.get("date"),
                invoice_details.get("date"),
                invoice_header.get("date"),
                src_data.get("docDate"),
                src_data.get("date"),
            )
        )
        fields["issue_time"] = DynamicInvoiceAnalyzer._normalize_time(pick(fields_src.get("issue_time"), src_data.get("issue_time"), txn.get("time"), invoice_details.get("time"), invoice_header.get("time"), src_data.get("time")))
        fields["currency"] = pick(fields_src.get("currency"), src_data.get("currency"), src_data.get("moneda"), txn.get("currency"), totals_info.get("currency"))
        fields["subtotal"] = DynamicInvoiceAnalyzer._normalize_money_number(pick(fields_src.get("subtotal"), src_data.get("subtotal"), totals.get("subtotal"), totals_info.get("subtotal"), summary.get("subtotal"), src_data.get("subtotal_amount")), fields.get("currency"))
        fields["tax_total"] = DynamicInvoiceAnalyzer._normalize_money_number(pick(fields_src.get("tax_total"), src_data.get("tax_total"), totals.get("tax_total"), totals_info.get("tax_total"), totals_info.get("iva"), summary.get("tax_total"), src_data.get("iva_total")), fields.get("currency"))
        fields["discount_total"] = DynamicInvoiceAnalyzer._normalize_money_number(pick(fields_src.get("discount_total"), src_data.get("discount_total"), totals.get("discount_total"), totals_info.get("discount_total"), summary.get("discount_total"), totals.get("descuento")), fields.get("currency"))
        fields["total"] = DynamicInvoiceAnalyzer._normalize_money_number(
            pick(
                fields_src.get("total"),
                src_data.get("total"),
                src_data.get("totalFact"),
                src_data.get("total_factura"),
                totals.get("total"),
                totals_info.get("total"),
                totals_info.get("grand_total"),
                summary.get("total"),
                src_data.get("total_amount"),
                src_data.get("finalCustom"),
            ),
            fields.get("currency"),
        )
        fields["payment_method"] = pick(fields_src.get("payment_method"), src_data.get("payment_method"), payment.get("method"), payment_info.get("method"), payment_info.get("payment_method"))
        fields["payment_brand"] = pick(fields_src.get("payment_brand"), src_data.get("payment_brand"), payment.get("brand"), payment.get("franquicia"), payment_info.get("brand"), payment_info.get("franchise"))
        fields["authorization_code"] = pick(fields_src.get("authorization_code"), src_data.get("authorization_code"), payment.get("authorization_code"), payment_info.get("authorization_code"), payment_info.get("approval_code"), src_data.get("authorization_code"))
        fields["customer_id"] = pick(fields_src.get("customer_id"), src_data.get("customer_id"), customer.get("id"), customer.get("nit"), client_info.get("id_number"), client_info.get("document"), client_information.get("client_id"))
        fields["customer_name"] = pick(fields_src.get("customer_name"), src_data.get("customer_name"), customer.get("name"), client_info.get("name"), client_information.get("client_name"))
        fields["customer_address"] = pick(fields_src.get("customer_address"), src_data.get("customer_address"), customer.get("address"), customer.get("direccion"), client_info.get("address"), client_information.get("client_address"))
        fields["customer_phone"] = pick(fields_src.get("customer_phone"), src_data.get("customer_phone"), customer.get("phone"), customer.get("telefono"), client_info.get("phone"), client_information.get("client_phone"))
        fields["salesperson"] = pick(fields_src.get("salesperson"), src_data.get("salesperson"), src_data.get("vendedor"))
        fields["pos_number"] = pick(fields_src.get("pos_number"), src_data.get("pos_number"), src_data.get("caja"), src_data.get("punto_venta"))
        fields["notes"] = pick(
            fields_src.get("notes"),
            src_data.get("notes"),
            src_data.get("comentarios"),
            src_data.get("observaciones"),
            company_info.get("fiscal_responsibility"),
            invoice_header.get("fiscal_responsibility"),
            invoice_details.get("type"),
            invoice_header.get("invoice_type"),
        )

        if isinstance(line_items_src, list):
            canonical["line_items"] = [
                DynamicInvoiceAnalyzer._normalize_line_item(i)
                for i in line_items_src
                if isinstance(i, dict)
            ]
        else:
            canonical["line_items"] = []

        # Fallback monetario best-effort usando items cuando el header viene incompleto.
        item_sum = 0
        has_item_sum = False
        for it in canonical["line_items"]:
            lt = it.get("line_total") if isinstance(it, dict) else None
            if isinstance(lt, (int, float)):
                item_sum += lt
                has_item_sum = True
        if fields.get("total") is None and has_item_sum:
            fields["total"] = DynamicInvoiceAnalyzer._normalize_money_number(item_sum, fields.get("currency"))
        if fields.get("subtotal") is None and has_item_sum:
            fields["subtotal"] = DynamicInvoiceAnalyzer._normalize_money_number(item_sum, fields.get("currency"))

        canonical["tables"] = DynamicInvoiceAnalyzer._build_canonical_tables(
            fields=fields,
            line_items=canonical["line_items"],
            existing_tables=tables_src,
        )
        canonical["confidence"] = DynamicInvoiceAnalyzer._normalize_number(confidence_src)
        DynamicInvoiceAnalyzer._attach_factura_gastos_aliases(canonical, fields)
        return canonical

    @staticmethod
    def _attach_factura_gastos_aliases(canonical: Dict[str, Any], fields: Dict[str, Any]) -> None:
        """
        Añade factura, gastos y totalFact en paralelo al contrato canónico (p. ej. ocrFactura / patch por raíz).
        """
        nit = fields.get("supplier_nit")
        ref = fields.get("invoice_number")
        fecha = fields.get("issue_date")
        moneda = fields.get("currency")
        if any(x not in (None, "") for x in (nit, ref, fecha, moneda)):
            factura: Dict[str, Any] = {}
            if nit not in (None, ""):
                factura["documento"] = nit
            if ref not in (None, ""):
                factura["referencia"] = ref
            if fecha not in (None, ""):
                factura["fecha"] = fecha
            if moneda not in (None, ""):
                factura["moneda"] = moneda
            if factura:
                canonical["factura"] = factura
        total = fields.get("total")
        if total not in (None, ""):
            canonical["totalFact"] = total
        items = canonical.get("line_items") or []
        if not isinstance(items, list) or not items:
            return
        gastos: list[Dict[str, Any]] = []
        for li in items:
            if not isinstance(li, dict):
                continue
            desc = li.get("description")
            val = li.get("line_total")
            if (desc is None or str(desc).strip() == "") and val in (None, 0, ""):
                continue
            row: Dict[str, Any] = {"descripcion": desc, "valor": val}
            if li.get("impuesto") not in (None, ""):
                row["impuesto"] = li["impuesto"]
            if li.get("expense_category") not in (None, ""):
                row["expense_category"] = li["expense_category"]
            if li.get("EntryTipoGto") is not None and li.get("EntryTipoGto") != "":
                row["EntryTipoGto"] = li["EntryTipoGto"]
            for ck in ("Campo1", "Campo2", "Campo3"):
                if li.get(ck) not in (None, ""):
                    row[ck] = li[ck]
            gastos.append(row)
        if gastos:
            canonical["gastos"] = gastos

    @staticmethod
    def _clean_json_response(content: str) -> str:
        """
        Obtiene un objeto JSON serializable a partir del texto del LLM.

        Muchos modelos devuelven Markdown (tablas, títulos) en lugar de JSON puro;
        el orquestador espera un dict parseable. Orden de intentos:
        1) Bloques ```json ... ``` / ``` ... ```
        2) Primer objeto { ... } en el texto
        3) Fallback estructurado (no pierde el texto; el orquestador puede leer parse_status / data)
        """
        if not content or not isinstance(content, str):
            return json.dumps({"error": "Contenido vacío"})

        original = content
        content = content.strip()

        # 1) Bloques de código (donde el modelo suele poner JSON si lo genera)
        for block in DynamicInvoiceAnalyzer._extract_fenced_blocks(content):
            data = DynamicInvoiceAnalyzer._try_json_loads(block)
            if data is not None:
                return json.dumps(data, ensure_ascii=False)
            # A veces el bloque tiene texto antes del JSON
            start_idx = block.find("{")
            end_idx = block.rfind("}")
            if start_idx != -1 and end_idx >= start_idx:
                data = DynamicInvoiceAnalyzer._try_json_loads(block[start_idx : end_idx + 1])
                if data is not None:
                    return json.dumps(data, ensure_ascii=False)

        best_candidate: Dict[str, Any] | None = None
        best_score = -1

        # 2) Candidatos por llaves balanceadas
        for candidate in DynamicInvoiceAnalyzer._extract_balanced_json_candidates(content):
            data = DynamicInvoiceAnalyzer._try_json_loads(candidate)
            if data is None:
                continue
            score = DynamicInvoiceAnalyzer._score_json_candidate(data)
            if score > best_score:
                best_score = score
                best_candidate = data

        if best_candidate is not None:
            return json.dumps(best_candidate, ensure_ascii=False)

        # 3) Primer '{' … último '}' en todo el texto
        start_idx = content.find("{")
        end_idx = content.rfind("}")
        if start_idx != -1 and end_idx >= start_idx:
            json_str = content[start_idx : end_idx + 1]
            data = DynamicInvoiceAnalyzer._try_json_loads(json_str)
            if data is not None:
                return json.dumps(data, ensure_ascii=False)
            json_str = json_str.replace("```json", "").replace("```", "").strip()
            data = DynamicInvoiceAnalyzer._try_json_loads(json_str)
            if data is not None:
                return json.dumps(data, ensure_ascii=False)

        # 4) No hay JSON parseable: retorna contrato canónico best-effort en nulls, sin fallar parseo del orquestador.
        return json.dumps(DynamicInvoiceAnalyzer._build_canonical_result_template(), ensure_ascii=False)
    @staticmethod
    def _detect_media_type(data: bytes) -> str:
        """Detecta MIME type por magic bytes. PDF, JPEG o PNG."""
        if data.startswith(b'%PDF-'):
            return "application/pdf"
        if data.startswith(b'\xff\xd8\xff'):
            return "image/jpeg"
        if data.startswith(b'\x89PNG\r\n\x1a\n'):
            return "image/png"
        raise ValueError("Tipo no soportado: debe ser PDF, JPEG o PNG (magic bytes)")

    @staticmethod
    def _decode_and_validate_document(b64: str) -> tuple[bytes, str]:
        """
        Decodifica Base64 y valida que sea PDF o imagen (JPEG/PNG).
        Returns:
            (bytes, mime_type) para enviar a Gemini.
        """
        try:
            decoded = base64.b64decode(b64, validate=True)
            if not decoded:
                raise ValueError("Contenido vacío")
            mime = DynamicInvoiceAnalyzer._detect_media_type(decoded)
            if mime == "application/pdf" and b'%%EOF' not in decoded:
                raise ValueError("PDF corrupto: falta marcador EOF")
            return (decoded, mime)
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Error decodificando documento: {str(e)}") from e

    @staticmethod
    def _build_prompt_text(system_prompt: str, user_query: str) -> str:
        """Texto del prompt (system + consulta) para enviar junto al documento o imagen."""
        return f"""
{system_prompt}

═══════════════════════════════════════════════════════════════════════════════

CONSULTA SOBRE EL DOCUMENTO (responde solo JSON según el contrato anterior):
{user_query}

"""

    @staticmethod
    def _extract_token_usage_from_genai(response: Any) -> Dict[str, int]:
        """Extrae uso de tokens de la respuesta de la API nativa de Google GenAI."""
        usage = getattr(response, "usage_metadata", None) or {}
        def _get(o, *keys):
            for k in keys:
                v = getattr(o, k, None)
                if v is not None:
                    return int(v)
                if isinstance(o, dict) and k in o:
                    return int(o[k])
            return 0
        return {
            "prompt_tokens": _get(usage, "prompt_token_count", "prompt_tokens"),
            "completion_tokens": _get(usage, "candidates_token_count", "completion_token_count", "completion_tokens"),
            "total_tokens": _get(usage, "total_token_count", "total_tokens"),
        }

    @staticmethod
    def execute(request: DynamicInvoiceAnalyzerRequest) -> DynamicInvoiceAnalyzerResponse:
        """
        Punto de entrada principal para la tool.

        FLUJO:
        1. Decodificar y validar PDF
        2. Crear instancia FRESCA del LLM (sin memoria previa)
        3. Construir prompt multimodal
        4. Invocar modelo
        5. Retornar resultado estructurado

        Args:
            request: Objeto validado con los parámetros de entrada

        Returns:
            DynamicInvoiceAnalyzerResponse: Resultado del análisis o error
        """
        request_id = _request_id(getattr(request, "pdf_base64", None))
        pdf_len = len(request.pdf_base64) if request.pdf_base64 else 0
        logger.info(
            "dynamic_invoice_analyzer entrada request_id=%s pdf_base64_len=%s",
            request_id,
            pdf_len if pdf_len else "missing",
        )

        with _in_flight_lock:
            global _in_flight
            _in_flight += 1
            in_flight_now = _in_flight
        logger.info("dynamic_invoice_analyzer concurrencia request_id=%s in_flight=%s", request_id, in_flight_now)

        start_total = time.perf_counter()
        had_429 = False
        last_response_out: DynamicInvoiceAnalyzerResponse | None = None
        try:
            # PASO 1: Decodificar documento o imagen (PDF, JPEG, PNG)
            doc_bytes, mime_type = DynamicInvoiceAnalyzer._decode_and_validate_document(
                request.pdf_base64
            )
            # PASO 2: Prompt de texto (system + consulta)
            combined_prompt = DynamicInvoiceAnalyzer._build_prompt_text(
                request.custom_system_prompt,
                request.user_query
            )
            # PASO 3: API nativa de Google GenAI — acepta PDF o imagen según mime_type
            client = genai.Client(api_key=get_effective_gemini_api_key_for_client())
            start_gemini = time.perf_counter()
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[
                        types.Part.from_bytes(data=doc_bytes, mime_type=mime_type),
                        types.Part.from_text(text=combined_prompt),
                    ],
                    config=types.GenerateContentConfig(
                        temperature=request.temperature,
                    ),
                )
            except Exception as gemini_err:
                end_gemini = time.perf_counter()
                duration_gemini_ms = (end_gemini - start_gemini) * 1000
                had_429 = _is_rate_limit_error(gemini_err)
                _log_gemini_error(request_id, gemini_err, duration_gemini_ms)
                raise
            end_gemini = time.perf_counter()
            duration_gemini_ms = (end_gemini - start_gemini) * 1000
            logger.info(
                "dynamic_invoice_analyzer Gemini ok request_id=%s duration_ms=%.0f",
                request_id,
                duration_gemini_ms,
            )

            raw_content = getattr(response, "text", None) or ""
            logger.info(
                "dynamic_invoice_analyzer LLM texto bruto request_id=%s chars=%s preview=%s",
                request_id,
                len(raw_content),
                _truncate_for_log(raw_content, 1200),
            )

            # Errores "suaves": respuesta sin contenido o bloqueada
            candidates = getattr(response, "candidates", None) or []
            if not raw_content and candidates:
                finish_reason = getattr(candidates[0], "finish_reason", None) or str(getattr(candidates[0], "finish_reason", ""))
                logger.warning(
                    "dynamic_invoice_analyzer Gemini respuesta vacía request_id=%s finish_reason=%s",
                    request_id,
                    finish_reason,
                )
            prompt_feedback = getattr(response, "prompt_feedback", None)
            if prompt_feedback is not None:
                block_reason = getattr(prompt_feedback, "block_reason", None) or getattr(prompt_feedback, "block_reason", "")
                if block_reason:
                    logger.warning(
                        "dynamic_invoice_analyzer Gemini prompt bloqueado request_id=%s block_reason=%s",
                        request_id,
                        str(block_reason)[:200],
                    )

            # PASO 4: Limpiar y blindar la salida (solo JSON; nunca base64 ni volcado crudo)
            clean_result = DynamicInvoiceAnalyzer._clean_json_response(raw_content)
            logger.info(
                "dynamic_invoice_analyzer JSON limpio request_id=%s chars=%s preview=%s",
                request_id,
                len(clean_result),
                _truncate_for_log(clean_result, 1200),
            )

            # PASO 5: Parsear a objeto para que la respuesta tenga result como JSON (sin escapes)
            try:
                parsed_obj = json.loads(clean_result)
            except json.JSONDecodeError as jde:
                logger.warning(
                    "dynamic_invoice_analyzer parse result falló request_id=%s json_err=%s clean_preview=%s",
                    request_id,
                    str(jde),
                    _truncate_for_log(clean_result, 800),
                )
                parsed_obj = {}

            result_obj = DynamicInvoiceAnalyzer._normalize_canonical_result(parsed_obj)
            if isinstance(result_obj, dict):
                fields_obj = result_obj.get("fields") if isinstance(result_obj.get("fields"), dict) else {}
                all_fields_null = bool(fields_obj) and all(v is None for v in fields_obj.values())
                has_no_items = not result_obj.get("line_items")
                if all_fields_null and has_no_items and raw_content:
                    logger.warning(
                        "dynamic_invoice_analyzer normalización vacía request_id=%s raw_preview=%s",
                        request_id,
                        _truncate_for_log(raw_content, 2500),
                    )

            token_usage = DynamicInvoiceAnalyzer._extract_token_usage_from_genai(response)
            response_out = DynamicInvoiceAnalyzerResponse(
                success=True,
                result=result_obj,
                metadata={
                    "model": "gemini-2.5-flash",
                    "temperature": request.temperature,
                    "pdf_size_bytes": len(doc_bytes),
                    "tokens": token_usage
                }
            )

            # Log estructura de respuesta antes de devolver
            _log_response_structure(request_id, response_out)
            last_response_out = response_out
            return response_out
        except ValueError as ve:
            logger.warning("dynamic_invoice_analyzer validación request_id=%s error=%s", request_id, str(ve))
            last_response_out = DynamicInvoiceAnalyzerResponse(
                success=False,
                error=f"Validation Error: {str(ve)}"
            )
            return last_response_out
        except Exception as e:
            had_429 = had_429 or _is_rate_limit_error(e)
            logger.error(
                "dynamic_invoice_analyzer error request_id=%s type=%s message=%s",
                request_id,
                type(e).__name__,
                str(e)[:500],
                exc_info=True,
            )
            last_response_out = DynamicInvoiceAnalyzerResponse(
                success=False,
                error=f"Internal Error: {type(e).__name__} - {str(e)}"
            )
            return last_response_out
        finally:
            with _in_flight_lock:
                _in_flight -= 1
            try:
                duration_total_ms = (time.perf_counter() - start_total) * 1000
                has_result = _response_has_data_result(last_response_out)
                logger.info(
                    "dynamic_invoice_analyzer resumen request_id=%s duration_total_ms=%.0f had_429=%s has_data_result=%s",
                    request_id,
                    duration_total_ms,
                    had_429,
                    has_result,
                )
            except Exception as log_err:
                logger.warning(
                    "dynamic_invoice_analyzer error en resumen request_id=%s %s",
                    request_id,
                    log_err,
                )


# ============================================
# LANGCHAIN TOOL (registrada en discovery y ejecutable vía POST /execute)
# ============================================

from langchain_core.tools import tool


@tool(args_schema=DynamicInvoiceAnalyzerRequest)
def dynamic_invoice_analyzer(
    pdf_base64: str,
    custom_system_prompt: str,
    user_query: str,
    temperature: float = 0,
) -> dict:
    """
    Análisis dinámico de PDFs o imágenes (facturas/documentos) con system prompt y query personalizados.
    Acepta PDF, JPEG o PNG en Base64 (máx 10MB). Sin persistencia entre llamadas.
    """
    try:
        request = DynamicInvoiceAnalyzerRequest(
            pdf_base64=pdf_base64,
            custom_system_prompt=custom_system_prompt,
            user_query=user_query,
            temperature=temperature,
        )
        response = DynamicInvoiceAnalyzer.execute(request)
        return response.model_dump()
    except Exception as e:
        logger.error(
            "dynamic_invoice_analyzer wrapper error type=%s message=%s",
            type(e).__name__,
            str(e)[:500],
            exc_info=True,
        )
        return DynamicInvoiceAnalyzerResponse(
            success=False,
            error=f"Tool Error: {type(e).__name__} - {str(e)}"
        ).model_dump()
```

## Checklist de verificación para la IA replicadora

1. ¿`bind_gemini_api_key` se llama antes de `get_vision_model` / `genai.Client`?
2. ¿El modelo es exactamente `gemini-2.5-flash` en **ambos** caminos (LangChain y SDK)?
3. ¿La tool imagen usa el mismo texto de `system_instructions` y el mismo prefijo `data:image/jpeg;base64,`?
4. ¿El analizador usa `types.Part.from_bytes` y `types.Part.from_text` en `contents` como en el código?
5. ¿La validación Pydantic de `pdf_base64` replica `max_size = 10 * 1024 * 1024` y los tres magic bytes?

---
*Los bloques de código son copias literales de los archivos del repositorio, salvo el preámbulo `from typing import Optional` / `field_validator` añadido al fragmento de `schemas` para poder pegarlo en un módulo mínimo.*
