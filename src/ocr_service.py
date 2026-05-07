import json
import logging
import os
import re

from google import genai
from google.genai import types

from .schemas import ExtractData, Factura, ItemFactura, Proveedor, Totales

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """Eres un experto en documentos fiscales colombianos. Analiza esta imagen de factura y extrae los datos. Devuelve ÚNICAMENTE el siguiente JSON, sin markdown, sin explicaciones, sin texto adicional antes ni después del JSON.

{
  "proveedor": {
    "nombre": "<razón social del proveedor o null>",
    "nit": "<NIT formato colombiano con dígito verificador o null>",
    "telefono": "<teléfono del proveedor o null>"
  },
  "factura": {
    "numero": "<número de factura tal como aparece o null>",
    "fecha": "<fecha en formato YYYY-MM-DD o null>",
    "forma_pago": "<contado | credito | null>"
  },
  "items": [
    {
      "descripcion": "<nombre del producto o servicio>",
      "cantidad": <número entero o decimal>,
      "unidad": "<und | kg | g | l | ml | caja | otro>",
      "precio_unitario": <número entero sin separadores>,
      "subtotal": <número entero sin separadores>
    }
  ],
  "totales": {
    "subtotal": <número entero o null>,
    "iva": <número entero o null>,
    "total": <número entero o null>
  },
  "confidence": <decimal entre 0.0 y 1.0>
}

REGLAS ESTRICTAS:
- NÚMEROS: sin separadores de miles (28400, NO 28.400 — los puntos en
  pesos colombianos son miles, no decimales). Devuelve números reales,
  no strings.
- FECHAS: siempre YYYY-MM-DD, convierte cualquier formato
  (15/01/2026 → 2026-01-15)
- NIT: con dígito verificador si está visible (900123456-7), sin puntos
  internos
- CAMPOS AUSENTES: usa null, NUNCA omitas un campo del schema
- ITEMS: extrae TODAS las líneas de productos visibles, no resumas
- IVA: estándar colombiano es 19%; si no aparece explícito calcúlalo
  del total
- FORMA DE PAGO: "contado" si el pago es inmediato; "credito" si hay
  fecha de vencimiento diferente a la emisión
- CONFIDENCE: 1.0 si todo es perfectamente legible; reduce por cada
  campo borroso, tachado o ilegible
- La imagen puede ser foto con celular: tolera inclinación o
  iluminación imperfecta
- Si la imagen NO es una factura (es otra cosa), devuelve todos los
  campos en null y confidence: 0.0
"""


def _clean_json_response(content: str) -> str:
    # Prioridad 1: bloque ```json...```
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
    if fence:
        return fence.group(1).strip()

    # Prioridad 2: primer { al último }
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        return content[start : end + 1]

    return content


def extract_invoice(image_bytes: bytes, mime_type: str) -> ExtractData:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY no configurada en el entorno")

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            types.Part.from_text(text=EXTRACTION_PROMPT),
        ],
        config=types.GenerateContentConfig(temperature=0),
    )

    raw_text = response.text
    logger.debug("Gemini raw response (first 300 chars): %s", raw_text[:300])

    clean = _clean_json_response(raw_text)

    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError as exc:
        logger.error("JSON decode failed. Raw response: %s", raw_text)
        raise ValueError(f"Gemini no devolvió JSON válido: {exc}") from exc

    proveedor_raw = parsed.get("proveedor") or {}
    factura_raw = parsed.get("factura") or {}
    totales_raw = parsed.get("totales") or {}
    items_raw = parsed.get("items") or []

    return ExtractData(
        proveedor=Proveedor(**{k: v for k, v in proveedor_raw.items() if k in Proveedor.model_fields}),
        factura=Factura(**{k: v for k, v in factura_raw.items() if k in Factura.model_fields}),
        items=[
            ItemFactura(**{k: v for k, v in item.items() if k in ItemFactura.model_fields})
            for item in items_raw
            if isinstance(item, dict)
        ],
        totales=Totales(**{k: v for k, v in totales_raw.items() if k in Totales.model_fields}),
        confidence=parsed.get("confidence"),
    )
