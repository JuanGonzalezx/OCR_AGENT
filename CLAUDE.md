# OCR Service — Iguana Café

## Propósito
Microservicio HTTP que recibe imagen de factura y devuelve JSON estructurado con los datos extraídos. Usado por la app mobile cuando un encargado de sucursal recibe insumos.

## Stack
- Python 3.11+
- FastAPI
- Google Gemini API (modelo: gemini-2.0-flash-exp o gemini-1.5-flash)
- Pydantic v2 para schemas
- Uvicorn para servir
- Deploy: Render.com

## Variables de entorno
- `GEMINI_API_KEY: ` — API key de Google AI Studio
- `PORT` — Puerto (Render lo inyecta automáticamente)
- `LOG_LEVEL` — INFO por defecto

## No hagas
- No uses Tesseract ni librerías OCR clásicas. Solo Gemini Vision.
- No persistas las imágenes recibidas en disco. Procesa en memoria.
- No expongas la API key en logs ni respuestas.
- No guardes resultados en BD todavía. Por ahora solo retorna JSON.

## Sí haz
- Valida que el archivo recibido sea imagen real (jpg/png/webp/heic).
- Maneja errores de Gemini con códigos HTTP claros (502 si Gemini falla, 400 si imagen inválida, 422 si no se pudo extraer).
- Logs estructurados (JSON) con request_id.
- Tiempo máximo de procesamiento: 30s.