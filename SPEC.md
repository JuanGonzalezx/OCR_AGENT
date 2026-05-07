# SPEC — OCR Service v0.1 (MVP Demo)

## Objetivo
Endpoint HTTP que recibe foto de factura colombiana y devuelve JSON con proveedor, fecha, items y total. Listo para demo en vivo el [fecha de la reunión].

## Endpoints

### `POST /ocr/extract`
**Request:**
- Content-Type: `multipart/form-data`
- Campo: `file` (imagen, max 10MB)

**Response 200:**
```json
{
  "request_id": "uuid-v4",
  "processing_time_ms": 3400,
  "data": {
    "proveedor": {
      "nombre": "Distribuidora XYZ S.A.S.",
      "nit": "900123456-7",
      "telefono": "3001234567"
    },
    "factura": {
      "numero": "FE-12345",
      "fecha": "2025-01-15",
      "forma_pago": "credito"
    },
    "items": [
      {
        "descripcion": "Leche entera 1L",
        "cantidad": 24,
        "unidad": "und",
        "precio_unitario": 4500,
        "subtotal": 108000
      }
    ],
    "totales": {
      "subtotal": 108000,
      "iva": 20520,
      "total": 128520
    },
    "confidence": 0.92
  }
}
```

**Response 400:** Imagen inválida o archivo corrupto.
**Response 422:** Gemini procesó pero no pudo extraer datos confiables.
**Response 502:** Gemini no respondió o falló.
**Response 504:** Timeout (>30s).

### `GET /health`
Devuelve `{"status": "ok"}`. Para health check de Render.

### `GET /`
Devuelve info básica del servicio.

## Criterios de aceptación

**Funcionalidad:**
- [ ] Acepta imágenes JPG, PNG, WEBP.
- [ ] Procesa factura colombiana real en menos de 10 segundos en condiciones normales.
- [ ] Devuelve JSON con la estructura exacta del schema.
- [ ] Si un campo no está en la factura, lo devuelve como `null`, no lo omite.
- [ ] Funciona con factura impresa fotografiada con celular (no perfecta).

**Robustez:**
- [ ] No crashea si la imagen no tiene factura (devuelve 422).
- [ ] No expone la API key de Gemini en ningún log o response.
- [ ] Maneja timeout de Gemini sin colgar el servidor.

**Deploy:**
- [ ] Dockerfile que builda sin errores.
- [ ] render.yaml configurado para deploy automático.
- [ ] Variable GEMINI_API_KEY se lee de entorno.
- [ ] Endpoint público accesible desde internet con HTTPS (Render lo da gratis).

**Testing manual:**
- [ ] Probado con 3 facturas reales distintas antes de deploy.
- [ ] Probado el endpoint desplegado desde curl o Postman.
- [ ] Probado desde la app mobile haciendo fetch real.

## Fuera de alcance (no hacer ahora)
- Autenticación / API keys de cliente.
- Rate limiting.
- Persistencia en BD.
- Soporte multi-tenant.
- Procesamiento en batch.
- Webhooks.

Todo eso se agrega después de la demo si el cliente firma.

## Prompt de Gemini
Ver `app/prompts.py`. Construir basándose en lo siguiente:

- Pedir extracción estructurada en JSON.
- Forzar formato de fecha ISO (YYYY-MM-DD).
- Forzar números sin separadores de miles (108000 no 108.000).
- Si un campo no es legible, usar null.
- Devolver nivel de confianza global 0-1.
- Mencionar explícitamente que es factura colombiana (formatos NIT, IVA 19%).