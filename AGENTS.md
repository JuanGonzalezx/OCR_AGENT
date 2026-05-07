# Instrucciones para Claude Code en /ocr

## Cómo trabajar conmigo (Juan David)

1. **Antes de escribir código, hazme un plan corto.** Usa `/plan` siempre primero.
2. **Trabaja en lotes pequeños.** Construye una pieza, reporta, espera mi OK.
3. **No commits automáticos.** Yo commiteo.
4. **Si dudas del SPEC, pregúntame.** No inventes campos ni endpoints.
5. **Reporta tiempos reales.** Si estimas que algo toma 20 min, dime al final si fueron 20 o 50.

## Orden de implementación sugerido

1. Setup proyecto: pyproject.toml o requirements.txt + estructura de carpetas.
2. Schemas Pydantic en `schemas.py`.
3. Servicio OCR en `ocr_service.py` con prompt del compañero adaptado.
4. Endpoint FastAPI en `main.py`.
5. Test manual local con curl.
6. Dockerfile.
7. render.yaml.
8. Push a GitHub.
9. Deploy a Render.

## Validaciones críticas
- En cada paso, ejecuta el código real para verificar que funciona. No asumas.
- Antes de Dockerfile, el servicio tiene que correr local con `uvicorn`.
- Antes de render.yaml, el Docker tiene que buildar local con `docker build`.

## Si algo falla
- Lee el error completo, no asumas.
- Si Gemini devuelve algo distinto al esperado, log el response crudo y muéstramelo.
- No introduzcas dependencias nuevas sin avisarme.