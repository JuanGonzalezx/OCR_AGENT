#!/bin/bash
# Uso: ./test_local.sh factura.jpg
# Si no se pasa argumento, busca factura.jpg en el directorio actual.
FILE=${1:-factura.jpg}

if [ ! -f "$FILE" ]; then
  echo "Error: archivo '$FILE' no encontrado."
  echo "Uso: ./test_local.sh <ruta-a-imagen>"
  exit 1
fi

echo "Testeando OCR con: $FILE"
echo "---"
curl -s -X POST http://localhost:8000/ocr/extract \
  -F "file=@$FILE" \
  | python3 -m json.tool
