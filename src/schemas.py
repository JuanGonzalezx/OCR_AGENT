from typing import List, Optional
from pydantic import BaseModel


class Proveedor(BaseModel):
    nombre: Optional[str] = None
    nit: Optional[str] = None
    telefono: Optional[str] = None


class Factura(BaseModel):
    numero: Optional[str] = None
    fecha: Optional[str] = None
    forma_pago: Optional[str] = None


class ItemFactura(BaseModel):
    descripcion: Optional[str] = None
    cantidad: Optional[float] = None
    unidad: Optional[str] = None
    precio_unitario: Optional[float] = None
    subtotal: Optional[float] = None


class Totales(BaseModel):
    subtotal: Optional[float] = None
    iva: Optional[float] = None
    total: Optional[float] = None


class ExtractData(BaseModel):
    proveedor: Optional[Proveedor] = None
    factura: Optional[Factura] = None
    items: List[ItemFactura] = []
    totales: Optional[Totales] = None
    confidence: Optional[float] = None


class ExtractResponse(BaseModel):
    request_id: str
    processing_time_ms: int
    data: ExtractData
