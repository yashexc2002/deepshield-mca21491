import uuid
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from forensic_engine import DualDomainForensicEngine
from database import init_db, insert_audit_log, fetch_audit_logs
from benchmark_generator import BENCHMARK_SUMMARY

app = FastAPI(
    title="DeepShield Explainable Media Forensics API",
    description="Hybrid EfficientNet-B0 + ViT-B/16 + 2D-FFT Deepfake Detection Microservice (Project ID: MCA21491)",
    version="1.4.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = DualDomainForensicEngine()


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/api/v1/health")
def health_check():
    return {
        "status": "operational",
        "project_id": "MCA21491",
        "engine": engine.MODEL_VERSION,
        "pipeline": "EfficientNet-B0 + ViT-B/16 + 2D-FFT + Grad-CAM",
        "queue_worker": "Celery/Redis Ready (Local Sync Fallback Enabled)",
        "orm_audit_store": "Prisma/SQLite Audit Store Connected",
    }


@app.post("/api/v1/analyze")
async def analyze_media(
    file: UploadFile = File(...),
    media_type: str = Form("Image"),
    compression_level: str = Form("Standard Web Upload"),
):
    raw_bytes = await file.read()
    result = engine.evaluate_media(raw_bytes, filename=file.filename or "sample.png", media_type=media_type)
    audit_id = f"DS-AUD-{str(uuid.uuid4())[:4].upper()}"
    result["audit_id"] = audit_id

    insert_audit_log({
        "id": audit_id,
        "filename": result["filename"],
        "media_type": media_type,
        "facial_region": result["localization"]["primary_manipulated_region"],
        "compression_level": compression_level,
        "generator_category": result["generator_category"],
        "sms_score": result["scores"]["spatial_manipulation_score_sms"],
        "fas_score": result["scores"]["frequency_anomaly_score_fas"],
        "authenticity_prob": result["authenticity_probability"],
        "verdict": result["verdict"],
        "inference_ms": result["inference_time_ms"],
    })
    return result


@app.get("/api/v1/benchmark")
def get_benchmark_results():
    return BENCHMARK_SUMMARY


@app.get("/api/v1/audit-logs")
def get_audit_logs():
    return {"logs": fetch_audit_logs(limit=30)}
