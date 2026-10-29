import os
from celery import Celery
from forensic_engine import DualDomainForensicEngine

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "deepshield_forensic_tasks",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Kolkata",
    enable_utc=True,
)

engine = DualDomainForensicEngine()


@celery_app.task(name="forensics.evaluate_media_task")
def evaluate_media_task(image_b64: str, filename: str, media_type: str = "Image"):
    import base64
    raw_bytes = base64.b64decode(image_b64)
    return engine.evaluate_media(raw_bytes, filename=filename, media_type=media_type)
