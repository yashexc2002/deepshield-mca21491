import sqlite3
import json
import os
from datetime import datetime
from typing import List, Dict, Any

DB_PATH = os.path.join(os.path.dirname(__file__), "deepshield_audit.sqlite3")


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS media_audit_logs (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            media_type TEXT NOT NULL,
            facial_region TEXT NOT NULL,
            compression_level TEXT NOT NULL,
            generator_category TEXT NOT NULL,
            sms_score REAL NOT NULL,
            fas_score REAL NOT NULL,
            authenticity_prob REAL NOT NULL,
            verdict TEXT NOT NULL,
            inference_ms REAL NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    cur.execute("SELECT COUNT(*) FROM media_audit_logs")
    count = cur.fetchone()[0]
    if count == 0:
        seed_records = [
            ("DS-AUD-1041", "dfdc_subject_089_frame42.mp4", "Video Frame", "Jawline & Face-Swap Blending Mask", "H.264 Social (CRF 28)", "Face-Swap (DeepFaceLab)", 0.912, 0.864, 10.62, "SYNTHETIC / MANIPULATED", 48.4, "2026-10-28 14:12:09"),
            ("DS-AUD-1040", "stylegan3_synth_portrait_412.png", "Image", "Ocular & Periocular Region", "Lossless PNG", "GAN Synthesis (StyleGAN3)", 0.884, 0.941, 9.43, "SYNTHETIC / MANIPULATED", 39.2, "2026-10-28 14:08:51"),
            ("DS-AUD-1039", "kyc_verification_passport_019.jpg", "Image", "Natural Facial Symmetry", "JPEG Q92", "Natural Camera Sensor (Authentic)", 0.082, 0.104, 90.96, "AUTHENTIC MEDIA", 36.8, "2026-10-28 13:55:20"),
            ("DS-AUD-1038", "flux_diffusion_avatar_882.webp", "Image", "Perioral & Lip Boundary", "WebP Social", "Diffusion Synthesis (Latent Upsampling)", 0.845, 0.918, 12.72, "SYNTHETIC / MANIPULATED", 41.5, "2026-10-28 13:42:11"),
            ("DS-AUD-1037", "interview_press_clip_frame108.mp4", "Video Frame", "Natural Facial Symmetry", "H.264 Broadcast", "Natural Camera Sensor (Authentic)", 0.115, 0.128, 88.01, "AUTHENTIC MEDIA", 44.0, "2026-10-28 13:30:04"),
        ]
        cur.executemany("""
            INSERT INTO media_audit_logs (
                id, filename, media_type, facial_region, compression_level,
                generator_category, sms_score, fas_score, authenticity_prob,
                verdict, inference_ms, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, seed_records)
        conn.commit()
    conn.close()


def insert_audit_log(record: Dict[str, Any]) -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO media_audit_logs (
            id, filename, media_type, facial_region, compression_level,
            generator_category, sms_score, fas_score, authenticity_prob,
            verdict, inference_ms, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        record["id"],
        record["filename"],
        record["media_type"],
        record["facial_region"],
        record.get("compression_level", "Standard Web Upload"),
        record["generator_category"],
        record["sms_score"],
        record["fas_score"],
        record["authenticity_prob"],
        record["verdict"],
        record["inference_ms"],
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    ))
    conn.commit()
    conn.close()


def fetch_audit_logs(limit: int = 25) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM media_audit_logs ORDER BY created_at DESC LIMIT ?", (limit,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows
