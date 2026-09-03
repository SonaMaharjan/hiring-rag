import sqlite3
import os
import json
from typing import List, Dict, Any, Optional

DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "hiring_rag.db"
)

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_connection() as conn:
        # Create job_postings table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS job_postings (
                job_id TEXT PRIMARY KEY,
                education TEXT,
                language TEXT,
                certifications TEXT,
                required_years INTEGER,
                approved INTEGER DEFAULT 0
            )
        """)
        
        # Create candidates table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS candidates (
                candidate_id TEXT PRIMARY KEY,
                job_id TEXT,
                name TEXT,
                email TEXT,
                phone TEXT,
                education TEXT,
                language TEXT,
                licenses TEXT,
                experience_years INTEGER
            )
        """)
        
        # Create score_cache table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS score_cache (
                job_id TEXT,
                candidate_id TEXT,
                similarity_score REAL,
                skill_overlap_score REAL,
                experience_score REAL,
                composite_score REAL,
                flags TEXT,
                flag_count INTEGER,
                flag_tier INTEGER,
                PRIMARY KEY (job_id, candidate_id)
            )
        """)
        
        # Create session_state table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS session_state (
                session_id TEXT PRIMARY KEY,
                job_id TEXT,
                candidate_id TEXT
            )
        """)
        conn.commit()

# --- Session State Helpers ---

def set_session_state(session_id: str, job_id: Optional[str] = None, candidate_id: Optional[str] = None) -> None:
    with get_connection() as conn:
        # Get existing state
        row = conn.execute("SELECT * FROM session_state WHERE session_id = ?", (session_id,)).fetchone()
        if row:
            # Update only non-None values
            new_job_id = job_id if job_id is not None else row["job_id"]
            new_candidate_id = candidate_id if candidate_id is not None else row["candidate_id"]
            conn.execute(
                "UPDATE session_state SET job_id = ?, candidate_id = ? WHERE session_id = ?",
                (new_job_id, new_candidate_id, session_id)
            )
        else:
            conn.execute(
                "INSERT INTO session_state (session_id, job_id, candidate_id) VALUES (?, ?, ?)",
                (session_id, job_id, candidate_id)
            )
        conn.commit()

def get_session_state(session_id: str) -> Dict[str, Optional[str]]:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM session_state WHERE session_id = ?", (session_id,)).fetchone()
        if row:
            return {"job_id": row["job_id"], "candidate_id": row["candidate_id"]}
        return {"job_id": None, "candidate_id": None}

# --- Job Postings Helpers ---

def create_job_posting(job_id: str, education: str, language: str, certifications: str, required_years: int) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO job_postings (job_id, education, language, certifications, required_years, approved)
            VALUES (?, ?, ?, ?, ?, 0)
            """,
            (job_id, education, language, certifications, required_years)
        )
        conn.commit()

def approve_job_posting(job_id: str) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE job_postings SET approved = 1 WHERE job_id = ?",
            (job_id,)
        )
        conn.commit()

def update_job_posting_requirements(job_id: str, education: str, language: str, certifications: str, required_years: int) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE job_postings 
            SET education = ?, language = ?, certifications = ?, required_years = ?
            WHERE job_id = ?
            """,
            (education, language, certifications, required_years, job_id)
        )
        conn.commit()

def get_job_posting(job_id: str) -> Optional[Dict[str, Any]]:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM job_postings WHERE job_id = ?", (job_id,)).fetchone()
        return dict(row) if row else None

# --- Candidates Helpers ---

def create_candidate(
    candidate_id: str, 
    job_id: str, 
    name: str, 
    email: str, 
    phone: str, 
    education: str, 
    language: str, 
    licenses: str, 
    experience_years: int
) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO candidates (candidate_id, job_id, name, email, phone, education, language, licenses, experience_years)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (candidate_id, job_id, name, email, phone, education, language, licenses, experience_years)
        )
        conn.commit()

def get_candidates_for_job(job_id: str) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM candidates WHERE job_id = ?", (job_id,)).fetchall()
        return [dict(row) for row in rows]

def get_candidate(candidate_id: str) -> Optional[Dict[str, Any]]:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM candidates WHERE candidate_id = ?", (candidate_id,)).fetchone()
        return dict(row) if row else None

# --- Score Cache Helpers ---

def cache_scores(
    job_id: str,
    candidate_id: str,
    similarity_score: float,
    skill_overlap_score: float,
    experience_score: float,
    composite_score: float,
    flags: List[str],
    flag_count: int,
    flag_tier: int
) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO score_cache (
                job_id, candidate_id, similarity_score, skill_overlap_score, 
                experience_score, composite_score, flags, flag_count, flag_tier
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id, candidate_id, similarity_score, skill_overlap_score,
                experience_score, composite_score, json.dumps(flags), flag_count, flag_tier
            )
        )
        conn.commit()

def get_cached_scores(job_id: str) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM score_cache WHERE job_id = ?", (job_id,)).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["flags"] = json.loads(d["flags"])
            result.append(d)
        return result

def get_cached_candidate_score(job_id: str, candidate_id: str) -> Optional[Dict[str, Any]]:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM score_cache WHERE job_id = ? AND candidate_id = ?",
            (job_id, candidate_id)
        )
        row = row.fetchone()
        if row:
            d = dict(row)
            d["flags"] = json.loads(d["flags"])
            return d
        return None

# Initialize db when this module is imported if needed, or call explicitly
init_db()
