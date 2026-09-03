import os
import shutil
import tempfile
import pytest
from unittest.mock import MagicMock, patch

from app import database
from app import scoring
from app import agent
from app import pipelines
from app import vector_store

@pytest.fixture(autouse=True)
def setup_test_db():
    # Use in-memory or a temporary sqlite DB for testing
    old_db_path = database.DB_PATH
    temp_dir = tempfile.mkdtemp()
    temp_db_path = os.path.join(temp_dir, "test_hiring_rag.db")
    database.DB_PATH = temp_db_path
    database.init_db()
    
    yield
    
    # Clean up
    database.DB_PATH = old_db_path
    shutil.rmtree(temp_dir)

def test_cosine_similarity():
    # Identical vectors
    v1 = [1.0, 2.0, 3.0]
    v2 = [1.0, 2.0, 3.0]
    assert abs(scoring.cosine_similarity(v1, v2) - 1.0) < 1e-6
    
    # Orthogonal vectors
    v3 = [1.0, 0.0]
    v4 = [0.0, 1.0]
    assert scoring.cosine_similarity(v3, v4) == 0.0
    
    # Opposite vectors
    v5 = [1.0, -1.0]
    v6 = [-1.0, 1.0]
    assert abs(scoring.cosine_similarity(v5, v6) - (-1.0)) < 1e-6

def test_session_state():
    database.set_session_state("session_123", job_id="job_abc", candidate_id="cand_xyz")
    state = database.get_session_state("session_123")
    assert state["job_id"] == "job_abc"
    assert state["candidate_id"] == "cand_xyz"
    
    # Partial update
    database.set_session_state("session_123", candidate_id="cand_111")
    state = database.get_session_state("session_123")
    assert state["job_id"] == "job_abc"
    assert state["candidate_id"] == "cand_111"

@patch("app.llm_client.get_gemini_client")
def test_submit_job_posting(mock_get_client):

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = '{"education": "BS in CS", "language": "English", "certifications": "None", "required_years": 3}'
    mock_client.models.generate_content.return_value = mock_response
    mock_get_client.return_value = mock_client
    
    # Submit job posting
    res = agent.submit_job_posting("We are looking for someone with BS in CS, English speaker, 3 years exp.", session_id="test_session")
    assert "BS in CS" in res
    assert "PENDING APPROVAL" in res
    
    # Verify in DB (approved = 0)
    state = database.get_session_state("test_session")
    job_id = state["job_id"]
    assert job_id is not None
    
    job_reqs = database.get_job_posting(job_id)
    assert job_reqs["approved"] == 0
    assert job_reqs["education"] == "BS in CS"

def test_approve_job_strict_keyword():
    database.set_session_state("test_session", job_id="job_999")
    database.create_job_posting("job_999", "BS", "English", "None", 2)
    
    # Non-approve keyword should fail
    res = agent.approve_job("yes looks good", session_id="test_session")
    assert "Approval failed" in res
    assert database.get_job_posting("job_999")["approved"] == 0
    
    # Strict "approve" keyword succeeds
    with patch("os.path.exists", return_value=True), \
         patch("os.listdir", return_value=[]):
        res = agent.approve_job("approve", session_id="test_session")
        assert "APPROVED" in res
        assert database.get_job_posting("job_999")["approved"] == 1

def test_tie_aware_ranking():
    job_id = "job_tie_test"
    database.create_job_posting(job_id, "BS", "EN", "None", 2)
    database.approve_job_posting(job_id)
    
    # Insert candidates
    database.create_candidate("cand_1", job_id, "Alice", "a@a.com", "123", "BS", "EN", "None", 2)
    database.create_candidate("cand_2", job_id, "Bob", "b@b.com", "456", "BS", "EN", "None", 2)
    database.create_candidate("cand_3", job_id, "Charlie", "c@c.com", "789", "BS", "EN", "None", 2)
    
    # Cache scores with identical composite score for cand_1 and cand_2, lower for cand_3
    # Alice and Bob should share Rank 1
    database.cache_scores(job_id, "cand_1", 0.9, 0.9, 0.9, 0.90, [], 0, 0)
    database.cache_scores(job_id, "cand_2", 0.9, 0.9, 0.9, 0.90, [], 0, 0)
    database.cache_scores(job_id, "cand_3", 0.7, 0.7, 0.7, 0.70, [], 0, 0)
    
    ranked = scoring.get_ranked_candidates(job_id, top_n=5)
    tier_0 = ranked[0]
    
    assert len(tier_0) == 3
    assert tier_0[0]["candidate_id"] in ["cand_1", "cand_2"]
    assert tier_0[0]["rank"] == 1
    assert tier_0[1]["candidate_id"] in ["cand_1", "cand_2"]
    assert tier_0[1]["rank"] == 1
    assert tier_0[2]["candidate_id"] == "cand_3"
    assert tier_0[2]["rank"] == 2  # Dense or competition: next distinct rank is 2
