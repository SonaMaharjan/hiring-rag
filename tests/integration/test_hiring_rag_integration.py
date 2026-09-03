import os
import shutil
import tempfile
import pytest
from unittest.mock import MagicMock, patch
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from app import database
from app.agent import root_agent

@pytest.fixture(autouse=True)
def setup_test_db():
    old_db_path = database.DB_PATH
    temp_dir = tempfile.mkdtemp()
    temp_db_path = os.path.join(temp_dir, "test_hiring_rag_integration.db")
    database.DB_PATH = temp_db_path
    database.init_db()
    
    yield
    
    database.DB_PATH = old_db_path
    shutil.rmtree(temp_dir)

@patch("app.pipelines.get_gemini_client")
def test_agent_hiring_pipeline_flow(mock_get_client):
    # Mocking LLM calls
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    
    # Mock return values for requirements extraction
    mock_response = MagicMock()
    mock_response.text = '{"education": "PhD in AI", "language": "German", "certifications": "None", "required_years": 5}'
    mock_client.models.generate_content.return_value = mock_response

    # ADK runner setup
    session_service = InMemorySessionService()
    session = session_service.create_session_sync(user_id="test_user", app_name="test")
    runner = Runner(agent=root_agent, session_service=session_service, app_name="test")

    # 1. User submits job posting
    message1 = types.Content(
        role="user", 
        parts=[types.Part.from_text(text="Please submit job posting: We need a PhD in AI with 5 years experience.")]
    )
    
    # We mock the runner's async generation stream to simulate agent tool calling
    # In a real environment, the ReAct agent decides to call submit_job_posting.
    # We verify the tool directly first, and can test the tool execution.
    res = root_agent.tools[0](posting_text="We need a PhD in AI with 5 years experience.", session_id=session.id)
    assert "PhD in AI" in res
    assert "PENDING APPROVAL" in res

    # Check state was stored
    state = database.get_session_state(session.id)
    job_id = state["job_id"]
    assert job_id is not None
    
    job_reqs = database.get_job_posting(job_id)
    assert job_reqs["approved"] == 0
    assert job_reqs["education"] == "PhD in AI"

    # 2. Try to rank before approval -> should block
    rank_res = root_agent.tools[2](session_id=session.id)
    assert "NOT approved yet. Ranking is blocked." in rank_res

    # 3. Approve the job
    with patch("os.path.exists", return_value=True), \
         patch("os.listdir", return_value=[]):
        approve_res = root_agent.tools[1](confirmation="approve", session_id=session.id)
        assert "APPROVED" in approve_res
        assert database.get_job_posting(job_id)["approved"] == 1

    # 4. Now rank candidates (should succeed, returning empty tables since no resumes)
    rank_res_after = root_agent.tools[2](session_id=session.id)
    assert "Ranks for Job" in rank_res_after
    assert "Tier 0" in rank_res_after
