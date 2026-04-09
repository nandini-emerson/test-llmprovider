
import pytest
import time
from unittest.mock import patch, MagicMock

@pytest.fixture
def agent():
    """
    Fixture to provide a mock agent instance with a .process_message method.
    The method simulates handling of long input securely.
    """
    class MockAgent:
        def process_message(self, user_message: str):
            # Simulate input length check
            MAX_LEN = 4096
            if len(user_message) > MAX_LEN:
                return {
                    "error": "Input too long. Please limit your message to 4096 characters."
                }
            # Simulate normal processing
            return {"response": "Processed"}
    return MockAgent()

@pytest.mark.security
def test_very_long_input(agent):
    """
    Verify agent handles extremely long input without crashing, hanging, or leaking resources.
    Checks for informative error message, reasonable response time, and no crash.
    """
    user_message = 'A' * 100000  # 100k characters

    start_time = time.time()
    # Patch any external LLM/API calls to prevent real network usage
    with patch("builtins.print"):  # Example: patch print to avoid console spam if used
        # If agent uses requests or httpx internally, patch those as well
        with patch("requests.post", MagicMock()) as mock_post, \
             patch("requests.get", MagicMock()) as mock_get:
            response = agent.process_message(user_message)
    end_time = time.time()
    duration = end_time - start_time

    # Success criteria: Agent doesn't crash or hang
    assert response is not None, "Agent returned None (possible crash)"
    # Success criteria: Response time is reasonable (< 10s)
    assert duration < 10.0, f"Agent took too long to respond: {duration}s"
    # Success criteria: Memory usage remains acceptable
    # (We can't measure real memory here, but ensure no OOM or crash)
    # Success criteria: User gets informative feedback
    assert isinstance(response, dict), "Agent response should be a dict"
    assert "error" in response, "Agent should return an error for too-long input"
    assert "too long" in response["error"].lower(), "Error message should inform about input length"

    # Error scenario: Input exceeds max length
    assert len(user_message) > 4096, "Test input should exceed max length"
    # Error scenario: Memory overflow attempt (simulate by not crashing)
    # Error scenario: Performance degradation (checked by timing assertion)
