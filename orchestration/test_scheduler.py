import pytest
from unittest.mock import patch, MagicMock
from orchestration.scheduler import start_scheduler

@pytest.mark.timeout(10)  # Timeout after 10 seconds
def test_scheduler_startup():
    """Test that the scheduler starts without errors."""
    with patch("schedule.run_pending", side_effect=StopIteration):
        with patch("time.sleep", side_effect=StopIteration):
            try:
                start_scheduler()
            except StopIteration:
                pass  # Expected to stop the infinite loop

@pytest.mark.timeout(10)  # Timeout after 10 seconds
def test_scheduler_job_scheduling():
    """Test that jobs are scheduled correctly."""
    with patch("schedule.every") as mock_schedule:
        with patch("orchestration.scheduler.run_price_collection_job") as mock_price_job:
            with patch("orchestration.scheduler.run_news_collection_job") as mock_news_job:
                start_scheduler()
                mock_schedule.assert_called()
                mock_price_job.assert_not_called()
                mock_news_job.assert_not_called()