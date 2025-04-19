import pytest
from unittest.mock import patch, MagicMock
from dashboard.app import app

@pytest.mark.timeout(10)  # Timeout after 10 seconds
def test_dashboard_app_initialization():
    """Test that the dashboard app initializes without errors."""
    with patch("flask.Flask.run", return_value=None):
        assert app is not None

@pytest.mark.timeout(10)  # Timeout after 10 seconds
def test_dashboard_app_routes():
    """Test that the dashboard app has the expected routes."""
    with app.test_client() as client:
        response = client.get("/")
        assert response.status_code == 200