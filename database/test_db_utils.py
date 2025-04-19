import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from database.db_utils import init_db, price_data, bulk_insert_data, df_to_db

@pytest.mark.timeout(10)  # Timeout after 10 seconds
def test_db_initialization_success():
    """Test that the database initializes successfully."""
    with patch("database.db_utils.engine.connect", return_value=MagicMock()):
        assert init_db() is True

@pytest.mark.timeout(10)  # Timeout after 10 seconds
def test_db_initialization_failure():
    """Test that the database initialization fails gracefully."""
    with patch("database.db_utils.engine.connect", side_effect=Exception("DB Error")):
        assert init_db() is False

@pytest.mark.timeout(10)  # Timeout after 10 seconds
def test_bulk_insert_data_success():
    """Test that bulk_insert_data inserts data successfully."""
    with patch("database.db_utils.engine.connect") as mock_connect:
        mock_connection = mock_connect.return_value.__enter__.return_value
        mock_connection.execute.return_value = None

        data = [{"id": 1, "symbol": "BTC", "price": 50000}]
        table = price_data

        with patch("database.db_utils.bulk_insert_data", return_value=1) as mock_bulk_insert:
            result = bulk_insert_data(data, table)
            assert result == 1
            mock_connection.execute.assert_called_once()

@pytest.mark.timeout(10)  # Timeout after 10 seconds
def test_bulk_insert_data_duplicate():
    """Test that bulk_insert_data skips duplicates."""
    with patch("database.db_utils.engine.connect") as mock_connect:
        mock_connection = mock_connect.return_value.__enter__.return_value
        mock_connection.execute.side_effect = [
            [{"symbol": "BTC"}],  # Simulate existing record
            None
        ]

        data = [{"id": 1, "symbol": "BTC", "price": 50000}]
        table = price_data

        with patch("database.db_utils.bulk_insert_data", return_value=0) as mock_bulk_insert:
            result = bulk_insert_data(data, table, unique_column="symbol")
            assert result == 0

@pytest.mark.timeout(10)  # Timeout after 10 seconds
def test_df_to_db_success():
    """Test that df_to_db writes a DataFrame to the database successfully."""
    with patch("pandas.DataFrame.to_sql") as mock_to_sql:
        df = pd.DataFrame([{"id": 1, "symbol": "BTC", "price": 50000}])
        df_to_db(df, "price_data")
        mock_to_sql.assert_called_once()