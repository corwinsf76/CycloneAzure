# /orchestration/main.py

import logging
import sys
import os
import threading
import pandas as pd
import pytz

# Ensure project root is discoverable if running main.py directly
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Absolute imports ---
import config
from database import db_utils
from orchestration import scheduler
from sentiment_analysis import analyzer
from modeling import predictor

# --- Setup Logging ---
log_level = getattr(logging, config.LOG_LEVEL, logging.INFO)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(levelname)s [%(name)s:%(lineno)d] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)

# Mute overly verbose libraries
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("schedule").setLevel(logging.WARNING)
logging.getLogger("transformers.modeling_utils").setLevel(logging.WARNING)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.WARNING)

log.info(f"Project root added to sys.path: {project_root}")
log.info(f"Logging configured with level: {config.LOG_LEVEL}")

def main():
    """Main function to initialize and run the application."""
    log.info("========================================")
    log.info("== Starting Cyclone v2 Trading Bot   ==")
    log.info("========================================")

    # --- 1. Configuration Check ---
    log.info("Checking essential configuration...")
    if not config.check_essential_config():
        log.critical("Essential configuration missing. Please check .env file or environment variables. Exiting.")
        sys.exit(1)
    log.info("Essential configuration loaded.")

    # --- 2. Database Initialization ---
    log.info("Initializing database connection and schema...")
    if not db_utils.engine:
        log.critical("Database engine not created. Check DATABASE_URL and DB connectivity. Exiting.")
        sys.exit(1)
    if not db_utils.init_db():
        log.warning("Database schema initialization failed or encountered issues. Check logs. Continuing...")
    else:
        log.info("Database schema initialized successfully.")

    # --- 3. Preload Models ---
    log.info("Preloading Sentiment Analysis model...")
    if analyzer.load_sentiment_model() is None:
        log.warning("Failed to preload sentiment model. Sentiment analysis job may fail.")
    else:
        log.info("Sentiment model preloaded.")

    log.info("Preloading Prediction model...")
    if predictor.load_model_and_metadata() is None:
        log.warning("Failed to preload prediction model. Prediction/Trading logic may fail. Ensure model is trained.")
    else:
        log.info("Prediction model preloaded.")

    # --- 4. Initialize Portfolio Manager State ---
    log.info("Initializing Portfolio Manager (in-memory state for now)...")

    # --- 5. Start Dashboard (Optional - Run in separate thread) ---
    run_dashboard = True
    if run_dashboard:
        log.info("Starting dashboard server in a separate thread...")
        try:
            from dashboard.app import app as dashboard_app
            dashboard_thread = threading.Thread(
                target=dashboard_app.run,
                kwargs={'debug': False, 'host': '0.0.0.0', 'port': 8051}
            )
            dashboard_thread.daemon = True
            dashboard_thread.start()
            log.info("Dashboard thread started. Access at http://<your-ip>:8051")
        except Exception as dash_err:
            log.error(f"Failed to start dashboard thread: {dash_err}", exc_info=True)

    # --- 6. Start Scheduler ---
    log.info("Starting main application scheduler...")
    try:
        scheduler.start_scheduler()
    except Exception as e:
        log.critical(f"Scheduler failed unexpectedly: {e}", exc_info=True)
    finally:
        log.info("Application shutdown initiated.")

    log.info("========================================")
    log.info("== Cyclone v2 Trading Bot Stopped    ==")
    log.info("========================================")

if __name__ == "__main__":
    main()
