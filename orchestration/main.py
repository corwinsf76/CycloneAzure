# /orchestration/main.py

import logging
import sys
import os
import threading # To potentially run dashboard in parallel
import pandas as pd # Added for Timedelta if needed, though config uses strings now
import pytz # Added for timezone awareness

# Ensure project root is discoverable if running main.py directly
# This adds the parent directory ('cyclonev2') to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    # Use logging instead of print after basicConfig is called
    # print(f"Added project root to sys.path: {project_root}")


# --- Absolute imports now should work ---
import config
from database import db_utils
from orchestration import scheduler
from sentiment_analysis import analyzer # To preload model
from modeling import predictor # To preload model
# Import dashboard app if running integrated
# from dashboard.app import app as dashboard_app

# --- Setup Logging ---
# Centralize logging configuration
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format='%(asctime)s - %(levelname)s [%(name)s:%(lineno)d] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

# Mute overly verbose libraries if needed
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("schedule").setLevel(logging.WARNING) # Schedule logs pending/running jobs
# Reduce verbosity from huggingface libraries if desired
logging.getLogger("transformers.modeling_utils").setLevel(logging.WARNING)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.WARNING)

# Log path addition after logging is configured
log.info(f"Project root added to sys.path: {project_root}")
log.info(f"Logging configured with level: {config.LOG_LEVEL}")


# --- Main Application ---
def main():
    """Main function to initialize and run the application."""
    log.info("========================================")
    log.info("== Starting Cyclone v2 Trading Bot   ==")
    log.info("========================================")

    # --- 1. Configuration Check ---
    log.info("Starting configuration check...")
    log.info("Checking essential configuration...")
    if not config.check_essential_config():
        log.critical("Essential configuration missing. Please check .env file or environment variables. Exiting.")
        sys.exit(1)

    log.info("Essential configuration loaded.")
    log.info("Configuration check completed successfully.")

    # --- 2. Database Initialization ---
    log.info("Starting database initialization...")
    log.info("Initializing database connection and schema...")
    if not db_utils.engine:
        log.critical("Database engine not created. Check DATABASE_URL and DB connectivity. Exiting.")
        sys.exit(1)
    if not db_utils.init_db():
        log.warning("Database schema initialization failed or encountered issues. Check logs. Continuing...")
    else:
        log.info("Database schema initialized successfully.")
    log.info("Database initialization completed.")

    # --- 3. Preload Models ---
    log.info("Starting model preloading...")
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
    log.info("Model preloading completed.")

    # --- 4. Initialize Portfolio Manager State ---
    log.info("Initializing Portfolio Manager (in-memory state for now)...")

    # --- 5. Start Dashboard (Optional - Run in separate thread/process) ---
    log.info("Starting dashboard initialization...")
    run_dashboard = True
    if run_dashboard:
        try:
            from dashboard.app import app as dashboard_app
            dashboard_thread = threading.Thread(
                target=dashboard_app.run,
                kwargs={'debug': False, 'host': '0.0.0.0', 'port': 8052}
            )
            dashboard_thread.daemon = True
            dashboard_thread.start()
            log.info("Dashboard thread started. Access at http://<your-ip>:8052")
        except Exception as dash_err:
            log.error(f"Failed to start dashboard thread: {dash_err}", exc_info=True)
            
    # Ensure this line is properly indented (not part of the if-block)
    log.info("Dashboard initialization completed.")

    # --- 6. Start Scheduler ---
    log.info("Starting scheduler...")
    try:
        scheduler.start_scheduler()
        log.info("Scheduler started successfully.")
    except Exception as e:
        log.critical(f"Scheduler failed unexpectedly: {e}", exc_info=True)
    finally:
        log.info("Application shutdown initiated.")

    log.info("========================================")
    log.info("== Cyclone v2 Trading Bot Stopped    ==")
    log.info("========================================")


if __name__ == "__main__":
    # Enable the main function call to start execution
    main()
