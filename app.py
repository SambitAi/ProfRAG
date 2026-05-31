import os

# Ignore optional vision-backend import noise from transformers under Streamlit's
# module watcher. This app does not use vision models.
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

from ui import main


if __name__ == "__main__":
    main()
