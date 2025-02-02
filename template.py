import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')


project_name = "Video-Analytics-for-Human-Activity-Recognition"


list_of_files = [
    ".github/workflows/.gitkeep",
    "setup.py",
    "requirements.txt",
    "app.py",
    "Dockerfile",
    "templates/index.html",
    "static/style.css",
    "research/notebook.ipynb",
    "src/__init__.py",
    "src/exception.py",
    "src/logger.py",
    "src/utils.py",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)


    if filedir !="":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")


    else:
        logging.info(f"{filename} is already exists")