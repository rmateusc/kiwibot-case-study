import os
from pathlib import Path

from dotenv import load_dotenv
from roboflow import Roboflow

CURR_DIR = Path(__file__).parent

load_dotenv(CURR_DIR.parent / ".env", override=True)


rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
project = rf.workspace("boxprediction").project("box-classification-s2q1a")
version = project.version(1)
dataset = version.download(
    "multiclass", location=str(CURR_DIR.parent / "data" / "box_classification")
)
