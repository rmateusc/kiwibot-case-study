import os
import zipfile
from pathlib import Path

import gdown
from dotenv import load_dotenv
from roboflow import Roboflow

CURR_DIR = Path(__file__).parent

load_dotenv(CURR_DIR.parent / ".env", override=True)


def download_cv_data():
    # connect with roboflow
    rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
    # define project to download
    project = rf.workspace("boxprediction").project("box-classification-s2q1a")
    version = project.version(1)
    # download the dataset
    dataset = version.download(
        "multiclass",
        location=str(CURR_DIR.parent / "data" / "box_classification"),
        overwrite=True,
    )
    print(
        f'succcesfully donwloaded `box_classification` data to {str(CURR_DIR.parent / "data" / "box_classification")}'
    )


def download_llm_data():
    # download data from google drive
    file_ids = [
        "1h17XM7o8DaQenP_b3XbtJ4MSm2gwXgIK",
        "1IahfqMcxWtsLYvgQNRF6URn1d29oL0Gk",
    ]
    for file_id in file_ids:
        url = f"https://drive.google.com/uc?id={file_id}"
        output_zip = "data.zip"
        extract_to = str(CURR_DIR.parent / "data" / "GNHK")

        # Download the file from Google Drive
        print("Downloading the zip file from Google Drive...")
        gdown.download(url, output_zip, quiet=False)

        # Create a directory to extract the contents, if it doesn't exist
        if not os.path.exists(extract_to):
            os.makedirs(extract_to)

        # Unzip the downloaded file
        print("Extracting the zip file...")
        with zipfile.ZipFile(output_zip, "r") as zip_ref:
            zip_ref.extractall(extract_to)

        print(
            "Extraction complete. Files are saved in the '{}' directory.".format(
                extract_to
            )
        )

        # Remove the zip file after extraction
        os.remove(output_zip)


if __name__ == "__main__":
    download_cv_data()
    download_llm_data()
