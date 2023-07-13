import requests
import zipfile
import io, os
import logging
from _helpers import configure_logging, progress_retrieve

logger = logging.getLogger(__name__)

if __name__ == "__main__":# URL of the Zenodo repository zip file
    response = requests.get(snakemake.config["natura_repository"]["url"])
    zip_file = zipfile.ZipFile(io.BytesIO(response.content))

    # Extract the .tiff file
    for file_name in zip_file.namelist():
        if file_name.endswith(".tiff"):
            output_path = snakemake.output[0]
            output_path = output_path[:output_path.rfind("/")] #remove the file name after the slash
            zip_file.extract(file_name, path=output_path)

    # Close the zip file
    zip_file.close()
