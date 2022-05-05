"""
unzip_file.py

Unzip archive
"""

__author__ = "Julien Lefebvre, Hugo Chauvary"
__email__ = 'numerai_2021@protonmail.com'


import zipfile

from common.util.log_item import log_item

@log_item
def unzip_file(filepath: str, target_folder: str) -> None:
    """
    Util function for unzipping file

    args:
        - filepath: path to file to be unzipped
        - target_folder: target location for unzipped files
    """
    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        zip_ref.extractall(target_folder)
