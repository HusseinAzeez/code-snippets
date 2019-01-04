"""
    Autour: Eraser (ตะวัน)
"""

# Standard library imports
import os
import re

# Third-party imports
import requests
from wand.image import Image


def create_file_list(path, extension='.png'):
    """Creates a files list containing all the files
        in the specified directory with the provided file extension

        Args:
            path: The path to the directory containing the required files
            extension: The files extensions (default: .png)

        Return:
            file_list: Python list containing all the files
    """
    file_list = []
    for root, _, files in os.walk(path, topdown=False):
        for name in files:
            if name.endswith(extension):
                full_name = os.path.join(root, name)
                file_list.append(full_name)

    return file_list


def download_pdf(pdf_url):
    """Downloads a PDF from a public storage and stored it locally.

        Args:
            paf_url: link to a downloadable PDF.

        Return:
            None.
    """
    response = requests.get(pdf_url, allow_redirects=True)
    open('./data/raw/full.pdf', 'wb').write(response.content)


def convert_pdf(pdf_path):
    """Converts locally stored PDF into image and save it on the same directory.

        Args:
            paf_path: local path to a PDF for converting it to TIFF format.

        Return:
            None.
    """
    with Image(filename=pdf_path, resolution=300, format="pdf") as pdf:
        pdf.convert('tiff')
        pdf.save(filename='./data/raw/full.tiff')


def split_filename(path):
    """Splits the provided file paths into three different strings.
        Each file name contains row number, file name, and the extenstion.
        Example:
            1.0.png or 2.3.png

        Args:
            path: The path to the directory containing the required files.

        Return:
            row: The number corsponding to row order in the original PDF.
            name: The file name (A number).
            extenstion: File extenstion (Default .png)
    """
    filename = os.path.basename(path)
    name, extension = os.path.splitext(filename)
    region = name.split('.')[0]

    return region, name, extension


def natural_sort(path):
    """Sorts the given list in the way that humans expect.

        Args:
            path: The file path.

        Return:
            A list of the sorted files.
    """
    numbers = re.compile('([0-9]+)')
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(numbers, path)]
