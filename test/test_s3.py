"""
test_s3.py

Implements s3 unit tests
"""

__author__ = "Julien Lefebvre, Hugo Chauvary"
__email__ = 'numerai_2021@protonmail.com'

import os
import sys
import inspect

import pytest
import botocore

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from common.module.aws.s3 import S3
from common.config.constant import DATA_FOLDER


def test_download_file():
    s3 = S3()
    s3.download_file('hello.txt', 'signalsdata', 'credential')

    assert os.path.exists('data/credential/hello.txt')


@pytest.mark.xfail(raises=botocore.exceptions.ClientError)
def test_download_file_not_exists():
    s3 = S3()
    s3.download_file('hugo.txt', 'signalsdata', 'credential')
