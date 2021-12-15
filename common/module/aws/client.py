"""
client.py

Defines Client
"""

__author__ = "Julien Lefebvre, Hugo Chauvary"
__email__ = 'numerai_2021@protonmail.com'

from boto3 import Session
from botocore.config import Config
from botocore import client

from common.config.constant import AWS_REGION
from common.util.get_aws_creds import get_aws_creds


class Client:
    @staticmethod
    def set_client(object) -> client:
        """
        Set AWS client

        returns:
            - botocore.client
        """
        creds = get_aws_creds()

        # build session object
        session = Session(aws_access_key_id=creds['access_key_id'],
                                aws_secret_access_key=creds['secret_access_key'])

        # build client object
        return session.client(service_name=object,
                              config=Config(region_name=AWS_REGION))
