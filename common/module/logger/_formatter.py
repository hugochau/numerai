"""
_formatter.py

Fixes logging.Formatter misbehavior when calling
functions outside of their original location
"""

__author__ = "Julien Lefebvre, Hugo Chauvary"
__email__ = 'numerai_2021@protonmail.com'

import logging


class _Formatter(logging.Formatter):
    """
    The logging decorator logs the file and function name
    where it's defined, not where it's used.
    _Formatter corrects this behavior
    """

    def format(self, record):
        """
        Takes LogRecord object as input
        replaces funcName and filename
        with override values if existing
        """
        # func/filename override defined in log_item.py
        if hasattr(record, 'func_name_override'):
            record.funcName = record.func_name_override

        if hasattr(record, 'file_name_override'):
            record.filename = record.file_name_override

        # returns superclass original method
        # but with updated record attributes
        return super(_Formatter, self).format(record)