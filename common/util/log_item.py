"""
log_item.py

Defines log_item
"""

__author__ = "Julien Lefebvre, Hugo Chauvary"
__email__ = 'numerai_2021@protonmail.com'

import sys
import os
import functools
from inspect import getframeinfo, stack

from common.module.logger.logger import Logger


def log_item(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = Logger().logger # get a logger

        # creating a list of positional args
        # repr() is similar but more precise than str()
        args_str = [repr(a) for a in args]
        # creating a list of keyword args
        # f-string formats each arg as key=value
        # where the !r specifier means that repr()
        # is used to represent the value
        kwargs_str = [f"{k}={v!r}" for k, v in kwargs.items()]
        basic_args = ", ".join(args_str + kwargs_str)

        # generate file/function name for calling functions
        # __func.name__ will give the name of the caller function
        # ie. wrapper and caller file name ie log_item.py
        # using extra param to get the actual function name
        # by leveraging inspect.getframeinfo
        pyfile = getframeinfo(stack()[1][0])
        extra_args = {
            'func_name_override': f'{func.__globals__["__name__"]}.{func.__name__}',
            'file_name_override': os.path.basename(pyfile.filename)
        }

        # executing function and logging args
        if basic_args:
            # logger.info(f"begin function, args: {basic_args}", extra=extra_args)
            logger.info(f"begin function")
        else:
            logger.info(f"begin function, no arg", extra=extra_args)
        try:
            value = func(*args, **kwargs)
            if value:
                # logger.info(f"end function, returned {value!r}", extra=extra_args)
                logger.info(f"end function", extra=extra_args)
            else:
                logger.info(f"end function, no return", extra=extra_args)

            return value
        except:
            # log error if fails but don't raise
            logger.error(f"exception: {str(sys.exc_info()[1])}", extra=extra_args)
            pass

    return wrapper
