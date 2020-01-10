import time
import inspect
from functools import wraps
from typing import Tuple


def arg_value(arg_name, f, args, kwargs):
    if arg_name in kwargs:
        return kwargs[arg_name]

    i = f.__code__.co_varnames.index(arg_name)
    if i < len(args):
        return args[i]

    return inspect.signature(f).parameters[arg_name].default


def logger(begin_message: str = None, log_args: Tuple[str] = None, end_message: str = None, log_time: bool = True):
    def logger_decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if begin_message is not None:
                print(begin_message, end='\n' if log_args is None else '')

            if log_args is not None:
                arg_logs = [arg_name + '=' + str(arg_value(arg_name, f, args, kwargs)) for arg_name in log_args]
                print(', '.join(arg_logs))

            start_time = time.time()
            result = f(*args, **kwargs)
            spent_time = time.time() - start_time

            if end_message is not None:
                print(end_message)

            if log_time:
                print('（耗时', spent_time, '秒）', sep='')

            return result
        return decorated
    return logger_decorator
