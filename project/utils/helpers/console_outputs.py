import inspect
import logging


def vprint(*args):
    """
    Print function that only prints when globally verbose is True.

    Parameters:
    - *args: Strings to be printed.
    """
    if inspect.currentframe().f_back.f_locals['verbose']:
        # print(*args)
        # logging.info('\033[0m' + arg)
        for arg in args:
            logging.info(arg)


def strfdelta(tdelta):
    """
    Format a timedelta object into a string of hours, minutes, and seconds.

    Parameters:
        tdelta (timedelta): A timedelta object representing the time difference.

    Returns:
        str: A formatted string representing the time difference in the format 'HHh MMm SSs'.
    """
    # Extract seconds
    s = tdelta.seconds

    # hours
    hours = s // 3600
    # remaining seconds
    s = s - (hours * 3600)
    # minutes
    minutes = s // 60
    # remaining seconds
    seconds = s - (minutes * 60)

    # total time
    formatted_tdelta = '{:02}h {:02}m {:02}s'.format(int(hours), int(minutes), int(seconds))
    formatted_tdelta = formatted_tdelta.replace('00h ', '').replace('00m ', '')

    return formatted_tdelta
