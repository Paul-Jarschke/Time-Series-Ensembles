import inspect
import logging


def vprint(*args):
    """
    Print function that only prints when globally verbose is True.

    Parameters:
    - *args: Strings to be printed.
    """
    if inspect.currentframe().f_back.f_locals["verbose"]:
        ### print(*args)
        ### logging.info('\033[0m' + arg)
        # Log each argument using the logging module
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
    # Extract total seconds
    s = tdelta.seconds
    # Calculate hours
    hours = s // 3600
    # remaining seconds
    s = s - (hours * 3600)
    # Calculate minutes
    minutes = s // 60
    # Remaining seconds
    seconds = s - (minutes * 60)

    # Format the timedelta into a string
    formatted_tdelta = "{:02}h {:02}m {:02}s".format(
        int(hours), int(minutes), int(seconds)
    )

    # Remove leading zeros for hours and minutes if they are zero
    formatted_tdelta = formatted_tdelta.replace("00h ", "").replace("00m ", "")

    return formatted_tdelta
