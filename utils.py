""" utility programs for FRAC """

import numpy as np
import sys
from traceback import extract_stack




def bs_name_func(back: int=2):
    """
    get the name of the current function, or further back in stack

    :param int back: 2 is current function, 3 the function that called it etc

    :return: a string
    """
    stack = extract_stack()
    filename, codeline, funcName, text = stack[-back]
    return funcName


def print_stars(title: str = None, n: int = 70) -> None:
    """
    prints a title within stars

    :param str title:  title

    :param int n: number of stars on line

    :return: prints a starred line, or two around the title
    """
    line_stars = '*' * n
    print()
    print(line_stars)
    if title:
        print(title.center(n))
        print(line_stars)
    print()


def bs_error_abort(msg: str="error, aborting"):
    """
    report error and abort

    :param str msg: a message

    :return: exit with code 1
    """
    print_stars(f"{bs_name_func(3)}: {msg}")
    sys.exit(1)


def npmaxabs(arr: np.array) -> float:
    """
    maximum absolute value in an array

    :param np.array arr: Numpy array

    :return: a float
    """
    return np.max(np.abs(arr))

