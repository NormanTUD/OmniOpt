import math
import itertools
import random
import re

def random_number_between_x_and_y(x, y):
    if x > y:
        tmp = y
        y = x
        x = tmp
    return random.randrange(x, y)

def get_largest_divisors(n):
    if n == 0 or n == 1:
        return {'x': 2, 'y': 1}

    if n == 2:
        return {'x': 2, 'y': 1}

    n = nearest_non_prime(n)

    sqrtnr = int(math.ceil(math.sqrt(n)))

    for i in reversed(range(2, sqrtnr + 1)):
        if n % i == 0:
            y = int(n / i)
            if i > y:
                return {'x': i, 'y': y}
            else:
                return {'x': y, 'y': i}
    raise Exception("Couldn't get any divisors for n = " + str(n))

def nearest_non_prime(n):
    while is_prime(n):
        n = n + 1
    return n

def is_prime(n):
    if n == 1:
        return False
    if n % 2 == 0 and n > 2:
        return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
        return True
    return True

def is_integer(tdata):
    prog = re.compile(r"^\d+$")
    if not isinstance(tdata, str):
        tdata = str(tdata)
    result = prog.match(tdata)
    if result:
        return 1
    else:
        return 0

def findsubsets(S, m):
    data = set(itertools.combinations(S, m))
    return data

def get_index_of_maximum_value(array):
    maximum_index = None
    maximum = float("-inf")
    index = 0
    for item in array:
        if item > maximum:
            maximum = item
            maximum_index = index
        index = index + 1
    return maximum_index

def get_index_of_minimum_value(array):
    minimum_index = None
    minimum = float('inf')
    index = 0
    for item in array:
        if item < minimum:
            minimum = item
            minimum_index = index
        index = index + 1
    return minimum_index

def get_min_value(array):
    minimum = float('inf')
    for item in array:
        if item < minimum:
            minimum = item
    return minimum


def get_max_value(array):
    maximum = float('-inf')
    for item in array:
        if item > maximum:
            maximum = item
    return maximum

