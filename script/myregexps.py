"""
This file should contain all regexes that are often used, such that
they can easily globally be changed.
"""
# Matches 5, +5, -5, 5.0, +5.0, -5.0, ...
floating_number = r'(?:[+-]?\d+(?:\.\d+)?)'
floating_number_limited = "^" + floating_number + "$"

integer = r'(?:[+-]?\d+)'
integer_limited = "^" + integer + "$"
