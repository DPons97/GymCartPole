import numpy as np

# Round the specified value number to the nearest rounding_interval multiplier
# value - The number to be rounded
# rounding_interval (default = 0.5) - The value that defines the rounding scale. value will be rounded to its nearest multiplier
# rounding_decimal (default = 1 decimal digit) - The amount of decimal digits that have to be preserved during the rounding
def round_to(value, rounding_interval = 0.5, rounding_decimal = 1):
    dec_round = round(value / rounding_interval)
    return round(dec_round * rounding_interval, rounding_decimal)