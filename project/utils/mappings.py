# Mapping frequency aliases to description
FREQ_MAPPING = {
    'B': 'business day',
    'C': 'custom business day',
    'D': 'calendar day',
    'W': 'weekly',
    'ME': 'month end',
    'SME': 'semi-month end (15th and end of month)',
    'BME': 'business month end',
    'CBME': 'custom business month end',
    'MS': 'month start',
    'SMS': 'semi-month start (1st and 15th)',
    'BMS': 'business month start',
    'CBMS': 'custom business month start',
    'QE': 'quarter end',
    'BQE': 'business quarter end',
    'QS': 'quarter start',
    'BQS': 'business quarter start',
    'YE': 'year end',
    'BYE': 'business year end',
    'YS': 'year start',
    'BYS': 'business year start',
    'h': 'hourly',
    'bh': 'business hour',
    'cbh': 'custom business hour',
    'min': 'minutely',
    's': 'secondly',
    'ms': 'milliseconds',
    'us': 'microseconds',
    'ns': 'nanoseconds'
}

# Mapping frequency aliases to periodicity number
SEASONAL_FREQ_MAPPING = {
    'MS': 12,
    'M': 12,
    'B': 252,
    'D': 365,
    'W': 52,
    'Q': 4,
    'Y': 10
}