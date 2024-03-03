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
    'MS': 12,  # 12 months per year
    'M': 12,  # 12 months per year
    'B': 5,  # 5 business days per week
    'D': 7,  # 7 days per week
    'W': 4,  # 4 weeks per month
    'Q': 4,  # 4 quarters per year
    'Y': 10  # 10 years per decade
}