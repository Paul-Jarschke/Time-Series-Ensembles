import pandas as pd


def display_ranking_table(metrics_ranking):
    pd.set_option('display.float_format', '{:.2f}'.format)

    # Format performance metric values
    def format_numeric(val):
        if isinstance(val, float):
            return '{:.3f}'.format(val)
        return val
    formatted_metrics = metrics_ranking.applymap(format_numeric)

    display(formatted_metrics.style)
    pd.reset_option('display.float_format')