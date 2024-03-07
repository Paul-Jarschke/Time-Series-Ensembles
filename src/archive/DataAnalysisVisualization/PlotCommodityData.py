import pandas as pd
import matplotlib.pyplot as plt

# Replace 'your_dataset.xlsx' with the actual file path
file_path = 'C:/Users/Work/OneDrive/GAU/3. Semester/Statistisches Praktikum/Git/Ensemble-Techniques-TS-Forecasting/Data/CMO-Historical-Data-Monthly.xlsx'

# Replace 'Sheet1' with the actual sheet name
sheet_name = 'monthly_prices'

# Load the Excel file into a DataFrame
df = pd.read_excel(file_path, sheet_name=sheet_name, index_col=0)

# Display the DataFrame
print(df)



df = pd.DataFrame(df)
df.index = pd.to_datetime(df.index, format='%YM%m').strftime('%m/%y')

filtered_df = df.iloc[:, :4]
filtered_df = filtered_df.rename(columns={
    'crude_avg': 'Crude Oil (Avg.)',
    'gas_eu': 'Gas (EU)',
    'palmoil': 'Palm Oil',
    'phosphate_rock': 'Phosphate Rock'
})



# Plot each column
for i, column in enumerate(filtered_df.columns):
    unit = ["($/bbl)", "($/mmbtu)", "($/mt)", "($/mt)"][i]
    ylabel = f'Price {unit}'
    
    plt.plot(filtered_df.index, filtered_df[column], label=column)
    plt.title(column)
    plt.xlabel('Time')
    plt.ylabel(ylabel)
    ticks_to_display = df.index[::12]
    plt.xticks(ticks_to_display, ticks_to_display)
    plt.show()
