#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

print("Libraries imported successfully!")

#loading the dataset
df = pd.read_csv(r"C:\Users\iampr\Downloads\FDI_in_India.csv")

print("Dataset loaded successfully!")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

#displaying the dataset
df.head()
df.tail()

#undrstanding the dataset
print("Dataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

#checking for missing values
missing_values = df.isnull().sum()

print("Missing values in each column:")
print(missing_values)

#missing value percentage
missing_percentage = (df.isnull().sum() / len(df)) * 100

print("Missing value percentage:")
print(missing_percentage)

#checking for duplicates
duplicates = df.duplicated().sum()

print("Number of duplicate rows:", duplicates)

#cleaning sector names
df['Sector'] = df['Sector'].str.strip()

print("Sector names cleaned successfully!")

#checking numeric data
year_columns = df.columns[1:]

for column in year_columns:
    df[column] = pd.to_numeric(df[column], errors='coerce')

print("Financial year columns converted to numeric.")

#checking for negative values
negative_values = (df[year_columns] < 0).sum().sum()

print("Number of negative FDI values:", negative_values)

#creating total FDI by sector
df['Total FDI'] = df[year_columns].sum(axis=1)

df[['Sector', 'Total FDI']].sort_values(
    by='Total FDI',
    ascending=False
).head(10)

#Average FDI by sector
df['Average FDI'] = df[year_columns].mean(axis=1)

df[['Sector', 'Average FDI']].sort_values(
    by='Average FDI',
    ascending=False
).head(10)

#Highest FDI year
#Firstly calculating total FDI by each year
yearly_fdi = df[year_columns].sum()

print(yearly_fdi)
#Then finding the year with the highest FDI
highest_fdi_year = yearly_fdi.idxmax()
highest_fdi_value = yearly_fdi.max()

print("Highest FDI year:", highest_fdi_year)
print("FDI value:", highest_fdi_value)

#Lowest FDI year
lowest_fdi_year = yearly_fdi.idxmin()
lowest_fdi_value = yearly_fdi.min()

print("Lowest FDI year:", lowest_fdi_year)
print("FDI value:", lowest_fdi_value)

#Overall FDI 
total_fdi = yearly_fdi.sum()

print("Total FDI during the study period:", total_fdi)

#Year Wise FDI Trend
plt.figure(figsize=(12, 6))

plt.plot(
    yearly_fdi.index,
    yearly_fdi.values,
    marker='o'
)

plt.title("FDI Trend in India (2000-01 to 2016-17)")
plt.xlabel("Financial Year")
plt.ylabel("FDI")
plt.xticks(rotation=45)
plt.grid(True)

plt.show()

#Top 10 Sectors by Total FDI
top_10_sectors = df[['Sector', 'Total FDI']].sort_values(
    by='Total FDI',
    ascending=False
).head(10)

top_10_sectors

#Top 10 Sectors Visualization
plt.figure(figsize=(10, 6))

plt.barh(
    top_10_sectors['Sector'],
    top_10_sectors['Total FDI']
)

plt.title("Top 10 Sectors by Total FDI")
plt.xlabel("Total FDI")
plt.ylabel("Sector")

plt.gca().invert_yaxis()

plt.show()

#Sector Contribution
df['FDI Contribution %'] = (
    df['Total FDI'] / df['Total FDI'].sum()
) * 100

df[['Sector', 'Total FDI', 'FDI Contribution %']].sort_values(
    by='FDI Contribution %',
    ascending=False
).head(10)

#Year over year growth
yoy_growth = yearly_fdi.pct_change() * 100

print("Year-over-Year FDI Growth:")
print(yoy_growth)
#visualizing it
plt.figure(figsize=(12, 6))

plt.bar(
    yoy_growth.index,
    yoy_growth.values
)

plt.title("Year-over-Year FDI Growth")
plt.xlabel("Financial Year")
plt.ylabel("Growth (%)")
plt.xticks(rotation=45)

plt.axhline(0, linewidth=1)

plt.show()

#Finding strongest growth year
growth_year = yoy_growth.idxmax()
growth_value = yoy_growth.max()

print("Highest FDI growth occurred in:", growth_year)
print("Growth:", round(growth_value, 2), "%")

#Finding biggest decline year
decline_year = yoy_growth.idxmin()
decline_value = yoy_growth.min()

print("Largest FDI decline occurred in:", decline_year)
print("Decline:", round(decline_value, 2), "%")

#Best performing sector 
best_sector = df.loc[df['Total FDI'].idxmax(), 'Sector']
best_sector_value = df['Total FDI'].max()

print("Top performing sector:", best_sector)
print("Total FDI:", best_sector_value)

#Sector  volatility
df['FDI Volatility'] = df[year_columns].std()

df[['Sector', 'FDI Volatility']].sort_values(
    by='FDI Volatility',
    ascending=False
).head(10)

#Most stable sector
df[['Sector', 'FDI Volatility']].sort_values(
    by='FDI Volatility'
).head(10)




#Creating an investment opportunity score
df['FDI Score'] = (
    df['Total FDI'] / df['Total FDI'].max()
) * 100

df['Average Score'] = (
    df['Average FDI'] / df['Average FDI'].max()
) * 100

df['Stability Score'] = (
    1 - (
        df['FDI Volatility'] /
        df['FDI Volatility'].max()
    )
) * 100

#Now creating the overall score
df['Investment Opportunity Score'] = (
    0.5 * df['FDI Score'] +
    0.3 * df['Average Score'] +
    0.2 * df['Stability Score']
)

#Ranking Sectors
investment_ranking = df[
    [
        'Sector',
        'Total FDI',
        'Average FDI',
        'FDI Volatility',
        'Investment Opportunity Score'
    ]
].sort_values(
    by='Investment Opportunity Score',
    ascending=False
)

investment_ranking.head(10)

#Creating the final cleaned dataset
long_df = df.melt(
    id_vars=['Sector'],
    value_vars=year_columns,
    var_name='Financial Year',
    value_name='FDI'
)

long_df.head()

#Adding year number
long_df['Year'] = long_df['Financial Year'].str[:4].astype(int)

long_df.head()

#Export cleaned dataset
long_df.to_csv(
    "FDI_Cleaned_Long_Format.csv",
    index=False
)

print("Cleaned dataset exported successfully!")

#Exporting sector summary
sector_summary = df[
    [
        'Sector',
        'Total FDI',
        'Average FDI',
        'FDI Volatility',
        'FDI Contribution %',
        'Investment Opportunity Score'
    ]
]

sector_summary.to_csv(
    "FDI_Sector_Summary.csv",
    index=False
)

print("Sector summary exported successfully!")

#Exporting yearly summary
yearly_summary = pd.DataFrame({
    'Financial Year': yearly_fdi.index,
    'Total FDI': yearly_fdi.values,
    'YoY Growth %': yoy_growth.values
})

yearly_summary.to_csv(
    "FDI_Yearly_Summary.csv",
    index=False
)

print("Yearly summary exported successfully!")