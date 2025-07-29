#Import Libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Load Dataset
df = pd.read_csv('shop_sales.csv')  # replace with your CSV file
df.head()

#Basic Analysis
df.info()
df.describe()
df.isnull().sum()

#Add a "Total" Column (if not available)
df['Total'] = df['Quantity'] * df['Price']

# Top-Selling Products
df.groupby('Product')['Total'].sum().sort_values(ascending=False).head(10).plot(kind='bar')
plt.title('Top 10 Products by Sales')

#Sales by Category
df.groupby('Category')['Total'].sum().plot(kind='pie', autopct='%1.1f%%')
plt.title('Sales by Category')
plt.ylabel('')

#Sales Over Time
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df['Total'].resample('D').sum().plot()
plt.title('Daily Sales')
