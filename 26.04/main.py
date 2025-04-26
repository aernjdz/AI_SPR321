import pandas as pd

df = pd.read_csv('orders.csv')
df['OrderDate'] = pd.to_datetime(df['OrderDate'])
df['TotalAmount'] = df['Quantity'] * df['Price']

# 3
total_revenue = df['TotalAmount'].sum()
average_total_amount = df['TotalAmount'].mean()
orders_per_customer = df['Customer'].value_counts()

# 4
high_value_orders = df[df['TotalAmount'] > 500]

# 5
sorted_by_order_date = df.sort_values(by='OrderDate', ascending=False)

# 6
orders_in_period = df[(df['OrderDate'] >= '2023-06-05') & (df['OrderDate'] <= '2023-06-10')]

# 7
category_grouped = df.groupby('Category').agg({
    'Product': 'count',
    'TotalAmount': 'sum'
}).rename(columns={'Product': 'ProductCount', 'TotalAmount': 'TotalSales'})

# 8
top_3_customers = df.groupby('Customer')['TotalAmount'].sum().sort_values(ascending=False).head(3)

print('Total Revenue:', total_revenue)
print('Average TotalAmount:', average_total_amount)
print('Orders per Customer:\n', orders_per_customer)
print('High Value Orders:\n', high_value_orders)
print('Sorted by OrderDate Descending:\n', sorted_by_order_date)
print('Orders from June 5 to June 10:\n', orders_in_period)
print('Category Grouped:\n', category_grouped)
print('Top 3 Customers:\n', top_3_customers)
