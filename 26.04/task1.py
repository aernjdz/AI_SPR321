import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Task 1
x = np.linspace(-10, 10, 400)
y = (x ** 2) * np.sin(x)
plt.figure(figsize=(8, 5))
plt.plot(x, y)
plt.title('Graph of f(x) = x^2 * sin(x)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.show()

# Task 2
np.random.seed(0)
data = np.random.normal(5, 2, 1000)
plt.figure(figsize=(8, 5))
plt.hist(data, bins=30, edgecolor='black')
plt.title('Histogram of Normally Distributed Data')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Task 3
hobbies = ['Reading', 'Gaming', 'Traveling', 'Music', 'Sports']
shares = [20, 25, 30, 15, 10]
plt.figure(figsize=(8, 5))
plt.pie(shares, labels=hobbies, autopct='%1.1f%%', startangle=140)
plt.title('My Favorite Hobbies')
plt.show()

# Task 4
np.random.seed(1)
fruit_types = ['Apple', 'Banana', 'Orange', 'Grape']
data = {
    fruit: np.random.normal(loc=150, scale=20, size=100) for fruit in fruit_types
}
df = pd.DataFrame(data)
plt.figure(figsize=(8, 5))
df.boxplot()
plt.title('Boxplot of Fruit Weights')
plt.ylabel('Weight (g)')
plt.grid(True)
plt.show()
