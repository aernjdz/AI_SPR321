import matplotlib.pyplot as plt
import numpy as np

# 1. Plot of y = x^2 * sin(x)
x = np.linspace(-10, 10, 1000)
y = x**2 * np.sin(x)

plt.plot(x, y)
plt.title("Graph of y = x²·sin(x)")
plt.xlabel("x")
plt.ylabel("y = x²·sin(x)")
plt.grid()
plt.show()

# 2. Histogram of normally distributed data
data = np.random.normal(loc=5, scale=2, size=1000)

plt.hist(data, bins=30, color='skyblue', edgecolor='black')
plt.title("Histogram of Normal Distribution")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

# 3. Pie chart
labels = ['Programming', 'Dancing', 'Sleeping', 'Travelling', "(._.')"]
sizes = [30, 45, 3.0, 30, 5.5]
total = sum(sizes)
normalized_sizes = [(value / total) * 100 for value in sizes]

plt.pie(normalized_sizes, labels=labels, autopct='%1.1f%%', startangle=90)
plt.title("Me")
plt.axis('equal') 
plt.show()

# 4. Box plot of fruit weights
np.random.seed(42)
apple_weights = np.random.normal(loc=150, scale=10, size=100)
banana_weights = np.random.normal(loc=120, scale=15, size=100)
orange_weights = np.random.normal(loc=130, scale=12, size=100)
pear_weights = np.random.normal(loc=140, scale=8, size=100)

data = [apple_weights, banana_weights, orange_weights, pear_weights]
fruit_names = ['Apples', 'Bananas', 'Oranges', 'Pears']

box = plt.boxplot(data, patch_artist=True, labels=fruit_names)

colors = ['red', 'yellow', 'orange', 'green']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

plt.title('Box Plot of Fruit Weights')
plt.xlabel('Fruits')
plt.ylabel('Weight (g)')
plt.grid()
plt.tight_layout()
plt.show()

# 5. Scatter plot of uniformly distributed points
x = np.random.uniform(0, 1, 100)
y = np.random.uniform(0, 1, 100)

plt.scatter(x, y, color='green', alpha=0.6)
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Scatter Plot of Uniform Distribution')
plt.grid()
plt.show()

# 6. Plot of sin(x), cos(x), and sin(x)+cos(x)
x = np.linspace(-10, 10, 1000)
y = np.sin(x)
z = np.cos(x)
d = y + z

plt.plot(x, y, label='sin(x)')
plt.plot(x, z, label='cos(x)')
plt.plot(x, d, label='sin(x) + cos(x)')
plt.title("Graph of Trigonometric Functions")
plt.legend(loc='upper right')
plt.xlabel("x")
plt.ylabel("Value")
plt.grid()
plt.show()
