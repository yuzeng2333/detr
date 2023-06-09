import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the values for x and y
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)

# Create a grid of points
x, y = np.meshgrid(x, y)

# Define the function z = x*y
#z = x * x + y * y
z = x + y * y

# Create a figure
fig = plt.figure()

# Create a 3D axis
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(x, y, z, cmap='viridis')

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show the plot
plt.show()

