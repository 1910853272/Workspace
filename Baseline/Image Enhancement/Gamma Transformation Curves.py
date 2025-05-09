import numpy as np
import matplotlib.pyplot as plt

# Define the input intensity L from 0 to 1
L = np.linspace(0, 1, 500)

# Define gamma values to plot the curves
gamma_values = [0.04, 0.10, 0.20, 0.40, 0.67, 1.0, 1.5, 2.5, 5.0, 10.0, 25.0]

# Create the plot
plt.figure(figsize=(8, 6))

# Plot each curve for the different gamma values
for gamma in gamma_values:
    S = L ** gamma  # Gamma transformation
    plt.plot(L, S, label=f'γ = {gamma}')

# Add labels and title
plt.title('Gamma Transformation Curves')
plt.xlabel('Input Grayscale r')
plt.ylabel('Output Grayscale s')
plt.xlim([0, 1])
plt.ylim([0, 1])

# Add grid and legend
plt.grid(True)
plt.legend()

# Show the plot

plt.savefig('Gamma Transformation Curves.png')
plt.show()