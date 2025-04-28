import numpy as np
import matplotlib.pyplot as plt

# Parameters
Gon =  2.7374E-04
Goff = 2.5981E-04
Ptot = 40
A = 9.6

Y = []
N = []

# Define GLTP function to calculate LTP conductance
def GLTP(x):
    B = (Gon - Goff) / (1 - np.exp(-Ptot / A))
    G = B * (1 - np.exp(-x / A)) + Goff
    return G

# Define GLTD function to calculate LTD conductance
def GLTD(x):
    B = (Gon - Goff) / (1 - np.exp(-Ptot / A))
    # Ensure (x - 20) is non-negative to avoid issues
    G = -B * (1 - np.exp(-max(x-20, 0) / A)) + Gon
    return G

# LTP Part
for i in range(1, 21):
    y = GLTP(i)
    Y.append(y)
    N.append(i)

# LTD Part
for i in range(21, 41):
    y = GLTD(i)
    Y.append(y)
    N.append(i)
