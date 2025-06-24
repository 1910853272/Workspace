# -*- coding: utf-8 -*-
"""

This script reproduces the experimental figures (Figure 3, Figure 4, Figure 5)  in the main text of "Principled Neuromorphic Reservoir Computing". 

"""

import os

# Simulations to reproduce all panels in Figure 3
os.system("python Figure_3_ac_Lorenz_fixed_seed.py")
os.system("python Figure_3_df_DoubleScroll_fixed_seed.py")
os.system("python Figure_3_gi_MackeyGlass_fixed_seed.py")

# Data to reproduce all panels in Figure 4
os.system("python Figure_4_a_Lorenz.py")
os.system("python Figure_4_b_DoubleScroll.py")
os.system("python Figure_4_c_MackeyGlass.py")

# Data to reproduce panels b & c in Figure 5
os.system("python Figure_5_Lorenz_loihi_vs_cpu.py")

# Simulations to reproduce all panels in Figure S.1
os.system("python Figure_S1_Kuramoto_Sivashinsky_fixed_seed.py")

# Simulations to reproduce all panels in Figure S.2
os.system("python Figure_S2_Kuramoto_Sivashinsky_vs_dimensionality.py")

# Data to reproduce Figure S.3
os.system("python Figure_S3_Lorenz_missing_a_f.py")
os.system("python Figure_S3_Lorenz_missing_g.py")

# Data to reproduce Figure S.4
os.system("python Figure_S4_MackeyGlass_radnom_SigmaPy.py")

# Data to reproduce Figure S.5
os.system("python Figure_S5_Lorenz_poly_kernel.py")

# Data to reproduce Figure S.6
os.system("python Figure_S6_Lorenz_ablation_orders.py")

# Data to reproduce Figure S.7
os.system("python Figure_S7_Lorenz_ablation_VSA_models.py")

# Data to reproduce Figure S.8
os.system("python Figure_S8_Lorenz_ablation_permutation.py")

# Data to reproduce Figure S.9
os.system("python Figure_S9_Lorenz_verification.py")

# Data to reproduce Figure S.10
os.system("python Figure_S10_Lorenz_training.py")

# Data to reproduce Figure S.11
os.system("python Figure_S11_Lorenz_noise.py")
