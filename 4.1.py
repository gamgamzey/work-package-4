import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import quad


#part 1: Aerodynamic Data Import & Interpolation

def load_xflr5_spanwise(filename):
    #skip the 20 metadata/header lines and read the 38 spanwise rows
    data = np.genfromtxt(filename, skip_header=20, max_rows=38)

    #keep only the right half-wing (y >= 0) for root-to-tip integration
    right_wing = data[data[:, 0] >= 0]  #NOTE TO SELF: this is a mask over the original data

    y = right_wing[:, 0]
    chord = right_wing[:, 1]
    cl = right_wing[:, 3]
    cd = right_wing[:, 4] + right_wing[:, 5]  #total Cd = profile drag Cd + induced drag Cd
    cm = right_wing[:, 7]  # CmAirf@chord/4
    return y, chord, cl, cd, cm


#load both alpha = 0 deg and alpha = 10 deg exports using the def. function
y0, c0, cl0, cd0, cm0 = load_xflr5_spanwise("MainWing_a=0.00_v=10.00ms.txt")
y10, c10, cl10, cd10, cm10 = load_xflr5_spanwise("MainWing_a=10.00_v=10.00ms.txt")

#create callable functions along the wingspan
c_func = interp1d(y10, c10, kind="cubic", fill_value="extrapolate")
CL0_func = interp1d(y0, cl0, kind="cubic", fill_value="extrapolate")
CL10_func = interp1d(y10, cl10, kind="cubic", fill_value="extrapolate")
CD0_func = interp1d(y0, cd0, kind="cubic", fill_value="extrapolate")
CD10_func = interp1d(y10, cd10, kind="cubic", fill_value="extrapolate")
Cm0_func = interp1d(y0, cm0, kind="cubic", fill_value="extrapolate")
Cm10_func = interp1d(y10, cm10, kind="cubic", fill_value="extrapolate")

#part 2: Load Scaling

#dummy values (where do i get the rest of these from?)
b_half = 9.10  #clanker got this from somewhere within the files, gotta double check
b = 2 * b_half
S_ref = 2 * quad(c_func, 0, b_half)[0]  #computes definite integral using quad function, y-coord. values from c_func
#NOTE TO SELF: the quad function returns a tuple, gotta take the first item

#TODO: how do i get these dummy values?
n = 1 #load factor
W = 1 #aircraft weight
q = 1 #dynamic pressure

#1. wing design load coefficient
CL_des = n * W /(q * S_ref)

#inside of the integral for the global wing CL
def cl0_weight(y):
    return CL0_func(y) * c_func(y)

def cl10_weight(y):
    return CL10_func(y) * c_func(y)

#global wing CL at alpha = 0 and alpha = 10 deg
CL0_wing = (2 / S_ref) * quad(cl0_weight, 0, b_half)[0]
CL10_wing = (2 / S_ref) * quad(cl10_weight, 0, b_half)[0]

#req. a.o.a. from lift-curve slope
alpha = 0.0 + (10.0 - 0.0) * (CL_des - CL0_wing) / (CL10_wing - CL0_wing)

#2. 