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
b_half = 9.10  #TODO: clanker got this from somewhere within the files, gotta double check where
b = 2 * b_half
S_ref = 2 * quad(c_func, 0, b_half)[0]  #computes definite integral using quad function, y-coord. values from c_func
#NOTE TO SELF: the quad function returns a tuple, gotta take the first item

#TODO: how do i get these dummy values? Anna said they'll come later
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
alpha_deg = 0.0 + (10.0 - 0.0) * (CL_des - CL0_wing) / (CL10_wing - CL0_wing)
alpha_rad = np.radians(alpha_deg)

#2. interpolate distributions at alpha_deg using interp1d (did not want to lambda function it)
def make_coef_at_flight(c0_func, c10_func):
    def coef_func(y):
        return c0_func(y) + (alpha_deg / 10.0) * (c10_func(y) - c0_func(y))
    return coef_func

#NOTE TO SELF: coefficients for the airfoil, not the entire wing!!!
Cl_des = make_coef_at_flight(CL0_func, CL10_func)
Cd_des = make_coef_at_flight(CD0_func, CD10_func)
Cm_des = make_coef_at_flight(Cm0_func, Cm10_func)

#3. dimensional loads and pitching moment per unit span (formula to put on Overleaf!!)
def L_prime(y):
    l_prime = q * c_func(y) * Cl_des(y)
    return l_prime

def D_prime(y):
    d_prime = q * c_func(y) * Cd_des(y)
    return d_prime

def M_prime(y): #at quarter-chord
    m_prime = q * c_func(y) * Cm_des(y)
    return m_prime

#4. normal force perp. to chord line
def N_prime(y):
    n_prime = L_prime(y) * np.cos(alpha_rad) + D_prime(y) * np.sin(alpha_rad)
    return n_prime

#part 3: Internal Loads Integration (proper 4.1)

#inputs: dummy inertial load distribution and chordwise positions
#TODO: change these dummy values later!
W_wing = 0.1 * W #wing weight
x_ac_frac = 0.2 #aerodynamic center frac.
x_ea_frac = 0.3 #elastic (so shear) center frac.
x_cg_frac = 0.4 #center of gravity frac.

#distributing weight of wing along chord
def w_inertial(y):
    return (W_wing / S_ref) * c_func(y)

#pos. of values along chord
def x_ac(y):
    return x_ac_frac * c_func(y)

def x_ea(y):
    return x_ea_frac * c_func(y)

def x_cg(y):
    return x_cg_frac * c_func(y)

#net distributed vertical load
def qz(y):
    return N_prime(y) - n * w_inertial(y)

#net distributed torque
def t_dist(y):
    return M_prime(y) + N_prime(y) * (x_ea(y) - x_ac(y)) - n * w_inertial(y) * (x_ea(y) - x_cg(y))

#integrating distributed loads to get shear, bending moment and torsion
def S(y): #shear
    return quad(qz, y, b_half)[0]

def M(y): #moment
    def moment_integrand(theta):
        return (theta - y) * qz(theta)
    return quad(moment_integrand, y, b_half)[0]

def T(y): #torque
    return quad(t_dist, y, b_half)[0]