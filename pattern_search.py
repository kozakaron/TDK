"""________________________________Libraries________________________________"""

import numpy as np
import time
import os
import random
from scipy.integrate import solve_ivp
import importlib

# dot.notation access to dictionary attributes
class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

# my own file:
import diffeq as de
importlib.reload(de)



"""________________________________Basic functions________________________________"""

# Returns a random point from the parameter space
def rand_point(ranges):
    point = []
    for key in ranges:
        point.append(random.uniform(ranges[key][0], ranges[key][1]))
    return point

# Returns True, if point is in the parameter space, False if not
def in_range(point, ranges):
    for i, key in enumerate(ranges):
        if point[i] < ranges[key][0] or ranges[key][1] < point[i]:
            return False
    return True

# Returns the smallest element of a list. Compare by last element
def minimal(points):
    best = points[0]
    for point in points:
        if point[-1] < best[-1]:
            best = point
    return best

# Returns a value to minimize: log10(Energy [MJ/kg])
def Energy(data):   
    Energy = data[17]
    if not np.isfinite(Energy) or Energy < 0 or data[1] == 2:
        return 30.0
    else:
        return np.log10(Energy)
    
# Calculates a point
def calculate_point(parameters, step_num=0, print_it=False):
    [R_E, ratio, P_inf, alfa_M, T_inf, surfactant] = parameters
    parameters = [0, R_E, ratio, P_inf, alfa_M, T_inf, de.VapourPressure(T_inf), de.par_indexOfArgon, surfactant]
    data = de.solve(parameters)
    if print_it:
        print(f'step_num: {step_num}; error: {data[1]}, steps: {data[2]}, runtime: {data[3]: .2f} s  |  R_E={(data[4]*1e6): .2f} [um]; ratio={data[5]: .2f}; ' + 
              f'P_inf={(1e-5*data[6]): .2f} [bar]; alfa_M={data[7]: .2f}; T_inf={data[8]: .2f} [K]; surfactant={data[11]: .2f}; Energy={data[17]: .0f} [MJ/kg]')
    return Energy(data), data



"""________________________________Parameter space________________________________"""
ranges = dotdict(dict(
    R_E=[0.1e-6, 30.0e-6],
    ratio=[1.5, 30.0],
    P_inf=[0.5e5, 100.0e5],
    alfa_M=[0.05, 0.40],
    T_inf=[273.15+5.0, 273.15+50.0],
    surfactant=[0.25, 1.0],
))



"""________________________________Pattern search________________________________"""
def pattern_search(run_num, ranges=ranges, max_steps=200, decay=0.85, min_step=0.005, first_step=0.2, stop_condition=0.0005):
# Setup
    start = time.process_time()
    last_point = rand_point(ranges)
    energy, data = calculate_point(last_point)
    last_point.append(energy)
    best_points = [last_point]
    datas = [data]
    step_num = 1
    loc_min = False
# Start steps
    while len(best_points) < max_steps and not loc_min:
        new_points = []
        # Check around in all directions:
        for i, key in enumerate(ranges):
            delta = abs(ranges[key][1] - ranges[key][0])
            delta = min_step*delta + first_step*delta*decay**step_num
            
            new_point = last_point[:-1]
            new_point[i] = last_point[i] - delta
            if in_range(new_point, ranges):
                energy, data = calculate_point(new_point)
                new_point.append(energy)
                new_points.append(new_point)
                datas.append(data)
                
            new_point = last_point[:-1]
            new_point[i] = last_point[i] + delta
            if in_range(new_point, ranges):
                energy, data = calculate_point(new_point)
                new_point.append(energy)
                new_points.append(new_point)
                datas.append(data)
                
        # Determine next step, save:
        last_point = minimal(new_points)
        best_points.append(last_point)
        step_num += 1
        
        # Stop condition:
        if step_num > 10:
            if abs(best_points[-2][-1] - best_points[-1][-1]) < stop_condition * best_points[-1][-1] and abs(best_points[-4][-1] - best_points[-3][-1]) < stop_condition * best_points[-1][-1]:
                loc_min = True
                
    end = time.process_time()
    elapsed = end - start
    return datas, step_num, elapsed