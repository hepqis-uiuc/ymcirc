"""Classes for juggling registers in quantum simulations of lattices."""
from __future__ import annotations
import copy
from math import isclose, ceil
from dataclasses import dataclass
import numpy as np
import json

# WIP. Can make an error class to add error bars to graphs

def calculate_statistical_error(mean, n_shots):
	sd = np.sqrt((mean*((1 - mean)**2)) + ((1-mean)*(mean**2)))
	return sd/np.sqrt(n_shots)

def calculate_trotter_error(n_trotter):
	trotter_error = ((t**4)/(64.0*(n_trotter**2)))
	trotter_error_meas = 2*np.sqrt(value)*trotter_error

	return trotter_error


