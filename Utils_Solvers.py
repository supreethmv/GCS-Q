#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings('ignore')

# Qiskit 
from qiskit.algorithms import NumPyMinimumEigensolver
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization import QuadraticProgram

# Dwaves
import dimod
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite

import numpy as np
import pandas as pd
from sympy import *
import re
import os


from qiskit.algorithms.optimizers import COBYLA

################### ########
#Different distributions data generator functions


def create_dir(path, log=False):
    if not os.path.exists(path):
        if log:
            print('The directory', path, 'does not exist and will be created')
        os.makedirs(path)
    else:
        if log:
            print('The directory', path, ' already exists')




#################################### SOLVER

def numpy_for_qubo(qubo, p=None):                      # Classical solver for QUBO
  """
  numpy_for_qubo solves the given QUBO using Numpy library functions
  :param
  qubo: CSG problem instance reduced to the form of qubo.

  return:
  result: An array of binary digits which denotes the solutionn of the input qubo problem
  """
  exact_mes = NumPyMinimumEigensolver()
  exact = MinimumEigenOptimizer(exact_mes)  # using the exact classical numpy minimum eigen solver
  result = exact.solve(qubo)
  return result




def solve_QUBO(linear, quadratic, algo, p=1):
  """
  solve_QUBO is ahigher order function to solve QUBO using the given algo parameter function
  :param
  linear: dictionary of linear coefficient terms in the QUBO formulation of the CSG problem.
  quadratic: dictionary of quadratic coefficient terms in the QUBO formulation of the CSG problem.
  algo: a callback function for qaoa_for_qubo or numpy_for_qubo

  return:
  result: An array of binary digits which denotes the solutionn of the input qubo problem


  """
  keys = list(linear.keys())
  keys.sort(key=natural_keys)
  qubo = QuadraticProgram()
  for key in keys:
    qubo.binary_var(key)                                                             # initialize the binary variables
  qubo.minimize(linear=linear, quadratic=quadratic)                                  # initialize the QUBO maximization problem instance
  op, offset = qubo.to_ising()
  qp=QuadraticProgram()
  qp.from_ising(op, offset, linear=True)
  result = algo(qubo, p)
  return result

###############################################

def natural_keys(text):
  """
  alist.sort(key=natural_keys) sorts in human order
  http://nedbatchelder.com/blog/200712/human_sorting.ht
  For example: Built-in function ['x_8','x_10','x_1'].sort() will sort as ['x_1', 'x_10', 'x_8']
  But using natural_keys as callback function for sort() will sort as ['x_1','x_8','x_10']
  param:
  text: a list of strings ending with numerical characters

  return:
  sorted list in a human way
  """
  return [ atoi(c) for c in re.split(r'(\d+)', text) ]


def atoi(text):
  """
  Function returns the corresponding value of a numerical string as integer datatype
  param:
  text: string conntaining only numerical charcaters

  return:
  integer value corresponding to the input text
  """
  return int(text) if text.isdigit() else text








def exact_solver(linear, quadratic, offset = 0.0):
    """
    Solve Ising hamiltonian or qubo problem instance using dimod API.
    dimod is a shared API for samplers.It provides:
    - classes for quadratic models—such as the binary quadratic model (BQM) class that contains Ising and QUBO models used by samplers such as the D-Wave system—and higher-order (non-quadratic) models.
    - reference examples of samplers and composed samplers.
    - abstract base classes for constructing new samplers and composed samplers.

    :params
    linear: dictionary of linear coefficient terms in the QUBO formulation of the CSG problem.
    quadratic: dictionary of quadratic coefficient terms in the QUBO formulation of the CSG problem.
    offset: Constant energy offset associated with the Binary Quadratic Model.

    :return
    sample_set: Samples and any other data returned by dimod samplers.
    """
    vartype = dimod.BINARY

    bqm = dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)
    sampler = dimod.ExactSolver()
    sample_set = sampler.sample(bqm)
    return sample_set



def dwave_solver(linear, quadratic, offset = 0.0, runs=1000):
    """
    Solve Ising hamiltonian or qubo problem instance using dimod API for using dwave system.

    :params
    linear: dictionary of linear coefficient terms in the QUBO formulation of the CSG problem.
    quadratic: dictionary of quadratic coefficient terms in the QUBO formulation of the CSG problem.
    runs: Number of repeated executions

    :return
    sample_set: Samples and any other data returned by dimod samplers.
    """
    
    vartype = dimod.BINARY

    bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, vartype)
    sampler = EmbeddingComposite(DWaveSampler())
    sample_set = sampler.sample(bqm, num_reads=runs)
    return sample_set



def extract_best_result(df):
    """
    A function to fetch the binary string with least energy of the input hamiltonian operator

    :params
    df: 

    :return
    x: an array of binary digits specifies the solution of the input problem instance
    fval: value of the operator corresponding to the binary digits in x
    """
    
    row_min = df[df.energy == df.energy.min()]

    cols = []
    for col in df.columns:
        if 'x_' in col:
            cols.append(col)

    x = []

    for col in cols:
        x.append(row_min.iloc[0][col])

    fval = row_min.energy.iloc[0]
    return x, fval

def from_bin_to_var(x, dictionary):
    """
    function to convert binary string to coalition structure

    :params
    x: an array of binary digits (specifies the solution of the input problem instance)
    dictionary: dictionary with coalitions as keys and coalition values as values (CSG problem instance)
    
    :return
    solution: list of lists. coalition structure.
    """
    solution = []
    for i in range(len(x)):
        if x[i] == 1:
            print(list(dictionary.keys())[i])
            solution.append(list(dictionary.keys())[i])
    return solution


def create_QUBO(linear_dict, quadratic_dict):    
    """
    create a QUBO problem instance using the linear and quadratic coefficients.

    :params
    linear: dictionary of linear coefficient terms in the QUBO formulation of the CSG problem.
    quadratic: dictionary of quadratic coefficient terms in the QUBO formulation of the CSG problem.

    :return
    Object of QuadraticProgram class corresponding to the input linear and quadratic coefficients.
    """
    qubo = QuadraticProgram()
    
    keys = list(linear_dict.keys())
    keys.sort(key=natural_keys)
    
    for key in keys:
        qubo.binary_var(key) 


    qubo.minimize(linear=linear_dict, quadratic=quadratic_dict)
    return qubo


def from_columns_to_string(df):

    cols = []
    for col in df.columns:
        if 'x_' in col:
            cols.append(col)

    df['x'] = 'x'
    for index, row in df.iterrows():
        x = ''
        for col in cols:
            x = x + str(row[col])
        df.loc[index, 'x'] = x
    return df[['x', 'num_occurrences', 'energy']]


def get_ordered_solution(dictionary):
    """
    Reordering the (key,value) pairs in the dictionary to fetch only the values in order.
    param:
    dictionary: input dictionary.

    return:
    solution: list of values after reordering the dictionary elements.
    """
    sortedDict = dict(sorted(dictionary.items(), key=lambda x: x[0].lower()))
    solution = []
    for k, v in sortedDict.items():
        solution.append(v)

    return solution


def results_from_dwave(sample_set, exact=False):
    """
    Fetch the details of the output_ from D-Wave system (Quantum Annealing).

    :params
    sample_set: Samples and any other data returned by dimod samplers.

    :return
    solution: a list of binary values corresponding to the solution provided by the output_ of D-Wave device.
    fval: The function value (operator value) of the input hamiltonian corresponding to the solution.
    prob: Probability of the solution.
    rank: rank of the solution out of all possible binary arrays.
    time: time taken by the D-Wave device to compute the solution.
    """
    df = sample_set.to_pandas_dataframe()
    row_min = df[df.energy == df.energy.min()]

    cols = []
    for col in df.columns:
        if 'x_' in col:
            cols.append(col)

    cols.sort(key=natural_keys)
    
    solution = []

    for col in cols:
        solution.append(row_min.iloc[0][col])
    
    
    fval = row_min.energy.iloc[0]

    if not exact:
        occ_min_fval = row_min.num_occurrences.to_list()[0]
        occurences = df.num_occurrences.to_list()

        occurences = sorted(occurences, reverse=True)
        
        time = sample_set.info['timing']['qpu_sampling_time']/1000

        rank = occurences.index(occ_min_fval)+1
        prob = occ_min_fval / sum(df.num_occurrences)
    else:
        rank = 1
        prob = 1
        time = 1

    return solution, fval, prob, rank, time