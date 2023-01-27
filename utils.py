import numpy as np
import itertools
import os
import json

import dimod
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite

import numpy as np

#Different distributions data generator functions

def normal(size=1, mu=0, sigma=50, low=-100, high=100):
  values = np.random.normal(mu, sigma, size)
  values = np.interp(values, (values.min(), values.max()), (low, high))
  return values

def uniform(size=1, low=-100, high=100):
  values = np.random.normal(low, high, size)
  values = np.interp(values, (values.min(), values.max()), (low, high))
  return values

def laplace(size=1, loc=0, scale=50, low=-100, high=100):
  values = np.random.laplace(loc, scale, size)
  values = np.interp(values, (values.min(), values.max()), (low, high))
  return values

def beta(size=1, a_beta=1 , b_beta=1, low=-100, high=100):
  values = np.random.beta(a_beta, b_beta, size)
  values = np.interp(values, (values.min(), values.max()), (low, high))
  return values

def binomial(size=1, n_trails=1000000, p_trails=0.5, low=-100, high=100):
  values = np.random.binomial(n_trails, p_trails, size)
  values = np.interp(values, (values.min(), values.max()), (low, high))
  return values

def chisquare(size=1, degree_of_freedom=2, low=-100, high=100):
  values = np.random.chisquare(degree_of_freedom,size)
  values = np.interp(values, (values.min(), values.max()), (low, high))
  return values

def exponential(size=1, scale=1.0, low=-100, high=100):
  values = np.random.exponential(scale, size)
  values = np.interp(values, (values.min(), values.max()), (low, high))
  return values

def gamma(size=1, shape=1.0, scale=1.0, low=-100, high=100):
  values = np.random.gamma(shape, scale, size)
  values = np.interp(values, (values.min(), values.max()), (low, high))
  return values

def geometric(size=1, p=0.001, low=-100, high=100):
  values = np.random.geometric(p, size)
  values = np.interp(values, (values.min(), values.max()), (low, high))
  return values

def rayleigh(size=1, scale=1.0, low=-100, high=100):
  values = np.random.rayleigh(scale, size)
  values = np.interp(values, (values.min(), values.max()), (low, high))
  return values

def random(size=1, low=-100, high=100):
  values = np.random.random_sample(size)
  values = np.interp(values, (values.min(), values.max()), (low, high))
  return values

def weibull(size=1, a=1.0, low=-100, high=100):
  values = np.random.weibull(a, size)
  values = np.interp(values, (values.min(), values.max()), (low, high))
  return values

def f_distribution(size=1, dfnum=1, dfden=100, low=-100, high=100):
  values = np.random.f(dfnum, dfden, size)
  values = np.interp(values, (values.min(), values.max()), (low, high))
  return values


  # Generate induced subgraph game
def generate_induced_subgraph_game(distribution, n_agents, **kwargs):
  induced_subgraph_game = {}
  keys = list(itertools.combinations(range(1,n_agents+1), 2))
  totalinteractions = len(keys)
  values = distribution(totalinteractions, **kwargs)
  for i,key in enumerate(keys):
    induced_subgraph_game[','.join(map(str,key))] = round(values[i],2)
  return induced_subgraph_game


  # Convert induced subgraph game to a generic coalitional game

def induced_subgraph_game_to_coalition_game(n_agents, induced_subgraph_game):
  coalition_game = {}
  totalcoalitions = 2 ** n_agents
  agents = list(map(str,(range(1, n_agents + 1))))
  for agent in agents:
    coalition_game[agent] = 0
  for key,value in induced_subgraph_game.items():
    coalition_game[key] = value
  for coalition_size in range(3,n_agents+1):
    coalitions = list(itertools.combinations(agents, coalition_size))
    for coalition in coalitions:
      value = []
      for key in induced_subgraph_game:
        if key.split(",")[0] in coalition and key.split(",")[1] in coalition:
          value.append(induced_subgraph_game[key])
      coalition_game[','.join(coalition)] = sum(value)
  return coalition_game


  # Convert generic coalitional game to induced subgraph game

def coalition_game_to_induced_subgraph_game(n_agents, coalition_game):
  induced_subgraph_game = {}
  for coalition in coalition_game:
    if len(coalition.split(","))==2:
      induced_subgraph_game[coalition] = coalition_game[coalition]
  return induced_subgraph_game


  #For converting the coalition game dictionary to the benchmark dataset schema

def coalition_game_to_dataset_format(n_agents, coalition_game):
  c_values = []
  c_values.append(0)
  for i in range(1,len(coalition_game)+1):
    coalition_binary_str = bin(i)[2:].zfill(n_agents)[::-1]
    agents = sorted([idx+1 for idx, b in enumerate(coalition_binary_str) if int(b)])
    agents = ','.join(map(str,agents))
    c_values.append(coalition_game[agents])
  return c_values


  # For converting the benchmark dataset schema to coalition game dictionary

def dataset_format_to_coalition_game(n_agents, c_values):
  coalition_game = {}
  for index,c_value in enumerate(c_values[1:]):
    index_binary = bin(index+1)[2:].zfill(n_agents)[::-1]
    key = []
    for agent,digit in enumerate(index_binary):
      if int(digit):
        key.append(agent+1)
    key = ','.join(map(str,sorted(key)))
    coalition_game[key] = c_value
  return coalition_game


def save_json(filename, dictionary):
    with open(filename + '.json', 'w') as fp:
        json.dump(dictionary, fp)


def create_dir(path, log=False):
    if not os.path.exists(path):
        if log:
            print('The directory', path, 'does not exist and will be created')
        os.makedirs(path)
    else:
        if log:
            print('The directory', path, ' already exists')


            
def dwave_solver(linear, quadratic, offset = 0.0, runs=10000):
    """
    Solve Ising hamiltonian or qubo problem instance using dimod API for using dwave system.
    :params
    linear: dictionary of linear coefficient terms in the QUBO formulation of the CSG problem.
    quadratic: dictionary of quadratic coefficient terms in the QUBO formulation of the CSG problem.
    runs: Number of repeated executions
    :return
    sample_set: Samples and any other data returned by dimod samplers.
    """
    # DWaveSampler()
    vartype = dimod.BINARY

    bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, vartype)
    sampler = EmbeddingComposite(DWaveSampler(solver={'topology__type': 'chimera'}))
    sample_set = sampler.sample(bqm, num_reads=runs)
    return sample_set

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


