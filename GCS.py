import itertools
import min_cut_solvers


def get_coalition_value(coalition, induced_subgraph_game):
    agents = coalition.split(',')
    return sum([induced_subgraph_game[','.join(map(str,sorted(map(int,key))))] for key in itertools.combinations(agents, 2)])


def evaluateSplits_min_cut(coalition, induced_subgraph_game, min_cut_solver = min_cut_solvers.min_cut_brute_force, **kwargs):
  #print("coalition",coalition,end='=')
    agents = coalition.split(',')
    n = len(agents)
    if n==1:
        return [coalition], 0
    if n==2:
        c_value = induced_subgraph_game[coalition]
        if c_value<=0:
            #print([agents[0],agents[1]], 0)
            return [agents[0],agents[1]], 0
        else:
            #print([coalition], c_value)
            return [coalition], c_value
    min_cut_mapping = {}
    for idx,agent in enumerate(agents):
        min_cut_mapping[agent] = str(idx+1)
    subproblem_as_induced_subgraph_game = {','.join([min_cut_mapping[vertex] for vertex in map(str,sorted(map(int,key)))]):induced_subgraph_game[','.join(map(str,sorted(map(int,key))))] for key in itertools.combinations(agents, 2)}
    xbest_brute, best_cost_brute = min_cut_solver(n,subproblem_as_induced_subgraph_game, **kwargs)
    if 0 in xbest_brute and 1 in xbest_brute:
        first_half = ','.join([agent for idx,agent in enumerate(agents) if xbest_brute[idx]])
        second_half = ','.join([agent for idx,agent in enumerate(agents) if not xbest_brute[idx]])
        bruteforce_solution_decoded = [first_half, second_half]
        best_cost_brute = get_coalition_value(first_half, induced_subgraph_game) + get_coalition_value(second_half, induced_subgraph_game)
    else:
        bruteforce_solution_decoded = [coalition]
        best_cost_brute = get_coalition_value(coalition, induced_subgraph_game)
    #print(bruteforce_solution_decoded, best_cost_brute)
    return bruteforce_solution_decoded, best_cost_brute


def gcs(induced_subgraph_game, min_cut_solver = min_cut_solvers.min_cut_brute_force, **kwargs):
    grand_coalition = ','.join(map(str,sorted(map(int,(set([key.split(',')[i] for i in range(2) for key in induced_subgraph_game]))))))
    temp = [grand_coalition]
    optimal_cs = []
    while(len(temp)):
        c = temp.pop()
        c_split_t,c_split_f = evaluateSplits_min_cut(c, induced_subgraph_game, min_cut_solver = min_cut_solver, **kwargs)
        if len(c_split_t)==1:
            optimal_cs+=c_split_t
        if len(c_split_t)>1:
            temp += c_split_t
    return optimal_cs, sum([get_coalition_value(c, induced_subgraph_game) for c in optimal_cs])