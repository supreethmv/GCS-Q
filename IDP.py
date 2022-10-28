import math
import itertools

import min_cut_solvers

def evaluateSplits(coalition, coalition_values, **kwargs):
    #print("coalition",coalition,end='=')
    agents = coalition.split(',')
    n_agents = len(agents)
    best_cost_brute = f[coalition]
    xbest_brute = [coalition]
    for b in range(1, 2**(n_agents-1)):
        x = [int(term) for term in reversed(list(bin(b)[2:].zfill(n_agents)))]
        first_half = ','.join([agent for i,agent in enumerate(agents) if int(x[i])])
        second_half = ','.join([agent for i,agent in enumerate(agents) if not int(x[i])])
        if best_cost_brute <= (f[first_half]+f[second_half]):
            best_cost_brute = f[first_half]+f[second_half]
            xbest_brute = [first_half, second_half]
    #print(xbest_brute, best_cost_brute)
    return xbest_brute, best_cost_brute


def idp(coalition_values, evaluateSplits = evaluateSplits, min_cut_solver = min_cut_solvers.min_cut_brute_force, **kwargs):
    n_agents = math.ceil(math.log(len(coalition_values),2))
    global t
    t = {}
    global f
    f = {}
    for coalition,coalition_value in coalition_values.items():
        t[coalition] = [coalition]
        f[coalition] = coalition_value
    for coalition_size in range(2, n_agents):
        if((math.ceil((2*n_agents)/3)<coalition_size) and (coalition_size < n_agents)):                  # Ignoring this condition will make this function work as DP instead of IDP
            continue
        coalitions_of_cur_size = list(itertools.combinations(map(str,range(1,n_agents+1)), coalition_size))
        for curCoalition in coalitions_of_cur_size:
            curCoalition = ','.join(curCoalition)
            split_t, split_f = evaluateSplits(curCoalition, coalition_values, min_cut_solver = min_cut_solver, **kwargs)
            if split_f > f[curCoalition]:
                t[curCoalition] = split_t
                f[curCoalition] = split_f
    grand_coalition = ','.join(map(str,range(1,n_agents+1)))

    split_t, split_f = evaluateSplits(grand_coalition, coalition_values, min_cut_solver = min_cut_solver, **kwargs)
    if split_f > f[grand_coalition]:
        t[grand_coalition] = split_t
        f[grand_coalition] = split_f
    temp = t[grand_coalition].copy()
    optimal_cs = []
    while(len(temp)):
        C = temp.pop()
        if len(t[C])==1:
            optimal_cs+=t[C]
        if(len(t[C])!=1):
            temp += t[C]
    optimal_cs_value = sum([f[coalition] for coalition in optimal_cs])
    return optimal_cs, optimal_cs_value
