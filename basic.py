import torch
from tqdm import trange, tqdm
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import combinations

import pickle
from copy import deepcopy
import wandb
from datetime import date
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu" #for testing on my computer

#Helper functions

def to_im(ten):
    return ten.cpu().detach().clone().squeeze(1).numpy()


def dround(ten, digits):
  a = 10 ^ digits
  return torch.round(ten * a) / a


def fitness_function(pop, targ):
    return (1 - torch.abs(pop.squeeze(1) - targ)).sum(axis=1) /pop.size()[-1] # the smaller the difference, the higher the fitness

def prepare_run(entity, project, args, folder_name="results"):

    run = wandb.init(config=args, entity=entity, project=project)

    today = date.today().strftime("%d-%m-%Y")
    folder = Path(folder_name) / run.name
    folder.mkdir(parents=True, exist_ok=True)

    return run, folder

def calc_strategy(phenotypes, selection_size, envs, args, curr_targ):
    perm = torch.randperm(phenotypes.size(0))
    idx = perm[:selection_size]
    sample = phenotypes[idx]

    spec_A = 0
    spec_B = 0
    gen = 0
    low = 0
    #high = 0

    first_fitness = fitness_function(sample, envs[0]) #env A
    #curr_targ = (curr_targ + 1) % 2
    second_fitness = 1-first_fitness #fitness_function(sample, envs[1]) #env B

    for i in range(len(first_fitness)):
        if (first_fitness[i] < 0.3 and second_fitness[i] < 0.3) or (0.3 <= first_fitness[i] <= 0.7 and second_fitness[i] < 0.3) or (first_fitness[i] < 0.3 and 0.3 <=second_fitness[i] <= 0.7):
          low += 1
        #if (first_fitness[i] > 0.7 and second_fitness[i] > 0.7) or (0.3 <= first_fitness[i] <= 0.7 and second_fitness[i] > 0.7) or (first_fitness[i] > 0.7 and 0.3 <=second_fitness[i] <= 0.7):
        #high += 1
        if (first_fitness[i] < 0.3 and second_fitness[i] > 0.7):
          spec_B += 1
        if (first_fitness[i] > 0.7 and second_fitness[i] < 0.3):
          spec_A += 1
        if 0.3 <= first_fitness[i] <= 0.7 and 0.3 <= second_fitness[i] <= 0.7:
          gen += 1

    run.log({'low': low}, commit=False)
    #run.log({'high': high}, commit=False) #impossible
    run.log({'spec_A': spec_A}, commit=False)
    run.log({'spec_B': spec_B}, commit=False)
    run.log({'gen': gen}, commit=False)

    #print(list(zip(first_fitnesses,second_fitnesses)))
    #torch.numel(first_fitnesses[first_fitnesses>0.7])

    return(low,spec_A,spec_B,gen)

def get_phenotypes(args, pop, num_indv, complexities, if_comp):
  state = torch.zeros(num_indv, 1, args.grn_size).to(device)
  state[:, :, 0] = 1.0 # create input to the GRNs

  state_before = torch.zeros(num_indv, 1, args.grn_size).to(device) # keeping track of the last state
  for l in range(args.max_iter):
    state = torch.matmul(state, pop) # each matrix in the population is multiplied
    state = state * args.alpha
    state = torch.sigmoid(state) # after which it is put in a sigmoid function to get the output, by default alpha = 1 which is pretty flat, so let's use alpha > 1 (wagner uses infinite) hence the above multiplication
    # state = dround(state, 2)
    diffs=torch.abs(state_before - state).sum(axis=(1,2))
    which_repeat = torch.where(diffs == 0)
    if if_comp:
      complexities[which_repeat] += 1
    state_before = state

  if if_comp:
    return state, complexities
  else:
    return state

# Evolve
def evolutionary_algorithm(args, title, folder):

    #Setting up

    pop = torch.randn((args.pop_size, args.grn_size, args.grn_size)).to(device) # create population of random GRNs
    num_genes_fit=int(args.num_genes_consider*args.grn_size)
    ones=torch.ones(1,int(num_genes_fit/2)).to(device)
    zeros=torch.zeros(1,int(num_genes_fit/2)).to(device)
    targA=torch.cat((ones,zeros),1)
    targB=torch.cat((zeros,ones),1)
    targs = [targA,targB]
    
    curr_targ = 0 # ID of which target is the current one
    previous_targ=curr_targ # will be used for measuring evolvability
    ages = torch.zeros(args.pop_size).to(device)
    time_since_change = 0 # num gens since last environmental switch
    epoc = 0

    # Keeping track

    max_fits = []
    ave_fits = []
    st_div_fits = []
    ave_complex = []
    max_ages = []
    ave_ages = []
    diversities = []
    low = []
    spec_A = []
    spec_B = []
    genal = []
    kid_stds = []

    delta_fit_envchange_max = []
    delta_fit_envchange = []
    rebound_time = torch.zeros(int(args.num_generations/args.season_len)).to(device)
    rebound_time_max = torch.zeros(int(args.num_generations/args.season_len)).to(device)

    best_grns = []

    # Main for loop
    for gen in trange(args.num_generations):

        time_since_change += 1

        # Generating phenotypes
        complexities = torch.zeros(args.pop_size).to(device)
        state, complexities=get_phenotypes(args, pop, args.pop_size, complexities, if_comp= True)
        ave_complex.append(args.max_iter-complexities.mean().item()) # 0 = never converged, the higher the number the earlier it converged so true "complexity" is inverse of this value
        run.log({'average_complexity': args.max_iter-complexities.mean().item()}, commit=False)

        phenos = state[:,:,:num_genes_fit]

        # TRACKING diversity among siblings of the same parent, from the previous generation
        if gen > 0:
          child_phenotypes = phenos[children_locs]
          reshaped=torch.reshape(child_phenotypes, (num_child, len(parent_locs), args.grn_size))
          stds=torch.std(reshaped,dim=(0))
          run.log({'std_of_children': stds.mean(1).mean().item()}, commit=False)
          kid_stds.append(stds.mean(1).mean().item())

        # Evaluate fitnesses 
        fitnesses = fitness_function(phenos, targs[curr_targ])

        # TRACKING reduction in fitness right after env change
        if previous_targ!=curr_targ:
          delta_fit_envchange.append((fitnesses.mean().item())/(ave_fits[-1]))
          run.log({'delta_fit_envchange': (fitnesses.mean().item())/(ave_fits[-1])}, commit=False)
          delta_fit_envchange_max.append((fitnesses.max().item())/(max_fits[-1]))
          run.log({'delta_fit_envchange_max': (fitnesses.max().item())/(max_fits[-1])}, commit=False)

        # TRACKING fitness redound time after env change
        if epoc != 0: # if it is not the first epoc
          if rebound_time[epoc] == 0: # if for this epoc, fitness rebound hasn't happened yet
            if ave_fits[(-1*time_since_change)] <= fitnesses.mean().item(): # if fitness rebounded this generation
              rebound_time[epoc] = time_since_change
              # if reound_time is 0 somewhere, that means that previous good fitness wasn't found in this epoc.
              # if there is a 0 right after, that means that the new "max" (final fitness before env change) was also not recovered.
              # note the shift in frame of reference

          if rebound_time_max[epoc] == 0: # if for this epoc, fitness rebound hasn't happened yet
            if max_fits[(-1*time_since_change)] <= fitnesses.max().item(): # if fitness rebounded this generation
              rebound_time_max[epoc] = time_since_change
              # if reound_time is 0 somewhere, that means that previous good fitness wasn't found in this epoc.
              # if there is a 0 right after, that means that the new "max" (final fitness before env change) was also not recovered.
              # note the shift in frame of reference
          if time_since_change == args.season_len: # if this is the end of the epoc (i.e. last gen in this season)
            if rebound_time[epoc] == 0: # if it didn't rebound in the epoc, change 0 to something big = never found previous fitness
              rebound_time[epoc] = args.season_len*2
            if rebound_time_max[epoc] == 0: 
              rebound_time_max[epoc] = args.season_len*2

            run.log({'rebound_time': rebound_time[epoc]}, commit=False)
            run.log({'rebound_time_max': rebound_time_max[epoc]}, commit=False)

        max_fits.append(fitnesses.max().item()) # keeping track of max fitness
        ave_fits.append(fitnesses.mean().item()) # keeping track of average fitness
        st_div_fits.append(fitnesses.std().item())
        run.log({'max_fits': fitnesses.max().item()}, commit=False)
        run.log({'ave_fits': fitnesses.mean().item()}, commit=False)
        run.log({'st_div_fits': fitnesses.std().item()}, commit=False)

        selection_size=int(args.pop_size*args.selection_prop)
        l,sA,sB,g=calc_strategy(phenos, selection_size, targs, args, curr_targ)
        low.append(l)
        spec_A.append(sA)
        spec_B.append(sB)
        genal.append(g)

        # SELECTION
        perm = torch.argsort(fitnesses, descending=True)
        parent_locs = perm[:args.truncation_size] # location of top x parents in the array of individuals
        children_locs = perm[args.truncation_size:] # location of individuals that won't survive and hence will be replaced by others' children

        #champions.append(state[perm[0]].detach().clone().cpu().squeeze(0).numpy()) # keeping tract of best solution's output
        best_grns.append(pop[perm[0]].detach().clone().cpu()) # keeping tract of best solution
        #run.log({'champions': state[perm[0]].detach().clone().cpu().squeeze(0).numpy()}, commit=False)
        #run.log({'best_grns': pop[perm[0]].detach().clone().cpu()}, commit=False)

        ages[parent_locs] += 1 # updating the ages of the individuals
        ages[children_locs] = 0

        parents = pop[parent_locs].detach().clone() # access parents' matricies
        num_child = int(args.pop_size/args.truncation_size) - 1
        children = parents.repeat([num_child, 1, 1]) # create copies of parents

        # MUTATION
        num_genes_mutate = int(args.grn_size*args.grn_size*len(children) * args.mut_rate)
        mylist = torch.zeros(args.grn_size*args.grn_size*len(children), device=device)
        mylist[:num_genes_mutate] = 1
        shuffled_idx = torch.randperm(args.grn_size*args.grn_size*len(children), device=device)
        mask = mylist[shuffled_idx].reshape(len(children),args.grn_size,args.grn_size) #select genes to mutate
        children = children + (children*mask)*torch.randn(size=children.shape, device=device) * args.mut_size  # mutate only children only at certain genes

        pop[children_locs] = children # put children into population

        max_ages.append(ages.max().item())
        ave_ages.append(ages.mean().item())
        run.log({'max_ages': ages.max().item()}, commit=False)
        run.log({'ave_ages': ages.mean().item()}, commit=False)

        d=torch.mean(torch.std(pop,unbiased=False, dim=0))
        diversities.append(d)
        run.log({'diversities': d}, commit=True)

        # CHANGE ENVIRONMENT
        previous_targ=curr_targ
        if gen % args.season_len == args.season_len - 1: # flip target
            curr_targ = (curr_targ + 1) % 2
            #print(time_since_change)
            time_since_change = 0
            epoc += 1

    # SAVE DATA
    stats = {}
    stats["max_fits"] = max_fits
    stats["ave_fits"] = ave_fits
    stats["st_div_fits"] = st_div_fits
    stats["ave_complex"] = ave_complex
    stats["delta_fit_envchange"] = delta_fit_envchange
    stats["delta_fit_envchange_max"] = delta_fit_envchange_max
    stats["rebound_time"] = rebound_time
    stats["rebound_time_max"] = rebound_time_max
    stats["max_ages"] = max_ages
    stats["ave_ages"] = ave_ages
    stats["best_grns"] = [grn.detach().cpu().numpy() for grn in best_grns]
    stats["diversities"] = diversities
    stats["low"] = low
    stats["spec_A"] = spec_A
    stats["spec_B"] = spec_B
    stats["gen"] = genal
    stats["kid_stds"] = kid_stds
    stats["args_used"] = args

    with open(f"{folder}/{title}.pkl", "wb") as f:
        pickle.dump(stats, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()


    # DONT CHANGE FOR FIRST ROUND

    parser.add_argument('-grn_size', type=int, default=50, help="GRN size") # number of genes in the GRN
    parser.add_argument('-pop_size', type=int, default=1000, help="Population size")
    parser.add_argument('-alpha', type=int, default=10, help="Alpha for sigmoid function")
    parser.add_argument('-num_genes_consider', type=float, default=1, help="proportion of genes considered for fitness")
    parser.add_argument('-max_age', type=int, default=1000000000000000, help="max age at which individual is replaced by its kid")
    parser.add_argument('-proj', type=str, default="phd_chapt_3", help="Name of the project (for wandb)")
    parser.add_argument('-crossover', type=str, default="NO", help="Options: NO, uniform, twopoint")
    parser.add_argument('-crossover_freq', type=float, default=0.5, help="number of individuals that will undergo crossover")
    parser.add_argument('-adaptive_mut', type=bool, default=False, help="if you want adaptive mutation rate")
    parser.add_argument('-meta_mut_rate', type=float, default=0.01, help="how much you increase or decrease mut_size and mut_rate")
    parser.add_argument('-selection_prop', type=float, default=0.1, help="what proportion of the population to test for strategy (specialist, generatist)")
    parser.add_argument('-exp_type', type=str, default="BASIC", help="Name your experiment for grouping")
    
    # DO CHANGE
    parser.add_argument('-mut_rate', type=float, default=0.1, help="rate of mutation (i.e. number of genes to mutate)")
    
    parser.add_argument('-mut_size', type=float, default=0.5, help="size of mutation")
   
    parser.add_argument('-num_generations', type=int, default=1000, help="number of generations to run the experiment for") # number of generations
    
    parser.add_argument('-truncation_prop', type=float, default=0.2, help="proportion of individuals selected for reproduction")
    
    parser.add_argument('-season_len', type=int, default=100, help="number of generations between environmental flips")
    

    args = parser.parse_args()

    print("running code")

    args.max_iter = 100 #int(3*args.grn_size) # "Maximum number of GRN updates") # number of times gene concentrations are updated to get phenotype

    args.truncation_size=int(args.truncation_prop*args.pop_size)

    print(args)


    assert (
        args.pop_size % args.truncation_size == 0
    ), "Error: select different trunction_prop, received {args.pops_size}"
    assert ( int(args.num_genes_consider*args.grn_size) % 2 == 0), "Error: select different num_genes_consider, needs to be a multiple of 2"

    run, folder = prepare_run("molanu", args.proj, args)

    evolutionary_algorithm(args, f"{args.exp_type}", folder)

