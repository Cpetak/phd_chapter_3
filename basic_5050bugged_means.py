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
    return (1 - torch.abs(pop.squeeze(1) - targ)).sum(axis=1) # the smaller the difference, the higher the fitness

class FakeArgs:
    """
    A simple class imitating the args namespace
    """

    def __repr__(self):
        attrs = vars(self)
        return "\n".join([f"{k}: {v}" for k, v in attrs.items()])

def prepare_run(entity, project, args, folder_name="results"):
    import wandb

    #folder = get_folder()
    #args.location = get_location()

    run = wandb.init(config=args, entity=entity, project=project)

    today = date.today().strftime("%d-%m-%Y")
    folder = Path(folder_name) / run.name
    folder.mkdir(parents=True, exist_ok=True)

    return run, folder

def calc_strategy(phenotypes, envs, args, curr_targ):
    perm = torch.randperm(phenotypes.size(0))
    idx = perm[:int(args.selection_size*args.pop_size)]
    sample = phenotypes[idx]

    spec_A = 0
    spec_B = 0
    gen = 0
    low = 0
    #high = 0

    first_fitness = fitness_function(sample, envs[0]) #env A
    #curr_targ = (curr_targ + 1) % 2
    second_fitness = fitness_function(sample, envs[1]) #env B

    first_fitness = first_fitness / (int(args.num_genes_consider*args.grn_size))
    second_fitness = second_fitness / (int(args.num_genes_consider*args.grn_size))

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
  state = torch.zeros(num_indv, 1, args["grn_size"]).to(device)
  state[:, :, 0] = 1.0 # create input to the GRNs

  state_before = torch.zeros(num_indv, 1, args["grn_size"]).to(device) # keeping track of the last state
  for l in range(args["max_iter"]):
    state = torch.matmul(state, pop) # each matrix in the population is multiplied
    state = state * args["alpha"]
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
    targ = torch.randint(2, size=(1, args.grn_size)).to(device) # create a random target, binary (for better visualisation)
    num_genes_fit=int(args.num_genes_consider*args.grn_size)
    targ = targ[:,:num_genes_fit] # create targets only for relevant genes
    targs = [targ, 1 - targ.detach().clone()] # alternative target is the exact opposite
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
        complexities = torch.zeros(args.pop_size)
        state, complexities=get_phenotypes(args, pop, args["pop_size"], complexities, if_comp= True)
        ave_complex.append(args.max_iter-complexities.mean().item()) # 0 = never converged, the higher the number the earlier it converged so true "complexity" is inverse of this value
        run.log({'average_complexity': args.max_iter-complexities.mean().item()}, commit=False)

        phenos = state[:,:,:num_genes_fit]

        # TRACKING diversity among siblings of the same parent, from the previous generation
        if gen > 0:
          child_phenotypes = phenos[children_locs]
          reshaped=torch.reshape(child_phenotypes, (num_child, len(parent_locs), args["grn_size"]))
          stds=torch.std(reshaped,dim=(0))

          #print(stds.mean(1)) #for each group of siblings, mean standard deviation across the genes
          #print(stds.mean(1).mean()) #mean standard deviation across the sibling groups

          #mylist=[[[0, 1]], [[0, 1]], [[0, 1]], [[1, 0]],[[1, 0]], [[1, 0]]] # 0.7071 is the max std then
          #test=torch.Tensor(mylist).to('cuda')

          run.log({'std_of_children': stds.mean(1).mean().item()}, commit=False)
          kid_stds.append(stds.mean(1).mean().item())
  

        # Evaluate fitnesses
        # ALTERNATIVE FITNESS FUNCTION

        num_clones=20

        c_genotypes = []
        c_phenotypes = []
        c_fits0 = []
        c_fits1 = []
        
        for c in range(num_clones): 
          
          # Make clones
          clones = pop.clone() # create one clone of each individual in the population
          
          # Mutate clones
          num_genes_mutate = int(args["grn_size"]*args["grn_size"]*len(clones) * args["mut_rate"])
          mylist = torch.zeros(args["grn_size"]*args["grn_size"]*len(clones), device="cuda")
          mylist[:num_genes_mutate] = 1
          shuffled_idx = torch.randperm(args["grn_size"]*args["grn_size"]*len(clones), device="cuda")
          mask = mylist[shuffled_idx].reshape(len(clones),args["grn_size"],args["grn_size"]) #select genes to mutate
          clones = clones + (clones*mask)*torch.randn(size=clones.shape, device="cuda") * args["mut_size"]  # mutate only at certain genes

          # Get clone phenotypes
          clone_states = get_phenotypes(args, clones, args["pop_size"], complexities, if_comp= False)
          clone_phenos = clone_states[:,:,:num_genes_fit]

          # Get clone fitnesses
          clone_fitnesses0=fitness_function(clone_phenos, targs[0])
          clone_fitnesses1=fitness_function(clone_phenos, targs[1])
          clone_fitnesses0=clone_fitnesses0/ (int(args["num_genes_consider"]*args["grn_size"])) # rescale to 0-1
          clone_fitnesses1=clone_fitnesses1/ (int(args["num_genes_consider"]*args["grn_size"]))

          # Save results
          c_genotypes.append(clones)
          c_phenotypes.append(clone_phenos)
          c_fits0.append(clone_fitnesses0)
          c_fits1.append(clone_fitnesses1)
                
        c_genotypes = torch.stack(c_genotypes)
        c_phenotypes = torch.stack(c_phenotypes)
        c_fits0 = torch.stack(c_fits0)
        c_fits1 = torch.stack(c_fits1)

        mean0= torch.mean(c_fits0,0) # for each individual, averaged across the clones
        mean1= torch.mean(c_fits1,0)
        diffs= 1-abs(mean0-mean1) #should be 0 if they are the same

        temp_fits = fitness_function(phenos, targs[curr_targ]) 
        temp_fits = temp_fits / (int(args["num_genes_consider"]*args["grn_size"]))
        fitnesses = temp_fits * diffs

        
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

        l,sA,sB,g=calc_strategy(phenos, targs, args, curr_targ)
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
    stats["best_grns"] = best_grns
    stats["diversities"] = diversities
    stats["low"] = low
    stats["spec_A"] = spec_A
    stats["spec_B"] = spec_B
    stats["gen"] = genal
    stats["kid_stds"] = kid_stds

    with open(f"{folder}/{title}.pkl", "wb") as f:
        pickle.dump(stats, f)

if __name__ == "__main__":
    #parser = argparse.ArgumentParser()

    args = FakeArgs()

    #parser.add_argument('-grn_size', type=int, default=50, help="GRN size") # number of genes in the GRN
    args.grn_size = 50
    #parser.add_argument('-pop_size', type=int, default=1000, help="Population size")
    args.pop_size = 1000
    #parser.add_argument('-alpha', type=int, default=10, help="Alpha for sigmoid function")
    args.alpha = 10
    #parser.add_argument('-num_genes_consider', type=float, default=0.5, help="proportion of genes considered for fitness")
    args.num_genes_consider = 1
    #parser.add_argument('-mut_rate', type=float, default=0.1, help="rate of mutation (i.e. number of genes to mutate)")
    args.mut_rate = 0.1
    #parser.add_argument('-mut_size', type=float, default=0.5, help="size of mutation")
    args.mut_size = 0.5
    #parser.add_argument('-num_generations', type=int, default=100000, help="number of generations to run the experiment for") # number of generations
    args.num_generations = 5000
    #parser.add_argument('-truncation_prop', type=float, default=0.2, help="proportion of individuals selected for reproduction")
    args.truncation_prop = 0.2
    #parser.add_argument('-max_age', type=int, default=30, help="max age at which individual is replaced by its kid")
    args.max_age = 1000000000000000
    #parser.add_argument('-season_len', type=int, default=100, help="number of generations between environmental flips")
    args.season_len = 500
    #parser.add_argument('-selection_size', type=float, default=1, help="what proportion of the population to test for strategy (specialist, generatist)")
    args.selection_size = 0.2
    #parser.add_argument('-proj', type=str, default="EC_final_project", help="Name of the project (for wandb)")
    args.proj = "phd_chapt_3"
    #parser.add_argument('-exp_type', type=str, default="BASIC", help="Name your experiment for grouping")
    args.exp_type = "season_length_20"

    #parser.add_argument('-crossover', type=str, default="NO", help="Options: NO, uniform, twopoint")
    args.crossover = "NO"
    #parser.add_argument('-crossover_freq', type=float, default=0.5, help="number of individuals that will undergo crossover")
    args.crossover_freq = 0.5
    #parser.add_argument('-adaptive_mut', type=bool, default=False, help="if you want adaptive mutation rate")
    args.adaptive_mut = False
    #parser.add_argument('-meta_mut_rate', type=float, default=0.01, help="how much you increase or decrease mut_size and mut_rate")
    args.meta_mut_rate = 0.01

    #args = parser.parse_args()

    print("running code")

    args.max_iter = int(3*args.grn_size) # "Maximum number of GRN updates") # number of times gene concentrations are updated to get phenotype

    args.truncation_size=int(args.truncation_prop*args.pop_size)


    print(args)

    args.num_crossover = int(args.crossover_freq * args.pop_size) #how many individuals will be involved in crossover

    #assert (
        #args.num_crossover % 2 == 0
    #), f"Error: select different crossover_freq: received {args.num_crossover}"
    assert (
        args.pop_size % args.truncation_size == 0
    ), "Error: select different trunction_prop, received {args.pops_size}"


    run, folder = prepare_run("molanu", args.proj, args)

    evolutionary_algorithm(args, f"{args.exp_type}", folder)

