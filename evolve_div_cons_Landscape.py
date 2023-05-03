#Started by duplicating basic.py on the 2nd of May, 2023
#Get GRN of best diversifier, conservative, specialist for A and B

import torch
from tqdm import trange
import argparse
import pickle
import random
import string
import numpy as np


device = "cuda" if torch.cuda.is_available() else "cpu" #for testing on my computer

#Helper functions

def to_im(ten):
    return ten.cpu().detach().clone().squeeze(1).numpy()


def dround(ten, digits):
  a = 10 ^ digits
  return torch.round(ten * a) / a


def fitness_function(pop, targ):
    return (1 - torch.abs(pop.squeeze(1) - targ)).sum(axis=1) /pop.size()[-1] # the smaller the difference, the higher the fitness

def get_phenotypes(args, pop, num_indv):
  state = torch.zeros(num_indv, 1, args.grn_size).to(device)
  state[:, :, 0] = 1.0 # create input to the GRNs

  for l in range(args.max_iter):
    state = torch.matmul(state, pop) # each matrix in the population is multiplied
    state = state * args.alpha
    state = torch.sigmoid(state) # after which it is put in a sigmoid function to get the output, by default alpha = 1 which is pretty flat, so let's use alpha > 1 (wagner uses infinite) hence the above multiplication
    
  return state

# Conservative: average distance of gene expression from 0.5
def con(my_phenos):
    #phenos=get_phenotypes(pop,len(pop))
    my_phenos=torch.squeeze(my_phenos)
    con=1-(abs(my_phenos-0.5).mean(1)) #value between 0.5 and 1, higher more conservative
    con=(con-0.5) *2 #rescaled to value between 0 and 1
    return con

#standard deviation between children fitnesses
def diver(my_pop, targA):
    num_clones=20
    clones = my_pop.repeat([num_clones, 1, 1])

    # Mutate clones
    num_genes_mutate = int(args.grn_size*args.grn_size*len(clones) * args.mut_rate)
    mylist = torch.zeros(args.grn_size*args.grn_size*len(clones), device=device)
    mylist[:num_genes_mutate] = 1
    shuffled_idx = torch.randperm(args.grn_size*args.grn_size*len(clones), device=device)
    mask = mylist[shuffled_idx].reshape(len(clones),args.grn_size,args.grn_size) #select genes to mutate
    clones = clones + (clones*mask)*torch.randn(size=clones.shape, device=device) * args.mut_size  # mutate only clones only at certain genes

    # Get clone phenotypes
    clone_phenos=get_phenotypes(clones, len(clones))
    clone_phenos=torch.squeeze(clone_phenos)
    
    reshaped=torch.reshape(clone_phenos, (num_clones, len(my_pop), args.grn_size))
    kid_fits=abs(reshaped-targA).sum(axis=2).T /args.grn_size #each row is 20 kids of the same parent (their fitness)
    fit_stds=torch.std(kid_fits,dim=(1))

    return fit_stds


# Evolve
def evolutionary_algorithm(args, title):

    #Setting up
    pop = torch.randn((args.pop_size, args.grn_size, args.grn_size)).to(device) # create population of random GRNs
    num_genes_fit=int(args.num_genes_consider*args.grn_size)
    ones=torch.ones(1,int(num_genes_fit/2)).to(device)
    zeros=torch.zeros(1,int(num_genes_fit/2)).to(device)
    targA=torch.cat((ones,zeros),1)
    targB=torch.cat((zeros,ones),1)
    targs = [targA,targB]
    curr_targ = 0 # ID of which target is the current one

    # Keeping track
    #Get GRN of best diversifier, conservative, specialist for A and B

    best_div=0
    best_con=0
    bestA=0
    bestB=0

    best_div_score=0
    best_con_score=0
    bestA_score=0
    bestB_score=0

    best_div_gen=0
    best_con_gen=0

    # Main for loop
    for gen in trange(args.num_generations):

        # Generating phenotypes
        state =get_phenotypes(args, pop, args.pop_size)
        phenos = state[:,:,:num_genes_fit]

        # Get bet-hedgingness
        cons=con(phenos)
        most_con=pop[np.argsort(cons.cpu().detach().numpy())[-1]]
        most_con_score=cons[np.argsort(cons.cpu().detach().numpy())[-1]]
        if most_con_score > best_con_score:
            best_con_score = most_con_score
            best_con = most_con.detach().clone().cpu()
            best_con_gen = gen

        divs=diver(pop, targA)
        most_div=pop[np.argsort(divs.cpu().detach().numpy())[-1]]
        most_div_score=divs[np.argsort(divs.cpu().detach().numpy())[-1]]
        if most_div_score > best_div_score:
            best_div_score = most_div_score
            best_div = most_div.detach().clone().cpu()
            best_div_gen=gen

        # Evaluate fitnesses 
        fitnesses = fitness_function(phenos, targs[curr_targ])
        perm = torch.argsort(fitnesses, descending=True)
        # keeping track of best solutions
        if curr_targ == 0:
            if fitnesses.max().item() > bestA_score:
                bestA_score=fitnesses.max().item()
                bestA=pop[perm[0]].detach().clone().cpu()
        else:
            if fitnesses.max().item() > bestB_score:
                bestB_score=fitnesses.max().item()
                bestB=pop[perm[0]].detach().clone().cpu()

        # SELECTION
        parent_locs = perm[:args.truncation_size] # location of top x parents in the array of individuals
        children_locs = perm[args.truncation_size:] # location of individuals that won't survive and hence will be replaced by others' children

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

        # CHANGE ENVIRONMENT
        if gen % args.season_len == args.season_len - 1: # flip target
            curr_targ = (curr_targ + 1) % 2


    # SAVE DATA
    stats = {}
    stats["best_div"] =[grn.detach().cpu().numpy() for grn in best_div]
    stats["best_con"] =[grn.detach().cpu().numpy() for grn in best_con]
    stats["bestA"] =[grn.detach().cpu().numpy() for grn in bestA]
    stats["bestB"] =[grn.detach().cpu().numpy() for grn in bestB]

    stats["best_div_score"] =best_div_score
    stats["best_con_score"] =best_con_score
    stats["bestA_score"] =bestA_score
    stats["bestB_score"] =bestB_score

    stats["best_div_gen"] =best_div_gen
    stats["best_con_gen"] =best_con_gen

    
    stats["args_used"] = args

    with open(f"landscape_results/{title}.pkl", "wb") as f:
        pickle.dump(stats, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()


    # DONT CHANGE FOR FIRST ROUND

    parser.add_argument('-grn_size', type=int, default=50, help="GRN size") # number of genes in the GRN
    parser.add_argument('-pop_size', type=int, default=1000, help="Population size")
    parser.add_argument('-alpha', type=int, default=10, help="Alpha for sigmoid function")
    parser.add_argument('-num_genes_consider', type=float, default=1, help="proportion of genes considered for fitness")
    parser.add_argument('-mut_size', type=float, default=0.5, help="size of mutation")
    
    # DO CHANGE
    parser.add_argument('-mut_rate', type=float, default=0.1, help="rate of mutation (i.e. number of genes to mutate)")
    
    parser.add_argument('-truncation_prop', type=float, default=0.05, help="proportion of individuals selected for reproduction")
    
    parser.add_argument('-season_len', type=int, default=300, help="number of generations between environmental flips")
    
    parser.add_argument('-num_generations', type=int, default=10000, help="number of generations to run the experiment for") # number of generations

    args = parser.parse_args()

    print("running code")

    args.max_iter = 100 #int(3*args.grn_size) # "Maximum number of GRN updates") # number of times gene concentrations are updated to get phenotype

    args.truncation_size=int(args.truncation_prop*args.pop_size)

    print(args)


    assert (
        args.pop_size % args.truncation_size == 0
    ), "Error: select different trunction_prop, received {args.pops_size}"
    assert ( int(args.num_genes_consider*args.grn_size) % 2 == 0), "Error: select different num_genes_consider, needs to be a multiple of 2"

    id=''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))

    evolutionary_algorithm(args, f"{id}")

