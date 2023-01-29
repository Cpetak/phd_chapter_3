import torch
from tqdm import trange

device = "cuda" if torch.cuda.is_available() else "cpu" #for testing on my computer

#Helper functions

def to_im(ten):
    return ten.cpu().detach().clone().squeeze(1).numpy()


def dround(ten, digits):
  a = 10 ^ digits
  return torch.round(ten * a) / a


def fitness_function(pop, targ):
    return (1 - torch.abs(pop.squeeze(1) - targ)).sum(axis=1) /pop.size()[-1] # the smaller the difference, the higher the fitness

class FakeArgs:
    """
    A simple class imitating the args namespace
    """

    def __repr__(self):
        attrs = vars(self)
        return "\n".join([f"{k}: {v}" for k, v in attrs.items()])

def get_phenotypes(args, pop, num_indv):
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
    state_before = state

  return state

# Evolve
def evolutionary_algorithm(args):

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

    kid_stds = 0

    # Main for loop
    for gen in trange(args.num_generations):

        # Generating phenotypes
        state=get_phenotypes(args, pop, args.pop_size)
        phenos = state[:,:,:num_genes_fit]

        # TRACKING diversity among siblings of the same parent, from the previous generation
        if gen > 0:
          child_phenotypes = phenos[children_locs]
          reshaped=torch.reshape(child_phenotypes, (num_child, len(parent_locs), args.grn_size))
          stds=torch.std(reshaped,dim=(0))
          
          kid_stds = stds.mean(1).mean().item()

        # Evaluate fitnesses 
        fitnesses = fitness_function(phenos, targs[curr_targ])

        # SELECTION
        perm = torch.argsort(fitnesses, descending=True)
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

    return kid_stds

if __name__ == "__main__":
    #parser = argparse.ArgumentParser()

    args = FakeArgs()

    # DONT CHANGE FOR FIRST ROUND

    #parser.add_argument('-grn_size', type=int, default=50, help="GRN size") # number of genes in the GRN
    args.grn_size = 50
    #parser.add_argument('-pop_size', type=int, default=1000, help="Population size")
    args.pop_size = 1000
    #parser.add_argument('-alpha', type=int, default=10, help="Alpha for sigmoid function")
    args.alpha = 10
    #parser.add_argument('-num_genes_consider', type=float, default=0.5, help="proportion of genes considered for fitness")
    args.num_genes_consider = 1
    #parser.add_argument('-max_age', type=int, default=30, help="max age at which individual is replaced by its kid")
    args.max_age = 1000000000000000
    #parser.add_argument('-proj', type=str, default="EC_final_project", help="Name of the project (for wandb)")
    args.proj = "phd_chapt_3"
    #parser.add_argument('-selection_size', type=float, default=1, help="what proportion of the population to test for strategy (specialist, generatist)")
    args.selection_prop = 0.1
    #parser.add_argument('-exp_type', type=str, default="BASIC", help="Name your experiment for grouping")
    args.exp_type = "basic_sl_20"
    #parser.add_argument('-num_generations', type=int, default=100000, help="number of generations to run the experiment for") # number of generations
    args.num_generations = 1500
    
    
    # DO CHANGE
    #parser.add_argument('-mut_rate', type=float, default=0.1, help="rate of mutation (i.e. number of genes to mutate)")
    args.mut_rate = 0.1
    #parser.add_argument('-mut_size', type=float, default=0.5, help="size of mutation")
    args.mut_size = 0.5
    #parser.add_argument('-truncation_prop', type=float, default=0.2, help="proportion of individuals selected for reproduction")
    args.truncation_prop = 0.2
    #parser.add_argument('-season_len', type=int, default=100, help="number of generations between environmental flips")
    args.season_len = 20
    
    
    

    #args = parser.parse_args()

    print("running code")

    args.max_iter = 100 #int(3*args.grn_size) # "Maximum number of GRN updates") # number of times gene concentrations are updated to get phenotype

    args.truncation_size=int(args.truncation_prop*args.pop_size)


    print(args)

   
    assert (
        args.pop_size % args.truncation_size == 0
    ), "Error: select different trunction_prop, received {args.pops_size}"
    assert ( int(args.num_genes_consider*args.grn_size) % 2 == 0), "Error: select different num_genes_consider, needs to be a multiple of 2"

    evolutionary_algorithm(args)

