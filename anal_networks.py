import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import torch


device = "cuda" if torch.cuda.is_available() else "cpu" #for testing on my computer

class CPU_Unpickler(pickle.Unpickler):
    import io
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

class FakeArgs:
    """
    A simple class imitating the args namespace
    """

    def __repr__(self):
        attrs = vars(self)
        return "\n".join([f"{k}: {v}" for k, v in attrs.items()])

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

# open a file, where you stored the pickled data
#file = open('results/fresh-frost-6/basic_place_holder_title.pkl', 'rb')
#file = open('results/sweet-peony-128/basic_place_holder_title.pkl', 'rb')
#file = open('dancing_fireworks.pkl', 'rb') 
file = open('results/bright-moon-137/testing.pkl', 'rb')

# dump information to that file
#data = pickle.load(file)
data = CPU_Unpickler(file).load()
args = data["args_used"] #arguments used in this experiment

# plot weight distribution
sns.distplot(torch.flatten(data["best_grns"][0])) #density=True, alpha = 0.3)
sns.distplot(torch.flatten(data["best_grns"][-1]))

final_weights=data["best_grns"][-1]
best_pheno=get_phenotypes(args, data["best_grns"][-1].to(device), 1, 3, False)

# plot weights of best solution as a matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(data["best_grns"][-1])
fig.colorbar(cax)

# plot phenotype of best solution
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.plot(best_pheno[0][0].cpu())

# make clones for best solution
num_clones=20
complexities = 0
num_genes_fit = int(args.num_genes_consider*args.grn_size)

clones = data["best_grns"][-1].to(device).repeat([num_clones, 1, 1]) # create copies of parents

# Mutate clones
num_genes_mutate = int(args["grn_size"]*args["grn_size"]*len(clones) * args["mut_rate"])
mylist = torch.zeros(args["grn_size"]*args["grn_size"]*len(clones), device=device)
mylist[:num_genes_mutate] = 1
shuffled_idx = torch.randperm(args["grn_size"]*args["grn_size"]*len(clones), device=device)
mask = mylist[shuffled_idx].reshape(len(clones),args["grn_size"],args["grn_size"]) #select genes to mutate
clones = clones + (clones*mask)*torch.randn(size=clones.shape, device=device) * args["mut_size"]  # mutate only children only at certain genes

# Get clone phenotypes
clone_states=get_phenotypes(args, clones, num_clones, complexities, if_comp= False)
clone_phenos = clone_states[:,:,:num_genes_fit]
        
#c_genotypes = torch.stack(c_genotypes)
#c_phenotypes = torch.stack(c_phenotypes)

# Calculate fitness from phenotypes of clones
#c_phenotypes=torch.squeeze(c_phenotypes)
#tops=c_phenotypes[: , :  , :int(num_genes_fit/2)].sum(axis=-1, keepdims=True)
#bots=c_phenotypes[: , :  , int(num_genes_fit/2):].sum(axis=-1, keepdims=True)

clones.size()

torch.squeeze(clone_phenos).size()

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(torch.squeeze(clone_phenos).cpu())
fig.colorbar(cax)

def get_internal_states(grn):
  grn=grn.to(device)
  states=[]
  state = torch.zeros(1, 1, args["grn_size"]).to(device)
  state[:, :, 0] = 1.0 # create input to the GRNs

  state_before = torch.zeros(1, 1, args["grn_size"]).to(device) # keeping track of the last state
  for l in range(args["max_iter"]):
    states.append(state)
    state = torch.matmul(state, grn) # each matrix in the population is multiplied
    state = state * args["alpha"]
    state = torch.sigmoid(state) # after which it is put in a sigmoid function to get the output, by default alpha = 1 which is pretty flat, so let's use alpha > 1 (wagner uses infinite) hence the above multiplication
    # state = dround(state, 2)
    diffs=torch.abs(state_before - state).sum(axis=(1,2))
    which_repeat = torch.where(diffs == 0)
    state_before = state

  states=torch.stack(states)
  states=torch.flatten(states,start_dim=1)
  return states

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
cax = ax.matshow(get_internal_states(data["best_grns"][-1]).cpu()[:50,:])
fig.colorbar(cax)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
cax = ax.matshow(get_internal_states(data["best_grns"][-1]).cpu()[:50,:]-get_internal_states(clones[2]).cpu()[:50,:])
fig.colorbar(cax)

disagreement = get_internal_states(data["best_grns"][-1]).cpu()[:50,:]-get_internal_states(clones[2]).cpu()[:50,:]
plt.plot(disagreement.abs().sum(axis=1))

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
cax = ax.matshow(get_internal_states(clones[2]).cpu()[:50,:])
fig.colorbar(cax)

fig,axs = plt.subplots(nrows=5,ncols=4, figsize=(10,12))
for i,ax in enumerate(axs.flatten()):
  start = get_internal_states(clones[i]).cpu()[:50,:]
  end = get_internal_states(clones[i]).cpu()[-50:,:]
  ax.imshow(torch.vstack((start, end)), interpolation="nearest")
  #ax.imshow(get_internal_states(clones[i]).cpu()[:4,:], interpolation="nearest")
  ax.axis("off")
plt.tight_layout()
plt.subplots_adjust(hspace=0.05,wspace=0.05)
plt.show()



fig,axs = plt.subplots(nrows=20,ncols=1, figsize=(10,12))
for i,ax in enumerate(axs.flatten()):
  #start = get_internal_states(clones[i]).cpu()[-1:,:]
  #end = get_internal_states(clones[i]).cpu()[2:3,:]
  #ax.imshow(torch.vstack((start, end)), interpolation="nearest")
  ax.imshow(get_internal_states(clones[i]).cpu()[2:3,:], interpolation="nearest")
  ax.axis("off")
plt.tight_layout()
plt.subplots_adjust(hspace=0.,wspace=0.)
plt.show()











"""-------"""

divmod(torch.abs(final_weights).argmax().item(), final_weights.shape[1]) #coordinates of highest deviation from 0

torch.abs(final_weights).mean(0).argmax() # which column on average highest deviation from 0

torch.abs(final_weights).mean(1).argmax() # which row on average highest deviation from 0

tops=torch.topk(torch.abs(final_weights).flatten(), 3).indices
for t in tops:
  print(t.item())
  print(divmod(t.item(), final_weights.shape[1]))

def calc_rob(pop, ssize, rounds, args):
    ave_exp_rob = []
    ave_fit_rob = []
    ave_ben_mut = []
    ave_del_mut = []
    sample = np.random.choice(pop, ssize, replace=False)
    for p in sample:
        exp_rob = 0
        fit_rob = 0
        ben_mut = 0
        del_mut = 0
        ori_grn_out = p.grn_output()
        ori_fitness = fitness_function(ori_grn_out,envs,state,args)
        for mut in range(rounds):
            sample_cp = deepcopy(p)
            sample_cp.mut_edge()
            new_grn_out = sample_cp.grn_output()
            new_fitness = fitness_function(new_grn_out,envs,state,args)
            if np.array_equal(ori_grn_out,new_grn_out):
                exp_rob += 1
                fit_rob += 1
            else:
                if new_fitness > ori_fitness:
                    ben_mut +=1
                elif new_fitness < ori_fitness:
                    del_mut += 1
                else:
                    fit_rob += 1
        if exp_rob > 0:
            ave_exp_rob.append(exp_rob/rounds)
        else:
            ave_exp_rob.append(exp_rob)
        if fit_rob > 0:
            ave_fit_rob.append(fit_rob/rounds)
        else:
            ave_fit_rob.append(fit_rob)
        if del_mut > 0:
            ave_del_mut.append(del_mut/(rounds - fit_rob))
        else:
            ave_del_mut.append(del_mut)
        if ben_mut > 0:
            ave_ben_mut.append(ben_mut/(rounds - fit_rob))
        else:
            ave_ben_mut.append(ben_mut)
    return(np.average(ave_exp_rob), np.average(ave_fit_rob), np.average(ave_ben_mut), np.average(ave_del_mut))