# PhD chapter 3

Collecting progress on my 3rd chapter.

# Plan number 1

The simple GRN model:

1 pop of individuals are matrices. Same input always. Output if result of iterative matrix multiplication. All nodes taken into consideration in fitness calculation. Truncation selection.

Optimal phenotypes: 2 random distributions.

## Questions/Things to measure

### Basics:

- *Average fitness of time. Also, distribution of fitnesses over time.*
- *Best strategies.*
- *Ages.*
- *Complexity.*

### Does evolvability evolve? Measure by:

- *fitness decrease when env changes*
- *num gens needed to regain prev fitness - best fitness and average fitness*
- *due to new muts or standing genetic variation? well, if we see the best improve - new mut, if just average improves - standing likely*
- *classify everyone as specialist A or B or neither or generalist*
- interesting that apart from the begining the number of "low" is very low, even after env change. is it just because I am sampling after selection? whould this be the case looking after mutation phase?
- <u>std among phenotypes of children of same parent</u> 


### What kind of networks evolve? Measure by:

- *distribution of weight values # calc retrospectively, save best network each gen*
- which weights large 
- <u>phenotypes over time</u>
- <u>robustness</u> # calc retrospectively, save best network each gen
- <u>modularity</u> # calc retrospectively, save best network each gen

### Variables to test:

Note: 5000 gens, 1000 pop_size with tracking, setting up, changing, and everything: time to run: ~12 min

Plan: run 1 with longer num gens -> if nothing interesting after 5000 -> repeat 4 times until 5000

Will need to run for more gens if season length longer to be able to compare!

Lapos figures, if decreased alpha more boring patterns

increasing alpha increasing convergence time plot

values of best solution phenotypes over time - why complexity decreases when averaging strategy emerges (is this true, check first), causation or correlation

Make optimal phenotype 000011111, 111110000 for better visualisation

time_to_rebound 0 non0 0 non0 0 non0 because spec_B is harder to learn because you are not starting from 50% fitness but from the opposite target - hence B has to catch up hence no rebound after env switch from A

- Frequency of env change [20,100,500] WIP
- Mut rates (introduce new ones?)
- Alpha (1,10,50)
- Number of updates to network (max_iter) (what if no network)
- Pop size
- max age
- network size
- selection pressure (truncation_size)

### To change after if there is time:

- only considering select nodes in fitness (implemented already)
- complexity part of fitness function? 0 fitness for non-converging? (easy)
- more than 1 pop (implemented already)
- Crossover? (implemented already)

## Pseudocode

Make population of size pop_size, and for each a random matrix of grn_size x grn_size. Normal distribution with mean 0 and variance 1 (also called the standard normal distribution).

Create optimal phenotypes: binary, random 1s and 0s. Alternative is exact opposite.

For gen in number of generations:

- Calculate phenotype for each individual: multiply with previous state, *alpha, then feed into sigmoid
- Calculate fitnesses (distance from optimal): (1 - torch.abs(pop.squeeze(1) - targ)).sum(axis=1), 1- sum of abs values of diffs, so the smaller the difference, the higher the fitness
- Non-convergers are not pelanised (by default)
- Sort the population according to fitness, calc num kids to create for each top parent: int(args.pop_size/args.truncation_size) - 1 -> -1 because we don't replace parents, children are mutated copies. Mutation: select x weights (mut_rate, percent of weights to mutate. 0.1 = 10% of edges will be mutated) in the matrix, apply noise to them (mut_size is magnitude of noise). Noise drawn from normal distribution with mean 0 and variance 1 (also called the standard normal distribution). Put children into pop.
  - -> net effect: top individuals remain unchanged in the pop, and the rest of the pop is fill with children of these top individuals, every individual reproduces the same number of times, and then the children are mutated 
- [Replace old individuals with their kid] - TODO change by just setting their fitness to 0

Change seasons every x generations.

Default parameters:


