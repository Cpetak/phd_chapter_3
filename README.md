# PhD chapter 3

Collecting progress on my 3rd chapter.

# Plan number 1

The simple GRN model:

1 pop of individuals are matrices. Same input always. Output if result of iterative matrix multiplication. All nodes taken into consideration in fitness calculation. Truncation selection.

Optimal phenotype: the 2 distributions.

## Questions/Things to measure

### Basics:

- Average fitness of time. Also, distribution of fitnesses over time.
- Best strategies.
- Ages.

### Does evolvability evolve? Measure by:

- fitness decrease when env changes (abs and relative)

- num gens needed to regain prev fitness 

- - due to new muts or standing genetic variation
  - classify everyone as specialist A or B or neither or generalist

### What kind of networks evolve? Measure by:

- distribution of weight values
- robustness (expression and fitness)
- modularity

### Variables to test:

- Mut rates (introduce new ones?)
- Number of updates to network
- Pop size
- Frequency of env change
- max age

### To change after if there is time:

- only considering select nodes in fitness
- env change is different
- more than 1 pop