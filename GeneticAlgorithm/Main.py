
from GeneticAlgorithm import GeneticAlgorithm
import numpy as np

class GAConfig:
    n_generations = 200
    n_individuals = 10
    # Mutation Settings
    r_mutation = 0.02
    mutation_balancing = True
    # Balance @ ~1 Change per individual
    mutation_balance_value = 1.5
    mutation_increment_value = 0.001
    mutation_r_expansion = 0.1
    # Crossover Settings
    r_crossover = 0.8
    n_crossover = 1
    # Elitism Settings
    n_elites = 3

df = np.load('../Data/matrices.npy')
GA = GeneticAlgorithm(df, GAConfig)
GA.run_generations()
GA.plot_fitness()
print("Done!")

# Issues
# C Layer is being ignored - this overspecialises m later
# Should lock it to contain at least 3 nodes?
# Extract mutation variables
# Multiplications coming up?
# Via Mutation? Crossover? Simplification?
