
from GeneticAlgorithm import GeneticAlgorithm
import numpy as np

class GAConfig:
    n_generations = 100
    n_individuals = 10
    # Mutation Settings
    r_mutation = 0.015
    mutation_balancing = True
    # Balance @ ~1 Change per individual
    mutation_balance_value = 1
    mutation_increment_value = 0.001
    # Crossover Settings
    r_crossover = 0.7
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

# Multiplications coming up?
# Via Mutation? Crossover? Simplification?
