
from GeneticAlgorithm import GeneticAlgorithm
import numpy as np

class GAConfig:
    n_generations = 10
    n_individuals = 10
    # Mutation Settings
    r_mutation = 0.06
    # Crossover Settings
    r_crossover = 0.6
    n_crossover = 1
    # Elitism Settings
    n_elites = 3

df = np.load('../Data/matrices.npy')
GA = GeneticAlgorithm(df, GAConfig)
GA.run_generations()
GA.plot_fitness()
print("Done!")
