import sympy.core.numbers

from Node import Node
from Tree import Tree
from TreeGenerator import TreeGenerator
import random
from sympy.parsing.sympy_parser import parse_expr
from graphviz import Source
from sympy import dotprint
import numpy as np
from copy import deepcopy
import math
from time import time

class Population:
    def __init__(self, n_pop, operations, variables):
        self.operations = operations
        self.variables = variables
        self.n_pop = n_pop
        self.individual_generator = TreeGenerator(self.operations, self.variables, 2)
        # List of individuals
        self.individuals = self.individual_generator.gen_trees(2, 2, n_pop)
        self.all_fitness = []
        self.all_node_depth = []
        # Statistics about population
        self.fittest_score = None
        self.fittest_equation_matrix = None
        self.fittest_individual = None


    def check_fitness(self, df):
        try:
            self.all_fitness = []
            self.all_node_depth = []

            for individual in self.individuals:
                individual_fit, n_nodes = individual.check_fitness(df, self.variables)
                self.all_node_depth.append(n_nodes)
                if individual_fit != individual_fit:
                    print("Nan Nan")
                # if isinstance(individual_fit, sympy.core.numbers.nan):
                #     print("Nan Nan 1")

                self.all_fitness.append(individual_fit)

                if individual_fit is not sympy.core.numbers.nan:
                    if individual_fit == individual_fit: # Filters out NaN fitness
                        if self.fittest_score is None:
                            self.fittest_individual = []
                            self.fittest_score = deepcopy(individual_fit)
                            self.fittest_equation_matrix = deepcopy(np.concatenate([individual.m_equation_matrix, individual.c_equation_matrix]))
                            self.fittest_individual = deepcopy(individual)
                            print("first fittest score: " + str(self.fittest_score) + " equation_matrix: " + str(self.fittest_equation_matrix))

                        if individual_fit < self.fittest_score:
                            self.fittest_individual = []
                            self.fittest_score = deepcopy(individual_fit)
                            self.fittest_equation_matrix = deepcopy(np.concatenate([individual.m_equation_matrix, individual.c_equation_matrix]))
                            self.fittest_individual = deepcopy(individual)
                            print("New fittest score! " + str(self.fittest_score) + " Equation: " + str(self.fittest_equation_matrix))

        except:
            print("exception occured")

    def mutate(self, r_mut, r_expansion):
        n_total_mutations = 0
        for i in self.individuals:
            n_total_mutations += i.mutate(self.operations, r_mut, r_expansion)
        return n_total_mutations

    def roulette_selection(self):
        new_individuals = []
        x = np.array(self.all_fitness)
        for index in range(len(self.all_fitness)):
            if isinstance(x[index], sympy.core.numbers.Infinity) or isinstance(x[index], type(sympy.core.numbers.nan)):
                print("Found nan or infinity - its not so bad :)")
                list_of_reals = []
                for num in x:
                    if num.is_real:
                        list_of_reals.append(num)
                x[index] = np.max(list_of_reals)

        # x = list(map(lambda idx: idx.replace(sympy.core.numbers.Infinity, (np.max(x))), x))
        if np.max(x)-np.min(x) == 0:
            print("----------------------------------------------------------------- o shit its div 0")
        norm_fitness = (x-np.min(x))/(np.max(x)-np.min(x) + 10**-100)
        inv_norm_fit = [(1 - fit) for fit in norm_fitness]

        # if they are all the same fitness, set inv norm fitness to all be the same
        if all(a==inv_norm_fit[0] for a in inv_norm_fit):
            inv_norm_fit = np.full(len(inv_norm_fit), 1)

        # cumulative_fitness = sum(self.all_fitness)
        # scores = [x / cumulative_fitness for x in self.all_fitness] # np.clip(self.all_fitness, None, 0.99)
        # a = np.sum(scores)
        # invscores = [(1 - prob) for prob in scores]
        for n_indiv in range(len(self.individuals)):
            selected = random.uniform(0, np.sum(inv_norm_fit))
            val = 0
            for i in range(len(self.individuals)):
                try:
                    if (val < selected and (val + inv_norm_fit[i]) > selected):
                        new_individuals.append(deepcopy(self.individuals[i]))
                        break
                    else:
                        val = val + inv_norm_fit[i]
                except:
                    print("An exception occurred")
        self.individuals = new_individuals

    def crossover(self, k_points, r_cross):
        try:
            child_individuals = []
            random.shuffle(self.individuals)  # Mix them up (just incase selection did not)
            i_p_half_pop = math.floor(self.n_pop / 2) # get idx at half pop (incrementing will allow pairing)
            for idx in range(math.floor(self.n_pop / 2)):
                child_individuals.extend(self.individuals[idx].crossover(self.individuals[i_p_half_pop], k_points, r_cross))
                i_p_half_pop = i_p_half_pop + 1
            self.individuals = child_individuals
        except Exception as e:
            print(e)
            print("Crossover Failed")

    def print_all(self):
        for i in range(len(self.individuals)):
            print(str(i) + " score: " + str(self.all_fitness[i]) + " equation: " + str(self.individuals[i].equation_matrix))


population = Population(5, ["*", "+", "-"], ["a", "b", "c", "pi"])

