from Population import Population
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy
import sys
import sympy
from sympy import simplify
import datetime
import os


class GeneticAlgorithm:
    def __init__(self, train_df, config):
        self.name = "RUN_" + f"{datetime.datetime.now():%Y-%m-%d-%H-%M}"
        self.children = []
        self.train_df = train_df
        self.population = []
        # Fitness
        self.fittest_scores = []
        self.q1_fitness = []
        self.q2_fitness = []
        self.q3_fitness = []
        # Fitness
        self.n_mutations = []
        self.average_depth = []
        # Elites
        self.elites = []
        self.elites_fitness = []
        self.n_eqs = train_df.shape[-1]**2
        # Variation
        self.genetic_variations = []
        # Working Solutions
        self.solutions = []
        self.solution_matrices = []
        # Config Settings
        self.config = config
        #
        self.mutation_each_gen = []

    def save_solutions(self):
        df = pd.DataFrame(self.solutions)
        output_dir = 'RunLogs'
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        df.to_csv(output_dir + '/' + self.name + '.csv', index=False, header=["Fitness", "Equations"])

    def run_generations(self):
        print("Starting...")
        variables = []
        for column in ["a","b"]:
            for row in range(0,self.n_eqs):
                variables.append(column+str(row + 1))
        # variables += ["-1", "0", "1", "2"] # Add extra variable to the list
        operations = ["+", "-"]
        # Generate Initial Population
        self.population = Population(self.config.n_individuals, operations, variables)
        # Calculate Initial Fitness
        self.population.check_fitness(self.train_df)
        self.save_elites()
        self.log_scores(0)
        # Setup loop for all generations
        for n_gen in range(self.config.n_generations):
            print("Generation " + str(n_gen))
            print("     Mutation")
            n_mut = self.population.mutate(self.config.r_mutation)
            print("         Total Mutations: ", n_mut)
            print("     Selection")
            self.population.roulette_selection()   
            print("     Crossover")
            self.population.crossover(self.config.n_crossover, self.config.r_crossover)
            print("     Fitness")
            self.population.check_fitness(self.train_df)
            print("     Post Fitness")
            # self.population.print_all()
            self.log_scores(n_mut)
            # elitism
            self.replace_unfit_with_elites()
            print("     Post Replacements")
            # self.population.print_all()
            # self.population.print_all()
            self.save_elites()
            self.balance_settings(n_mut)
            # self.population.print_all()
            # self.population.check_fitness(train_df)
            if self.population.fittest_score == 0:
                print("Found Solution")
                return

    def balance_settings(self, n_mut):
        if self.config.mutation_balancing == True:
            n_desired_mutations = self.config.n_individuals * self.config.mutation_balance_value
            if n_mut < n_desired_mutations:
                self.config.r_mutation += self.config.mutation_increment_value
            else:
                self.config.r_mutation -= self.config.mutation_increment_value


    def log_scores(self, n_mut):
            fitness = np.array([x for x in self.population.all_fitness if x == x])

            fitness_no_outliers = np.array([y for y in fitness if not isinstance(y, sympy.core.numbers.Infinity)])
            fitness_no_outliers = np.array([z for z in fitness_no_outliers if not isinstance(z, type(sympy.core.numbers.nan))])
            # Don't log outliers that are huge i.e. 1.85267342779706e+336 as it cant be cast as float
            fitness = np.clip(fitness_no_outliers, 0, 1000)
            print(fitness)

            print("Fittest Score: %.2f" % self.population.fittest_score + " Equation: " + str(
                self.population.fittest_equation_matrix))
            self.fittest_scores.append(self.population.fittest_score)
            self.q1_fitness.append(np.percentile(fitness, 25))
            self.q2_fitness.append(np.percentile(fitness, 50))
            self.q3_fitness.append(np.percentile(fitness, 75))

            self.average_depth.append(np.mean(self.population.all_node_depth))
            self.n_mutations.append(n_mut)
            self.mutation_each_gen.append(self.config.r_mutation)
            #self.check_variation()

    # def check_variation(self):
    #     for individual in self.population.individuals:
    #         print(self.population.fittest_individual.equation_matrix)
    #         print(individual.equation_matrix)
    #         print(simplify(self.population.fittest_individual.equation_matrix - individual.equation_matrix))

    def reject_outliers(self, data, m=3):
        return data[abs(data - np.median(data)) < m  * np.std(data)]

    def plot_fitness(self):
        fig, (ax1, ax2, ax3) = plt.subplots(3,1)
        ax1.plot(self.fittest_scores, 'r-')
        ax1.plot(self.q1_fitness, 'r-.')
        ax1.plot(self.q2_fitness, 'g-.')
        ax1.plot(self.q3_fitness, 'b-.')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Fitness')
        ax1.set_ylim(0, 150)
        ax1.ticklabel_format(useOffset=False)
        ax1.axhline(y=0, color='k')

        print(self.fittest_scores)
        print(np.diff(self.fittest_scores))

        ax2.plot(self.n_mutations, 'r-', label='N mutations')
        ax2.plot(self.average_depth, 'b-', label='Average node tree depth')
        ax2.plot(np.absolute(np.append(np.diff(self.fittest_scores), 0) / self.fittest_scores * 100), 'g-',
                 label='% Improvement top fitness')
        ax2.set_xlabel('Generation')
        ax2.ticklabel_format(useOffset=False)
        ax2.axhline(y=0, color='k')
        ax2.legend(loc="upper left")

        ax3.plot(np.divide(self.n_mutations, self.config.n_individuals), 'r-', label='Mutations per individual')
        ax3.plot(np.multiply(np.subtract(self.average_depth,8), self.mutation_each_gen), 'b-', label='Predicted number of mutation')
        ax3.set_xlabel('Generation')
        ax3.ticklabel_format(useOffset=False)
        ax3.axhline(y=0, color='k')
        ax3.legend(loc="upper left")
        plt.show()

        print(np.absolute(np.diff(self.fittest_scores)))
        print(self.fittest_scores)
        print(np.absolute(np.append(np.diff(self.fittest_scores), 0) / self.fittest_scores * 100))

    def save_elites(self):

        fitness_with_elite = self.population.all_fitness + self.elites_fitness
        pop_with_elite = self.population.individuals + self.elites

        for index in range(len(fitness_with_elite)):
            if isinstance(fitness_with_elite[index], sympy.core.numbers.Infinity) or isinstance(fitness_with_elite[index], type(sympy.core.numbers.nan)):
                print("Found nan or infinity - its not so bad :)")
                list_of_reals = []
                for num in fitness_with_elite:
                    if num.is_real:
                        list_of_reals.append(num)
                fitness_with_elite[index] = np.max(list_of_reals) * 2

        fitness_with_elite_order_index = np.argsort(fitness_with_elite)
        # zipped_lists = zip(fitness_with_elite, range(len(fitness_with_elite), pop_with_elite))
        # sorted_pairs = sorted(zipped_lists)
        # pop_with_elite_in_order = [x for _, x in sorted(zip(fitness_with_elite, pop_with_elite))]
        self.elites = []
        elite_equations = []
        self.elites_fitness = []

        # Fit Elites
        elite_idx = 0
        for idx in fitness_with_elite_order_index:
            #Check fitness matches guy!
            f, n = pop_with_elite[idx].check_fitness(self.train_df, self.population.variables)
            if f != fitness_with_elite[idx]:
                print("Something gone very wrong here")

            concat_eq_matrices = deepcopy(np.concatenate([pop_with_elite[idx].m_equation_matrix, pop_with_elite[idx].c_equation_matrix]))
            # Check if the fittest has been logged, if not log it
            if idx == fitness_with_elite_order_index[0]:
                if not any(np.array_equal(concat_eq_matrices, i) for i in self.solution_matrices):
                    self.solution_matrices.append(concat_eq_matrices)
                    self.solutions.append([f, concat_eq_matrices])
                    self.save_solutions()

            if not self.elites:
                self.elites.append(deepcopy(pop_with_elite[idx]))
                self.elites_fitness.append(deepcopy(fitness_with_elite[idx]))
                elite_equations.append(concat_eq_matrices)
                elite_idx += 1
            elif not any(np.array_equal(concat_eq_matrices, i) for i in elite_equations):
                self.elites.append(deepcopy(pop_with_elite[idx]))
                self.elites_fitness.append(deepcopy(fitness_with_elite[idx]))
                elite_equations.append(concat_eq_matrices)
                elite_idx += 1
            if elite_idx == self.config.n_elites:
                break

        print("Fit Elites: ")
        for i in range(len(self.elites)):
            print("Score: %.2f" % self.elites_fitness[i])
            print(np.matrix(np.concatenate([self.elites[i].m_equation_matrix, self.elites[i].c_equation_matrix])))


    def replace_unfit_with_elites(self):
        x = np.array(self.population.all_fitness)
        for index in range(len(self.population.all_fitness)):
            if isinstance(x[index], sympy.core.numbers.Infinity) or isinstance(x[index], type(sympy.core.numbers.nan)):
                print("Found nan or infinity - its not so bad :)")
                list_of_reals = []
                for num in x:
                    if num.is_real:
                        list_of_reals.append(num)
                x[index] = np.max(list_of_reals) * 2
        inds = np.argsort(x)[::-1]

        print("Elites: ")
        for i in range(len(self.elites)):
            print("Score: %.2f" % self.elites_fitness[i])
            print(np.matrix(np.concatenate([self.elites[i].m_equation_matrix, self.elites[i].c_equation_matrix])))

        #
        # print("Worst: ")
        # for i in range(len(self.elites)):
        #     print("Score: %.2f" %  self.population.all_fitness[inds[i]])
        #     print(np.matrix(np.concatenate([self.population.individuals[inds[i]].m_equation_matrix, self.population.individuals[inds[i]].c_equation_matrix])))
        # for i in range(len(self.elites)):
        #     self.population.individuals[inds[i]] = deepcopy(self.elites[i])
        #     self.population.all_fitness[inds[i]] = deepcopy(self.elites_fitness[i])

