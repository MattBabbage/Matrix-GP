from sympy.parsing.sympy_parser import parse_expr
from graphviz import Source
from sympy import dotprint, sympify, simplify, expand
from numpy import nan
import numpy as np
import random
from copy import deepcopy
import sys
from time import time


class Tree:
    def __init__(self, m_node, m_equation, m_graph, c_node, c_equation, c_graph, m_variables, c_variables):
        self.n_nodes = None

        self.m_node_matrix = m_node
        self.m_equation_matrix = m_equation
        self.m_graph_matrix = m_graph

        self.c_node_matrix = c_node
        self.c_equation_matrix = c_equation
        self.c_graph_matrix = c_graph

        self.m_variables = m_variables
        self.c_variables = c_variables

        self.fitness_matrix = None

    def refresh_c_equation_from_node(self, n):
        eq = self.c_node_matrix[n].get_equation_as_string()
        eq2 = simplify(parse_expr(eq))
        self.c_equation_matrix[n] = eq2

    def refresh_c_tree_from_equation(self, eq, n):
        try:
            self.c_node_matrix[n].refresh_node_from_equation(eq)
            self.refresh_c_graph_from_equation(n)
        except Exception as e:
            print("Issue in refresh_tree_from_equation", e)

    def refresh_c_graph_from_equation(self, n):
        self.c_graph_matrix[n] = Source(dotprint(self.c_equation_matrix[n]))

    def refresh_m_equation_from_node(self, n):
        eq = self.m_node_matrix[n].get_equation_as_string()
        eq2 = simplify(parse_expr(eq))
        self.m_equation_matrix[n] = eq2

    def refresh_m_tree_from_equation(self, eq, n):
        try:
            self.m_node_matrix[n].refresh_node_from_equation(eq)
            self.refresh_m_graph_from_equation(n)
        except Exception as e:
            print("Issue in refresh_tree_from_equation", e)

    def refresh_m_graph_from_equation(self, n):
        self.m_graph_matrix[n] = Source(dotprint(self.m_equation_matrix[n]))

    def check_fitness(self, df, variables):
        # Variables [a1,...,an,b1,...,bn]

        e_m = self.m_equation_matrix
        e_c = self.c_equation_matrix
        # e = self.equation_matrix
        # print(self.node.get_count())
        # print(e)
        total_nodes = 0
        for i in self.m_node_matrix:
            total_nodes += i.get_count()
        for i in self.c_node_matrix:
            total_nodes += i.get_count()

        errors = []
        for n_row in range(0, df.shape[0]):
            subs_array = []
            c_subs_array = []
            flattened_matrix = np.concatenate([df[n_row][0].flatten(), df[n_row][1].flatten()])
            flattened_matrix1 = df[n_row][1].flatten()
            for n in range(0, len(flattened_matrix)):
                subs_array.append([variables[n], flattened_matrix[n]])

            st = time()
            try:
                # print(str(e))
                ea = sympify(np.array(e_m)).subs(subs_array)

                for n in range(0, len(ea)):
                    c_subs_array.append([self.c_variables[n], ea[n]])

                c_ea = sympify(e_c).subs(c_subs_array)

                realans = df[n_row][-1].flatten()
                errors.append(np.absolute(np.array(realans) - np.array(c_ea)))
                # print(time() - st)
            except:
                print(time() - st)
                print("Something went wrong with the fitness check for eq: " + str(self.m_equation_matrix))
                errors.append(sys.float_info.max)
                print("Set error to max float value")
        if np.mean(errors) is nan:
            print("NAN ERROR!!!")
            return 0.9999

        self.fitness_matrix = np.mean(errors, axis=0, dtype=float)
        average_dif = np.mean(errors, dtype=float)
        return average_dif, total_nodes

    def mutate(self, operations, r_mut):
        n_mutations = 0
        for n in range(0, len(self.m_node_matrix)):
            if self.m_node_matrix[n].mutate(operations, self.m_variables, r_mut):
                self.refresh_m_equation_from_node(n)
                self.refresh_m_graph_from_equation(n)#
                n_mutations += 1

        for n in range(0, len(self.c_node_matrix)):
            if self.c_node_matrix[n].mutate(operations, self.c_variables, r_mut):
                self.refresh_c_equation_from_node(n)
                self.refresh_c_graph_from_equation(n)
                n_mutations += 1

        return n_mutations

        #
        # for x in range(0, len(self.node_matrix[0])):
        #     for y in range(0, len(self.node_matrix[1])):
        #         if self.node_matrix[x, y].mutate(operations, variables, r_mut):
        #             self.refresh_equation_from_node(x, y)
        #             #self.graph_matrix[x, y].render('output_premutation.gv', view=True)
        #             self.refresh_graph_from_equation(x, y)
        #             #self.graph_matrix[x, y].render('output_postmutation.gv', view=True)
        #             # print("Mutated!")

    def crossover(self, mate, k_points, r_cross):
        try:
            for i in range(k_points):
                n1 = np.random.randint(0, len(self.m_equation_matrix))
                n2 = np.random.randint(0, len(self.m_equation_matrix))

                if random.random() < r_cross:
                    # Pick whether its M or C getting crossover
                    selected = np.random.randint(0, len(self.m_equation_matrix) + len(self.c_equation_matrix))
                    if selected < len(self.m_equation_matrix):
                        mate = self.m_crossover(mate, r_cross)
                    else:
                        mate = self.c_crossover(mate, r_cross)
        except Exception as e:
            print(e)
            print("Crossover Failed")
        return [self, mate]

    def m_crossover(self, mate, r_cross):
        try:
            # Select 2 random ones
            n1 = np.random.randint(0, len(self.m_equation_matrix))
            n2 = np.random.randint(0, len(self.m_equation_matrix))

            if random.random() < r_cross:
                self.refresh_m_tree_from_equation(self.m_equation_matrix[n1], n1)
                mate.refresh_m_tree_from_equation(self.m_equation_matrix[n2], n2)

                self_count = self.m_node_matrix[n1].get_count()
                if self_count == 0:
                    self_selected = 1
                else:
                    self_selected = random.randrange(1, self.m_node_matrix[n1].get_count())

                mate_count = mate.m_node_matrix[n2].get_count()
                if mate_count == 0:
                    mate_selected = 1
                else:
                    mate_selected = random.randrange(1, mate.m_node_matrix[n2].get_count())

                self_node, ph = deepcopy(self.m_node_matrix[n1].get_node(self_selected))
                mate_node, ph = deepcopy(mate.m_node_matrix[n2].get_node(mate_selected))

                self.m_node_matrix[n1].replace_node(self_selected, mate_node)
                mate.m_node_matrix[n2].replace_node(mate_selected, self_node)

                self.refresh_m_equation_from_node(n1)
                self.refresh_m_graph_from_equation(n1)
                mate.refresh_m_equation_from_node(n2)
                mate.refresh_m_graph_from_equation(n2)

        except Exception as e:
            print(e)
            print("Crossover Failed")
        return mate

    def c_crossover(self, mate, r_cross):
        try:
            # Select 2 random ones
            n1 = np.random.randint(0, len(self.c_equation_matrix))
            n2 = np.random.randint(0, len(self.c_equation_matrix))

            if random.random() < r_cross:
                self.refresh_c_tree_from_equation(self.c_equation_matrix[n1], n1)
                mate.refresh_c_tree_from_equation(self.c_equation_matrix[n2], n2)

                self_count = self.c_node_matrix[n1].get_count()
                if self_count == 0:
                    self_selected = 0
                else:
                    self_selected = random.randrange(1, self.c_node_matrix[n1].get_count())

                mate_count = mate.c_node_matrix[n2].get_count()
                if mate_count == 0:
                    mate_selected = 0
                else:
                    mate_selected = random.randrange(1, mate.c_node_matrix[n2].get_count())

                self_node, ph = deepcopy(self.c_node_matrix[n1].get_node(self_selected))
                mate_node, ph = deepcopy(mate.c_node_matrix[n2].get_node(mate_selected))

                self.c_node_matrix[n1].replace_node(self_selected, mate_node)
                mate.c_node_matrix[n2].replace_node(mate_selected, self_node)

                self.refresh_c_equation_from_node(n1)
                self.refresh_c_graph_from_equation(n1)
                mate.refresh_c_equation_from_node(n2)
                mate.refresh_c_graph_from_equation(n2)

        except Exception as e:
            print(e)
            print("Crossover Failed")
        return mate


# Source(dotprint(parse_expr(self_node.get_equation_as_string()))).render('self_selected_node.gv', view=True)
# Source(dotprint(parse_expr(mate_node.get_equation_as_string()))).render('mate_selected_node.gv', view=True)

# Source(dotprint(parse_expr( self.node_matrix[x, y].get_equation_as_string()))).render('self_post_node.gv', view=True)
# Source(dotprint(parse_expr( mate.node_matrix[x, y].get_equation_as_string()))).render('mate_post_node.gv', view=True)
# print("Self Post Nut: ", self.equation_matrix[x, y])
# print("Mate Post Nut: ", mate.equation_matrix[x, y])
# self.graph.render('self_post_cross.gv', view=True)
# mate.graph.render('mate_post_cross.gv', view=True)