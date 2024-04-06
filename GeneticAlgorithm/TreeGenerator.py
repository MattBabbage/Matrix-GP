from Node import Node
from Tree import Tree
import random
from sympy.parsing.sympy_parser import parse_expr
from graphviz import Source
from sympy import dotprint
import numpy as np
from sympy import dotprint, sympify, simplify, expand

class TreeGenerator:
    def __init__(self, operations, m_variables, matrix_dimension=2, n_multiplications=8):
        self.operations = operations
        self.m_variables = m_variables
        self.matrix_dimension = matrix_dimension
        self.n_multiplications = n_multiplications
        self.c_variables = []
        for row in range(0, self.n_multiplications):
            self.c_variables.append("m"+str(row + 1))

    def gen_tree(self, depth=1, width=2, init_depth=0):
        # Declare Matrices and lists
        m_nodes = []
        m_equations = []
        m_graphs = []
        c_nodes = []
        c_equations = []
        c_graphs = []
        # Create M Equations
        # Loop over tree adding to lists
        for i in range(0, self.n_multiplications):
            new_node = Node("*", locked_multiplication=True)
            new_node.add_children(self.operations, self.m_variables, width, depth)
            new_equation = parse_expr(new_node.get_equation_as_string())
            m_nodes.append(new_node)
            m_equations.append(new_equation)
            m_graphs.append(Source(dotprint(new_equation)))

        # Create C Equations
        for i in range(0, self.matrix_dimension**2):
            new_node = Node(random.choice(self.operations), locked_operation=True)
            new_node.add_children(self.operations, self.c_variables, width, depth)
            new_equation = parse_expr(new_node.get_equation_as_string())
            c_nodes.append(new_node)
            c_equations.append(new_equation)
            c_graphs.append(Source(dotprint(new_equation)))

        return Tree(np.array(m_nodes), np.array(m_equations), np.array(m_graphs),
                    np.array(c_nodes), np.array(c_equations), np.array(c_graphs), self.m_variables, self.c_variables)

    def gen_trees(self, depth=1, width=2, n_trees=1, init_depth=0):
        trees = []
        for i in range(n_trees):
            trees.append(self.gen_tree(depth, width))
        return trees


# variables = ["a1", "a2","a3", "a4", "b1", "b2", "b3", "b4"]
# operations = ["+", "-"]
#
# tree_gen = TreeGenerator(operations, variables, 2)
# tree1 = tree_gen.gen_tree()
# tree2 = tree_gen.gen_tree()
#
# tree1.crossover(tree2,1,1)
# #tree1.mutate(operations,variables,0.2)
#
# # print(tree1.equation_matrix)
# # print("n mults: ", tree1.get_n_multiplications())
# #
# # tree1.equation_matrix[0, 0] = expand(simplify(parse_expr(tree1.node_matrix[0, 0].get_equation_as_string())))
#
#
# df = np.load('../Data/matrices.npy')
# fit = tree1.check_fitness(df, variables)
# print(df.shape[-1])
# print(fit)
# print("hello world!")