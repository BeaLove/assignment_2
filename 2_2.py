""" This file is created as a suggested solution template for question 2.2 in DD2434 - Assignment 2.

    We encourage you to keep the function templates as is.
    However this is not a "must" and you can code however you like.
    You can write helper functions however you want.

    If you want, you can use the class structures provided to you (Node, Tree and TreeMixture classes in Tree.py
    file), and modify them as needed. In addition to the sample files given to you, it is very important for you to
    test your algorithm with your own simulated data for various cases and analyse the results.

    For those who do not want to use the provided structures, we also saved the properties of trees in .txt and .npy
    format. Let us know if you face any problems.

    Also, we are aware that the file names and their extensions are not well-formed, especially in Tree.py file
    (i.e example_tree_mixture.pkl_samples.txt). We wanted to keep the template codes as simple as possible.
    You can change the file names however you want (i.e tmm_1_samples.txt).

    For this assignment, we gave you three different trees (q_2_2_small_tree, q_2_2_medium_tree, q_2_2_large_tree).
    Each tree has 5 samples (whose inner nodes are masked with np.nan values).
    We want you to calculate the likelihoods of each given sample and report it.

    Note: The alphabet "K" is K={0,1,2,3,4}.
"""

import numpy as np
from Tree import Tree
from Tree import Node


def calculate_likelihood(tree_topology, theta, beta):
    """
    This function calculates the likelihood of a sample of leaves.
    :param: tree_topology: A tree topology. Type: numpy array. Dimensions: (num_nodes, )
    :param: theta: CPD of the tree. Type: numpy array. Dimensions: (num_nodes, K, K)
    :param: beta: A list of node assignments. Type: numpy array. Dimensions: (num_nodes, )
                Note: Inner nodes are assigned to np.nan. The leaves have values in [K]
    :return: likelihood: The likelihood of beta. Type: float.

    This is a suggested template. You don't have to use it.
    """

    # TODO Add your code here
    num_nodes, num_values = len(theta), len(theta[0])
    #sample_values = np.where(not np.isnan(beta))
    s_node_value = np.zeros((num_nodes, num_values))
    for i, value in enumerate(np.flip(beta)):
        node = len(beta) - i - 1
        #print("node", node, "value", value)
        if not np.isnan(value):
            s_node_value[node, int(value)] = 1
        else:
            #print(s_node_value)
            ##get children:
            children = []
            for index, parent in enumerate(tree_topology):
                #print("parent: ", parent, "node ", node)
                if parent == float(node):
                    children.append(index)
            #print("children ", children)
            for val in range(num_values):
                child_probabilities = []
                for child in children:
                    s_child = s_node_value[child, :]
                    print("s_child", s_child, "type ", type(s_child))
                    theta_val = theta[child][:][int(val)]
                    #print("THETA", theta_val)
                    prob_given_below = s_child * theta_val
                    #print("given below", prob_given_below)
                    #prob_given_below = []
                    child_probabilities.append(sum(prob_given_below))
                    #for i in range(num_values):
                        #print("theta array ", theta[child][i], type(theta[child][i]))

                        #print("prob given below: ", prob_given_below)
                        #for j in range(num_values):
                            #probability = s_node_value[child, j]*theta[child][i][j]
                            #prob_given_below.append(probability)
                        #print("prob given below ", prob_given_below)

                    #print("child probabilities:", child_probabilities)
                product = child_probabilities[0]*child_probabilities[1]
                #print("product", product)
                s_node_value[node, int(val)] = product
    likelihood = 1
    #print("s_node_value", s_node_value)
    likelihood = np.sum(s_node_value[0])
    '''for value in beta:
        if not np.isnan(value):
            likelihood *= s_node_value[0][int(value)]'''

    '''for node, value in enumerate(beta):
        #print(node)
        if np.isnan(value):
            #print("row", s_node_value[int(node), :])
            likelihood *= sum(s_node_value[int(node), :])'''

    return likelihood


def main():
    print("Hello World!")
    print("This file is the solution template for question 2.2.")

    print("\n1. Load tree data from file and print it\n")

    filename = "data/q2_2/q2_2_small_tree.pkl"  # "data/q2_2/q2_2_medium_tree.pkl", "data/q2_2/q2_2_large_tree.pkl"
    t = Tree()
    t.load_tree(filename)
    t.print()
    print("K of the tree: ", t.k, "\talphabet: ", np.arange(t.k))

    print("\n2. Calculate likelihood of each FILTERED sample\n")
    # These filtered samples already available in the tree object.
    # Alternatively, if you want, you can load them from corresponding .txt or .npy files

    #beta = t.filtered_samples[0]
    #sample_likelihood = calculate_likelihood(t.get_topology_array(), t.get_theta_array(), beta)
    #print("likelihood: ", sample_likelihood)
    for sample_idx in range(t.num_samples):
        beta = t.filtered_samples[sample_idx]
        print("\n\tSample: ", sample_idx, "\tBeta: ", beta)
        sample_likelihood = calculate_likelihood(t.get_topology_array(), t.get_theta_array(), beta)
        print("\tLikelihood: ", sample_likelihood)


if __name__ == "__main__":
    main()
