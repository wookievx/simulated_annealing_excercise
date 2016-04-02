from math import exp
from random import random
from random import randint
from random import choice
import matplotlib.pyplot as plt
from numpy import array, delete, insert


class ListWithLabel:
    """
    This class provides easy interface for creating new state from existing one
    """
    def __init__(self, inner_data: array, cost: float):
        self.data = inner_data
        if not cost:
            self.cost = calculate_cost(self.data)
        else:
            self.cost = cost

    def get_neighbour(self, first, second):
        tmp_data = delete(self.data, second, 0)
        tmp_data = insert(tmp_data, first, self.data[second], 0)
        return ListWithLabel(tmp_data, None)

    def cost_difference(self, first, second):
        """
        This method DOESN'T work so far so it is not used
        It should be used to significantly increase time performance
        :param first:
        :param second:
        :return: diffrence in cost
        """
        tmp = 0.0
        tmp -= dist(self.data[first], self.data[first + 1])
        tmp -= dist(self.data[second], self.data[second - 1])
        tmp += dist(self.data[second], self.data[first + 1])
        tmp += dist(self.data[first], self.data[second - 1])
        if first > 0:
            tmp -= dist(self.data[first], self.data[first - 1])
            tmp += dist(self.data[second], self.data[first - 1])
        if second < len(self.data) - 1:
            tmp -= dist(self.data[second], self.data[second + 1])
            tmp += dist(self.data[first], self.data[second + 1])
        return tmp


def dist(first_pair, second_pair):
    return ((first_pair[0] - second_pair[0]) ** 2 + (first_pair[1] - second_pair[1]) ** 2) ** 0.5


def calculate_cost(list):
    """
    :param list:
    :return: cost (sum of distances between consecutive points in sequence) of the paramter list
    """
    tmp = 0
    for i in range(len(list) - 1):
        tmp += dist(list[i], list[i + 1])
    return tmp


def acceptance(old_cost, new_cost, T):
    return exp(min((old_cost - new_cost) / T, 1.0))


def simulated_annealing(start_input):
    """
    Function performs simulated annealing for the tsp problem
    :param start_input: input data
    :return: final state of the data, list of the costs
    """
    temperature = start_input.cost / len(start_input.data) * 100
    temperature_min = temperature / 100000
    alpha = 0.94
    cost_list = []
    while temperature > temperature_min:
        for i in range(200):
            a = randint(0, len(start_input.data) - 3)
            b = randint(a + 1, len(start_input.data) - 1)
            new_input = start_input.get_neighbour(a, b)
            ap = acceptance(start_input.cost, new_input.cost, temperature)
            if ap > random():
                start_input = new_input
            cost_list.append(start_input.cost)
        temperature *= alpha
    return start_input, cost_list


def data_generator(size, cluster, cluster_range):
    """

    :param size: number of nodes
    :param cluster: list of cluster seeds
    :param cluster_range: range of the single cluster
    :return:
    """
    return array([[random() * cluster_range + choice(cluster), random() * cluster_range + choice(cluster)] for i in
                  range(size)])


#example usage
if __name__ == '__main__':
    tmpData = ListWithLabel(data_generator(120, (0, 40, 80), 4), None)
    print(tmpData.data)
    print(tmpData.cost)
    for i in range(len(tmpData.data) - 1):
        plt.plot([tmpData.data[i][0], tmpData.data[i + 1][0]], [tmpData.data[i][1], tmpData.data[i + 1][1]], color='k',
                 linestyle='-', linewidth=2)
    plt.savefig('before_optimisation.png')
    tmpData, cost_list = simulated_annealing(tmpData)
    plt.clf()
    for i in range(len(tmpData.data) - 1):
        plt.plot([tmpData.data[i][0], tmpData.data[i + 1][0]], [tmpData.data[i][1], tmpData.data[i + 1][1]], color='k',
                 linestyle='-', linewidth=2)
    plt.savefig('after_optimastion.png')
    print(tmpData.data)
    print(tmpData.cost)
    plt.clf()
    plt.plot(range(0, len(cost_list)), cost_list)
    plt.savefig('cost.png')
