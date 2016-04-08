from numpy import array
from copy import copy
from random import randint, random, sample
from numpy import e
import matplotlib.pyplot as plt

class Integer:
    """
    Simple wrapper around int, with ability to remember it's position in two dimensional space
    """

    def __init__(self, data: int, x: int, y: int):
        self.data = data
        self.x = x
        self.y = y

    def __int__(self):
        return self.data

    def __repr__(self):
        return repr(self.data)


def read_file():
    with open('sudoku.txt', 'r') as file:
        m = [[int(c) if c != 'x' else -1 for c in line.split()] for line in file]
    return array(m)


def generate_input(input_quads):
    """
    This function filles the fields of sudoku passed by it's argument, such way that they satisfy sudoku condition regarding
    lists in wich they are passed (input_quads means that we want the quarters of the game field to have sudoku restrictions met,
    but it can also be columns, rows)
    :param input_quads:
    :return:
    """
    for rect in input_quads:
        values_dict = ints_from_integers(rect)
        for val in rect:
            tmp = randint(1, 9)
            while val.data == -1:
                if tmp in values_dict:
                    tmp = randint(1, 9)
                else:
                    values_dict.add(tmp)
                    val.data = tmp


def ints_from_integers(input_data):
    return {integer.data for integer in input_data}


def generate_new_state(quads, data_quads, rows, columns):
    """
    This function generates new state of the task, by swapping values inside one quarter (can be row/column)
    :param quads: space of the simulation
    :param data_quads: input state (to indicate which fields can be swapped, and which remained untouched)
    :param rows: subsequent rows of the sudoku table
    :param columns: subsequent columns
    :return: new cost, list of performed swaps
    """
    swaps = list()
    quad, data_quad = sample(list(zip(quads, data_quads)), 1)[0]
    tmp = swap_valid_candidates(quad, data_quad)
    while tmp == (-1, -1):
        tmp = swap_valid_candidates(quad, data_quad)
    swaps.append((tmp, quad))
    return calculate_whole_cost(columns, rows), swaps


def calculate_whole_cost(columns, rows):
    return sum([calculate_cost(row) for row in rows] + [calculate_cost(column) for column in columns])


def swap_valid_candidates(quad, data_quad):
    """
    This function performs swap of the values in the sudoku fields if it's possible
    :param quad:
    :param data_quad:
    :return: (-1,-1) if swap was imposible, else (a,b) where a,b - numbers of fields swapped (regarding their quad)
    """
    if len(list(filter(lambda x: x == -1, data_quad))) == 0:
        return -1, -1
    a, b = randint(0, 8), randint(0, 8)
    while data_quad[a] != -1:
        a = randint(0, 8)
    while data_quad[b] != -1 or b == a:
        b = randint(0, 8)
    quad[a].data, quad[b].data = quad[b].data, quad[a].data
    return a, b


def restore_swaps(swaps):
    """
    This function restores the state of the sudoku table
    :param swaps:
    :return:
    """
    for pair, quad in swaps:
        quad[pair[0]].data, quad[pair[1]].data = quad[pair[1]].data, quad[pair[0]].data


def calculate_cost(row_column):
    """
    This function calculates the cost of the sudoku table
    :param row_column:
    :return:
    """
    sum_dict = {i: 0 for i in range(10)}
    for numb in row_column:
        sum_dict[numb.data] += 1
    return max(sum_dict.values()) - 1


def acceptance(old_cost, new_cost, temperature):
    return e ** (min((old_cost - new_cost) / temperature, 1.0))




def algo(data):
    current_state = array([[Integer(data[i][j], i, j) for j in range(9)] for i in range(9)])
    rows = array([[current_state[i][j] for j in range(0, 9)] for i in range(0, 9)])
    columns = array([[current_state[j][i] for j in range(0, 9)] for i in range(0, 9)])
    helper = array([(0, 0), (0, 3), (0, 6), (3, 0), (3, 3), (3, 6), (6, 0), (6, 3), (6, 6)])
    quads = array(
        [array([current_state[helper[i][0] + x][helper[i][1] + y] for x in range(3) for y in range(3)]) for i in
         range(9)])
    data_quads = array([array([data[helper[i][0] + x][helper[i][1] + y] for x in range(3) for y in range(3)]) for i in
                        range(9)])
    generate_input(quads)
    temperature = 300.0
    alpha = 0.9
    energy = calculate_whole_cost(columns, rows)
    energy_list = [energy]
    while energy > 0 and temperature > 0.001:
        for i in range(40):
            new_energy, swaps = generate_new_state(quads, data_quads, rows, columns)
            if new_energy == 0:
                return current_state, calculate_whole_cost(columns, rows), energy_list
            if acceptance(energy, new_energy, temperature) < random():
                restore_swaps(swaps)
            else:
                energy = new_energy
                energy_list.append(energy)
        temperature *= alpha

    return current_state, calculate_whole_cost(columns, rows), energy_list


if __name__ == '__main__':
    data = read_file()
    best_result = None
    best_result_energy_list = None
    lowest_cost = 100
    i = 0
    while i < 100 and lowest_cost > 0:
        new_state, new_cost, result_energy_list = algo(data)
        if (new_cost < lowest_cost):
            lowest_cost = new_cost
            best_result = new_state
            best_result_energy_list = result_energy_list
        i += 1
    print(best_result)
    print(lowest_cost)
    plt.plot(best_result_energy_list)
    plt.show()
