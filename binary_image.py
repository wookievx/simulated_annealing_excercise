from math import exp
from random import randint, random, sample
import matplotlib.pyplot as plt
from itertools import product
from PIL import Image


class Point:
    """
    This class wraps simple
    """

    def __init__(self, is_filled=True):
        self.positive_neighbourhood = list()
        self.negative_neighbourhood = list()
        self.neighbourhood = list()
        self.is_filled = is_filled

    @property
    def positive_values(self):
        return [1.0 if v.is_filled else 0.0 for v in self.positive_neighbourhood]

    @property
    def negative_values(self):
        return [1.0 if v.is_filled else 0.0 for v in self.negative_neighbourhood]

    def __str__(self):
        return str(self.is_filled)

    def __repr__(self):
        return repr(self.is_filled)


class Neighbourhood:
    def __init__(self, table, neighbourhood_positions, start_position, size):
        self.table = table
        self.neighbourhood_positions = neighbourhood_positions
        self.start_position = start_position
        self.size = size

    def __iter__(self):
        self.index = 0
        return self

    def is_index_valid(self, index):
        return 0 <= index < self.size

    def __next__(self):
        if not len(self.neighbourhood_positions) > self.index:
            raise StopIteration
        while not self.is_index_valid(self.start_position[0] + self.neighbourhood_positions[self.index][0]) \
                or not self.is_index_valid(self.start_position[1] + self.neighbourhood_positions[self.index][1]):
            self.index += 1
            if not len(self.neighbourhood_positions) > self.index:
                raise StopIteration
        self.index += 1
        return self.table[self.start_position[0] + self.neighbourhood_positions[self.index - 1][0]][
            self.start_position[1] + self.neighbourhood_positions[self.index - 1][1]]


def calculate_energy(point: Point, positive_factor, negative_factor):
    return 0. if not point.is_filled else sum(point.positive_values) * positive_factor - sum(
        point.negative_values) * negative_factor


def create_new_state(last_state, energy, positive_factor, negative_factor):
    """
    This function generates new state for our simulation, it simply swaps the 'is_filled' field between two Point objects
    and performs all needed calculations
    :param last_state: previous state of the simulation
    :param energy: previous energy of the simulation
    :param positive_factor: positive factor for energy calculation
    :param negative_factor: negative factor for energy calculation
    :return: position of points swapped, energy of the new state
    """

    # here we obtain fields to swap in order to create new state
    # ------
    def test_one(ituple):
        return last_state[ituple[0]][ituple[1]].is_filled

    def test_done(ituple):
        return not last_state[ituple[0]][ituple[1]].is_filled

    former_positions = [(randint(0, len(last_state) - 1), randint(0, len(last_state) - 1)) for i in
                        range(2 * len(last_state))]
    new_positions = [(randint(0, len(last_state) - 1), randint(0, len(last_state) - 1)) for i in
                     range(2 * len(last_state))]
    former_positions = list(filter(test_one, former_positions))
    new_positions = list(filter(test_done, new_positions))
    former_positions = former_positions[:min(len(former_positions), len(new_positions), 3)]
    new_positions = new_positions[:len(former_positions)]
    print(new_positions)
    # ------
    for i in range(len(former_positions)):
        x_1 = former_positions[i][0]
        x_2 = new_positions[i][0]
        y_1 = former_positions[i][1]
        y_2 = new_positions[i][1]
        energy = subtract_old_values(energy, last_state, negative_factor, positive_factor, x_1, x_2, y_1, y_2)
        last_state[x_1][y_1].is_filled, last_state[x_2][y_2].is_filled = \
            last_state[x_2][y_2].is_filled, last_state[x_1][y_1].is_filled
        energy = add_new_values(energy, last_state, negative_factor, positive_factor, x_1, x_2, y_1, y_2)
    return former_positions, new_positions, energy


def subtract_old_values(energy, last_state, negative_factor, positive_factor, x_1, x_2, y_1, y_2):
    """
    This function subtracts the values from the fromer energy value, has sense combined with add_new_values
    :param energy:
    :param last_state:
    :param negative_factor:
    :param positive_factor:
    :param x_1:
    :param x_2:
    :param y_1:
    :param y_2:
    :return: change in energy of the universe
    """
    energy -= sum(
        [calculate_energy(point, positive_factor, negative_factor) for point in last_state[x_1][y_1].neighbourhood])
    energy -= sum(
        [calculate_energy(point, positive_factor, negative_factor) for point in last_state[x_2][y_2].neighbourhood])
    return energy


def add_new_values(energy, last_state, negative_factor, positive_factor, x_1, x_2, y_1, y_2):
    """
    This function is similar to the subtract_old_values
    :param energy:
    :param last_state:
    :param negative_factor:
    :param positive_factor:
    :param x_1:
    :param x_2:
    :param y_1:
    :param y_2:
    :return: change in the energy of universe
    """
    energy += sum(
        [calculate_energy(point, positive_factor, negative_factor) for point in last_state[x_1][y_1].neighbourhood])
    energy += sum(
        [calculate_energy(point, positive_factor, negative_factor) for point in last_state[x_2][y_2].neighbourhood])
    return energy


def restore_change(first_position, second_position, state, energy, positive_factor, negative_factor):
    """
    This function restores the changes made to the universe if it's new state is not accepted
    :param first_position: list of former positions of filled cells
    :param second_position: list of former position of empty cells
    :param state:
    :param energy:
    :param positive_factor: factor of the positive neighbourhood
    :param negative_factor: factor of the negative neighbourhood
    :return: energy of the previous state
    """
    for i in range(len(first_position)):
        energy = subtract_old_values(energy, state, negative_factor, positive_factor, first_position[i][0],
                                     second_position[i][0], first_position[i][1], second_position[i][1])
        state[first_position[i][0]][first_position[i][1]].is_filled, state[second_position[i][0]][
            second_position[i][1]].is_filled = \
            state[second_position[i][0]][second_position[i][1]].is_filled, state[first_position[i][0]][
                first_position[i][1]].is_filled
        energy = add_new_values(energy, state, negative_factor, positive_factor, first_position[i][0],
                                second_position[i][0], first_position[i][1], second_position[i][1])
    return energy


def acceptance(old_cost, new_cost, T):
    return exp(min((old_cost - new_cost) / T / 2, 1.0))


def input_factory(size, density, positive_neighbourhood, negative_nieghbourhood, positive_factor, negative_factor):
    """

    :param size: size of the universe
    :param density: density of filled cells
    :param positive_neighbourhood:
    :param negative_nieghbourhood:
    :return: tuple consisting of generated universe and it's energy
    """
    tmp_table = [
        [Point(is_filled=True if density > random() else False) for j in range(0, size)] for i in
        range(0, size)
        ]
    for i in range(size):
        for j in range(size):
            tmp_table[i][j].negative_neighbourhood.extend(
                Neighbourhood(tmp_table, negative_nieghbourhood, (i, j), size))
            tmp_table[i][j].positive_neighbourhood.extend(
                Neighbourhood(tmp_table, positive_neighbourhood, (i, j), size))
            for point in Neighbourhood(tmp_table, negative_nieghbourhood, (i, j), size):
                point.neighbourhood.append(tmp_table[i][j])
            for point in Neighbourhood(tmp_table, positive_neighbourhood, (i, j), size):
                point.neighbourhood.append(tmp_table[i][j])

    energy = table_energy(tmp_table, positive_factor, negative_factor)

    return tmp_table, energy


def table_energy(tmp_table, positive_factor, negative_factor):
    """
    This function calculates the energy of the universe, for sake of the optimisation used only during initialisation
    :param tmp_table: universe state
    :param positive_factor: factor of positive neighbourhood
    :param negative_factor: factor of
    :return: energy of the universe
    """
    energy = 0.0
    for row in tmp_table:
        for point in row:
            energy += calculate_energy(point, positive_factor, negative_factor)
    return energy


def simulated_annealing(start_input, start_energy, positive_factor, negative_factor):
    """
    Function performs simulated_annealing in order to simulate our system
    :param start_input: the input state of the universe
    :param start_energy: the start energy of the universe
    :return: tuple consisting of final state of universe and list of energy of the all states it went through
    """
    T = abs(start_energy / 170.0)
    T_min = T / 50000.0
    alpha_factor = 0.88
    energy_list = list()
    while T > T_min:
        for i in range(130):
            point_a, point_b, energy = create_new_state(start_input, start_energy, positive_factor, negative_factor)
            ap = acceptance(start_energy, energy, T)
            if ap < random():
                energy = restore_change(point_a, point_b, start_input, energy, positive_factor, negative_factor)
            start_energy = energy
            energy_list.append(start_energy)
        T *= alpha_factor
    return start_input, energy_list


def generate_image(input_data, width, height, name):
    img = Image.frombytes('L', (width, height), bytes([0 if p.is_filled else 255 for row in input_data for p in row]))
    img.save(name)


def scale_neighbourhood(input_neighbourhood, scale: int):
    return tuple(set(product(((el[0] * i) for el in input_neighbourhood for i in range(1, scale + 1)),
                             ((el[1] * i) for el in input_neighbourhood for i in range(1, scale + 1)))))


# example of usage
if __name__ == '__main__':

    test_case_data = [
        ('attracting_25_normal', 25, 0.25, (), scale_neighbourhood(((1, 0), (-1, 0), (0, 1), (0, -1)), 1), 5, 5),
        ('phobic_25_normal', 25, 0.20, scale_neighbourhood(((1, 0), (-1, 0), (0, 1), (0, -1)), 1), (), 5, 5),
        ('attracting_50_normal', 50, 0.25, (), scale_neighbourhood(((1, 0), (-1, 0), (0, 1), (0, -1)), 2), 5, 5),
        ('phobic_50_normal', 50, 0.25, scale_neighbourhood(((1, 0), (-1, 0), (0, 1), (0, -1)), 2), (), 5, 5),
        ('attracting_50_low_density', 50, 0.1, (), scale_neighbourhood(((1, 0), (-1, 0), (0, 1), (0, -1)), 2), 5, 5),
        ('atoms_simulation_100', 100, 0.1, scale_neighbourhood(((1, 0), (-1, 0), (0, 1), (0, -1)), 7),
         scale_neighbourhood(((1, 0), (-1, 0), (0, 1), (0, -1)), 2), 5, 100),
        ('atoms_simulation_50', 50, 0.15, scale_neighbourhood(((1, 0), (-1, 0), (0, 1), (0, -1)), 10),
         scale_neighbourhood(((1, 0), (-1, 0), (0, 1), (0, -1)), 2), 5, 100)
    ]
    for i, test_case in enumerate(test_case_data):
        tab, energy = input_factory(*test_case[1:])
        generate_image(tab,test_case[1],test_case[1],'binary_images/before_'+test_case[0]+'.png')
        print(test_case[-2:])
        tab, energy_list = simulated_annealing(tab,energy,*test_case[-2:])
        plt.clf()
        plt.plot(range(0, len(energy_list)), energy_list)
        generate_image(tab,test_case[1],test_case[1],'binary_images/after_'+test_case[0]+'.png')
        plt.savefig('binary_images/cost_'+test_case[0]+'.png')
