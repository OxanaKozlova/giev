import math
import random
from matplotlib import pyplot as plt
import numpy as np
import time

class Item:
    def __init__(self, value, min = 0, max = 0):
        self.min = min
        self.max = max
        self.value = 0
        self.set_value(value)
        self.fitnes = pow(self.value, 2)
        self.fitness_prob = 0

    # def set_fitness(self, fitness):
    #     self.fitness = fitness

    def set_value(self, value):
        if (value + self.value) < self.min:
            self.value = self.min
        elif (value + self.value) > self.max:
            self.value = self.max
        else:
            self.value = value

    def set_fitness_prob(self, fitness_prob):
        self.fitness_prob = fitness_prob


class Population:
    def __init__(self, population_size, min_value, max_value, parents_count, mutation, steps):
        self.population_size = population_size
        self.min_value = min_value
        self.max_value = max_value
        self.items =  [Item(random.randint(min_value, max_value), min_value, max_value) for _ in range(0, population_size)]
        self.parents_count = parents_count
        self.mutation = mutation
        self.parents = []
        self.couples = []
        self.new_population = []
        self.steps = steps

    # def calculate_fitness(self):
    #     for i in self.items:
    #         i.set_fitness(pow(i.value, 2))

    def calculate_fitness_prob(self):
        prob_a = random.uniform(1, 2)
        prob_b = 2 - prob_a

        for key, item in enumerate(self.items):
            item.set_fitness_prob((1 / self.population_size) * (prob_a - (prob_a - prob_b) * (key - 1) / (self.population_size - 1)))

    def set_parents(self):
        sorted_by_fitness_prob = sorted(self.items, key=lambda item: item.fitness_prob, reverse=True)
        self.parents = sorted_by_fitness_prob[:self.parents_count]

    def create_couples(self):
        for key, parent in enumerate(self.parents):
            available_parents_index = [i for i in range(self.parents_count)]
            available_parents_index.remove(key)
            rand_parent = self.parents[random.choice(available_parents_index)]
            self.couples.append({'first_parent': parent, 'second_parent': rand_parent})

    def generate_new_population(self):
        for couple in self.couples:
            rand_param = random.uniform(0, 1)
            self.new_population.append(Item(couple['first_parent'].value + rand_param * (couple['second_parent'].value - couple['first_parent'].value), self.min_value, self.max_value))

    def mutate(self):
        for item in self.new_population:
            isMutate = random.uniform(0, 1)
            if self.mutation < isMutate:
                isMinus = random.randint(0, 1)
                mutationValue =  0.5 * random.uniform(0, 1) * ( -1 if isMinus else 1)
                item.set_value(item.value + mutationValue)

    def update_population(self):
        self.items = self.items[:self.population_size - self.parents_count] + self.new_population

    def run(self):
        for step in range(self.steps):
            # self.calculate_fitness()
            self.items = sorted(self.items, key=lambda item: item.fitnes, reverse=True)
            self.calculate_fitness_prob()

            self.items = sorted(self.items, key=lambda item: item.fitness_prob, reverse=True)
            self.parents = self.items[:self.parents_count]

            self.create_couples()
            self.generate_new_population()
            self.mutate()
            self.update_population()

        self.items = sorted(self.items, key=lambda item: item.fitnes, reverse=True)
        self.calculate_fitness_prob()
        self.items = sorted(self.items, key=lambda item: item.fitness_prob, reverse=True)
        print('res', self.items[0].value)


population_size = 50
min_value = 6
max_value = 10
parents_count = 30
mutation = 0.5
steps = 100

population = Population(population_size, min_value, max_value, parents_count, mutation, steps)
population.run()
