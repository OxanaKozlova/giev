import math
import random
from matplotlib import pyplot as plt
import numpy as np
import time
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show

def calculate_digits_count(MIN, max, precision):
    parts_count = (max - MIN) * pow(10, precision)
    return math.ceil(math.log(parts_count,2))

def convert_to_float(gen):
    int_gen = 0
    for i in range((len(gen) - 1), 0, -1):
        int_gen += gen[i] * pow(2, i)
    return Config.MIN + int_gen * (Config.MAX - Config.MIN) / (pow(2, len(gen)) - 1)

class Config:
    MIN = -10
    MAX = 10
    PRECISION = 2
    POPULATION_COUNT = 50
    PARENTS_COUNT = 30
    MUTATION_PROB = 0.01
    STEPS = 50
    GENES_COUNT = 2
    FITNESS_FUNCTION = lambda x : pow(x[0], 2)
    GEN_SIZE = calculate_digits_count(MIN, MAX, PARENTS_COUNT)

class Gen:
    def __init__(self, size, value=None):
        if (value == None):
            self.value = [random.randrange(0,2) for _ in range(size)]
        else: self.value = value

class Chromosome:
    def __init__(self, genes=None):
        if (genes == None):
            self.genes = [Gen(Config.GEN_SIZE) for _ in range(Config.GENES_COUNT)]
        else: self.genes = genes
        self.fitness_value = self.calculate_fitness_function()
        self.select_prob = 0

    def calculate_fitness_function(self):
        float_gens = []
        for gen in self.genes:
            float_gens.append(convert_to_float(gen.value))
        return Config.FITNESS_FUNCTION(float_gens)

class Population:
    def __init__(self):
        self.chromosomes = [Chromosome() for _ in range(Config.POPULATION_COUNT)]
        self.parents = []
        self.parent_combinations = []
        self.new_chromosomes = []
        self.best_chromosome = None

    def sort_by_fitness_value(self):
        self.chromosomes = sorted(self.chromosomes, key=lambda chromosome: chromosome.fitness_value)

    def ranging(self):
        rand_a = random.uniform(1, 2)
        rand_b = 2 - rand_a
        for key, chromosome in enumerate(self.chromosomes):
            chromosome.select_prob = (1 / Config.POPULATION_COUNT) * (rand_a - (rand_a - rand_b) * (key - 1) / (Config.POPULATION_COUNT - 1))

    def sort_by_select_prob(self):
        self.chromosomes = sorted(self.chromosomes, key=lambda chromosome: chromosome.select_prob, reverse=True)

    def choose_parents_combination(self):
        for key, parent in enumerate(self.parents):
            available_parents_index = [i for i in range(Config.PARENTS_COUNT)]
            available_parents_index.remove(key)
            rand_parent = self.parents[random.choice(available_parents_index)]
            self.parent_combinations.append({'first_parent': parent, 'second_parent': rand_parent})

    def generate_chromosome(self, first_parent_genes, second_parent_genes, crossing_points):
        genes = []
        for i in range(Config.GENES_COUNT):
            genes.append(Gen(Config.GEN_SIZE, first_parent_genes[i].value[:crossing_points[i]] + first_parent_genes[i].value[crossing_points[i]:]))
        self.new_chromosomes.append(Chromosome(genes))

    def generate_new_chromosomes(self):
        for parent_combination in self.parent_combinations:
            crossing_points = [random.randint(0, (Config.GEN_SIZE - 1)) for _ in range(Config.GENES_COUNT)]
            self.generate_chromosome(parent_combination['first_parent'].genes, parent_combination['second_parent'].genes, crossing_points)

    def mutate(self):
        for chromosome in self.new_chromosomes:
            for gen in chromosome.genes:
                is_mutate = random.uniform(0, 1)
                if (is_mutate < Config.MUTATION_PROB):
                    index = random.randint(0, (Config.GEN_SIZE - 1));
                    gen.value[index] = 1 if gen.value[index] == 0 else 0

    def update_population(self):
        self.chromosomes = self.new_chromosomes + self.chromosomes[:Config.POPULATION_COUNT - Config.PARENTS_COUNT]

    def find_best_chromosome(self):
        self.sort_by_fitness_value()
        self.ranging()
        self.sort_by_select_prob()
        return self.chromosomes[0]

    def run(self):
        for step in range(Config.STEPS):
            self.sort_by_fitness_value()
            self.ranging()
            self.sort_by_select_prob()
            self.parents = self.chromosomes[:Config.PARENTS_COUNT]
            self.choose_parents_combination()
            self.generate_new_chromosomes()
            self.mutate()
            self.update_population()
        return self.find_best_chromosome()


def plot_fitness_function_two_var():
    x = range(Config.MIN, Config.MAX)
    y = range(Config.MIN, Config.MAX)
    X,Y = meshgrid(x, y)
    Z = Config.FITNESS_FUNCTION([X, Y])
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                          cmap=cm.RdBu,linewidth=0, antialiased=False)

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

def plot_fitness_function_one_var():
    precision = 1 / pow(10, Config.PRECISION)
    x = np.arange(Config.MIN, Config.MAX, precision)
    y = [ Config.FITNESS_FUNCTION([i]) for i in x]
    plt.plot(x,y)
    plt.show()

population = Population()
result = population.run()

for gen in result.genes:
    print(convert_to_float(gen.value))

if Config.GENES_COUNT == 1:
    plot_fitness_function_one_var()
if Config.GENES_COUNT == 2:
    plot_fitness_function_two_var()
