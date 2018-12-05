import math
import random
from matplotlib import pyplot as plt
import time
from scipy.spatial.distance import cdist


class Config:
    POPULATION_COUNT = 100
    PARENTS_COUNT = 40
    MUTATION_PROB = 0.01
    STEPS = 100
    NUM_NODES = 15
    
class Edge:
    def __init__(self, node_1, node_2, weight):
        self.node_1 = node_1
        self.node_2 = node_2
        self.weight = weight

class Site:
    def __init__(self, nodes):
        self.nodes = nodes
        self.edges = [
            [None] * Config.NUM_NODES for _ in range(Config.NUM_NODES)]
        for i in range(Config.NUM_NODES):
            for j in range(i + 1, Config.NUM_NODES):
                self.edges[i][j] = self.edges[j][i] = Edge(i, j, math.sqrt(
                    pow(self.nodes[i][0] - self.nodes[j][0], 2.0) + pow(self.nodes[i][1] - self.nodes[j][1], 2.0)))

    def get_tour(self):
        node_indexes = list(range(Config.NUM_NODES))
        random.shuffle(node_indexes)
        return node_indexes

    def get_distance(self, tour):
        distance = 0.0
        for i in range(Config.NUM_NODES):
            distance += self.edges[tour[i]][tour[(i + 1) % Config.NUM_NODES]].weight
        return distance

class Chromosome:
    def __init__(self, site, tour=None):
        self.site = site
        self.select_prob = 0
        if tour == None:
            self.tour = self.site.get_tour()
        else:
            self.tour = tour
        self.fitness_value = self.calculate_fitness_function()

    def calculate_fitness_function(self):
        return self.site.get_distance(self.tour)

class Population:
    def __init__(self, site):
        self.chromosomes = [Chromosome(site) for _ in range(Config.POPULATION_COUNT)]
        self.parents = []
        self.parent_combinations = []
        self.new_chromosomes = []
        self.site = site

    def sort_by_fitness_value(self):
        self.chromosomes = sorted(self.chromosomes, key=lambda chromosome: chromosome.fitness_value)

    def ranging(self):
        rand_a = random.uniform(1, 2)
        rand_b = 2 - rand_a
        for key, chromosome in enumerate(self.chromosomes):
            chromosome.select_prob = (1 / Config.POPULATION_COUNT) * ((rand_a - ((rand_a - rand_b) * (key - 1) / (Config.POPULATION_COUNT - 1))))

    def sort_by_select_prob(self):
        self.chromosomes = sorted(self.chromosomes, key=lambda chromosome: chromosome.select_prob, reverse=True)

    def choose_parents_combination(self):
        for key, parent in enumerate(self.parents):
            available_parents_index = [i for i in range(Config.PARENTS_COUNT)]
            available_parents_index.remove(key)
            rand_parent = self.parents[random.choice(available_parents_index)]
            self.parent_combinations.append({'first_parent': parent, 'second_parent': rand_parent})

    def is_valid_tour(self, tour):
        return len(tour) == len(list(set(tour)))

    def generate_new_chromosomes(self):
        for parent_combination in self.parent_combinations:
            crossing_point = random.randint(1, (Config.NUM_NODES - 1))
            first_part_of_chr = parent_combination['first_parent'].tour[:crossing_point]
            second_part_of_chr = parent_combination['second_parent'].tour[crossing_point:]
            new_tour = first_part_of_chr + second_part_of_chr
            nodes = list(range(Config.NUM_NODES))
            if not self.is_valid_tour(new_tour):
                not_visited_nodes =  list(set(nodes) - set(first_part_of_chr))
                random.shuffle(not_visited_nodes)
                new_tour = first_part_of_chr + not_visited_nodes
            self.new_chromosomes.append(Chromosome(self.site, tour=new_tour))

    def mutate(self):
        for chromosome in self.new_chromosomes:
            is_mutate = random.uniform(0, 1)
            if (is_mutate < Config.MUTATION_PROB):
                node_indexes = random.sample(range(0, Config.NUM_NODES - 1), 2)
                node = chromosome.tour.pop(node_indexes[0])
                chromosome.tour.insert(node_indexes[1], node)

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


def plot(nodes, best_tour, annotation_size=8, dpi=120):
    x = [nodes[i][0] for i in best_tour]
    labels = range(1, len(nodes) + 1)
    x.append(x[0])
    y = [nodes[i][1] for i in best_tour]
    y.append(y[0])
    plt.plot(x, y, linewidth=1)
    plt.scatter(x, y, s=math.pi * 4)
    plt.title('Best tour')
    for i in best_tour:
        plt.annotate(labels[i], nodes[i], size=annotation_size)

    name = 'plots/best_tour.png'
    plt.savefig(name, dpi=dpi)
    plt.show()
    plt.gcf().clear()

nodes = [(random.uniform(-1500, 1500), random.uniform(-1500, 1500))
         for _ in range(0, Config.NUM_NODES)]

population = Population(Site(nodes))
result = population.run()
plot(nodes, result.tour)
