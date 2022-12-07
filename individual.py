import random

from genetic_algorithm import Genetic_Algoritm


class Individual:
    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.fitness = 0

    def calculate_fitness(self):
        # calculates the fitness-score for this individual
        score = '''????????_______self.chromosome'''
        return score

    def mutate(self, mutation_rate):
        for i in range(len(self.chromosome)):
		    # check for a mutation
            if (random.uniform(0,1) < mutation_rate):
                self.chromosome[i] = '''????????????????????????'''
        return self.chromosome
