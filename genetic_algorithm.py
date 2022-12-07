'''

    This file runs the Genetic Algorithm


'''

#import packages
import numpy as np



# import scripts & other classes
import individual


class Genetic_Algoritm():


    def initialize_population(self, population_size, ):
        '''
        Initializing population

        consider: potential input for seeding
        '''



    # def objective_func():
    #     '''
    #     Fitness funtion:
    #         gets: individual 
    #         returns: fitness score    

    #     To consider:
    #      * binary  
    #      * non-binary 
    #      * categorial
    #     '''

    def selecting_parents(population, number_parents):
        '''
        Selecting Parents
        '''
        parents         = []
        fitness_list    = []

        for idx, parent in enumerate(population):
            fitness_list[idx] = parent.calculate_fitness()

        for parent_nr in range(number_parents):
            max_fitness_idx = np.where(population[parent_nr].calculate_fitness() == np.max(fitness_list))[0][0]
            
            #add to new parents
            parents[parent_nr] = population[max_fitness_idx]
            #remove from old population
            population.pop(max_fitness_idx)
            fitness_list.pop(max_fitness_idx)

        return parents

    def generate_new_individuals():
        '''
        Generate new individuals for "filling up" the population after selection
        '''



    def mating():
        '''
        Mating
        -Genertating new Individuals
            -Crossover
            -Mutation
        
        gets:       set/list/... of individuals
        returns:    set/list/... of individuals with mutated genom
        '''
    




