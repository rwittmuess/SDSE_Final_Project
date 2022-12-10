'''

    This file runs the Genetic Algorithm


'''

#import packages
import random
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


# import scripts & other classes
import individual


class Genetic_Algoritm():
    def __init__(self, model, param_grid, pop_size = 10, nb_gen = 10):
        self.model = model
        self.pop_size = pop_size
        self.nb_gen = nb_gen
        self.param_grid = param_grid


    def init_pop(self, pop_size) -> list:
        '''
        Initializing population

        consider: potential input for seeding
        '''
        models = []
        for i in range(self.pop_size):
            # Getting random parameters
            rand_params = {}
            for key, value in self.param_grid.items():
                rand_params[key] = random.choice(value)

            # Creating new models
            temp_model = copy.deepcopy(self.model)
            temp_model.set_params(**rand_params)
            models.append(temp_model)

        # scores = []
        # for model in models:
        #     model.fit(Xtrain, ytrain)
        #     scores.append(model.score(Xtest, ytest))

    

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
        parents = []
        fitness_list = []

        for idx, parent in enumerate(population):
            fitness_list[idx] = parent.calculate_fitness()

        for parent_nr in range(number_parents):
            max_fitness_idx = np.where(
                population[parent_nr].calculate_fitness() == np.max(fitness_list))[0][0]

            # add to new parents
            parents[parent_nr] = population[max_fitness_idx]
            # remove from old population
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
        -Generating new Individuals
            -Crossover
            -Mutation

        gets:       set/list/... of individuals
        returns:    set/list/... of individuals with mutated genom
        '''


if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    rng_seed = 2434

    df = pd.read_csv('pulsar_data.csv')
    feature_names = df.columns.values[:-1]  # removing the traget (last one)
    numnulls_in_feature = dict().fromkeys(feature_names, 0)
    for feature in feature_names:
        numnulls_in_feature[feature] = df[feature].isna().sum()
    df = df.dropna(axis=1)
    feature_names = df.columns.values[:-1]

    Xtrain, Xtest, ytrain, ytest = train_test_split(df.iloc[:,:-1],
                                                    df['target_class'], 
                                                    test_size=0.1, 
                                                    random_state=rng_seed )
    
    print("Data ready to train")

    model = LogisticRegression(solver='liblinear', random_state=rng_seed)

    param_grid = {
        'model__penalty' : ['l1','l2'],
        'model__C' : np.logspace(-20, 10, 100),
        'model__fit_intercept': [True, False],
        'model__intercept_scaling' : np.logspace(0, 1, 100)}

    gen = Genetic_Algoritm(model,param_grid)

    # gen.fit(Xtrain, ytrain)
