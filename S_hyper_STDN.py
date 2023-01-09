import numpy as np
import copy
from population import *
from utils import *

class S_Hyper_STDN:
    def __init__(self, crossover_prob, mutation_prob, population_size, initial_temp = 1, cooling_ratio = 0.9):
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.population_size = population_size
        self.initial_temp = 1
        self.cooling_ratio = 0.9
        self.temp = self.initial_temp
        self.best_indi = None

    def initialize_popualtion(self):
        print("initializing population with number {}...".format(self.population_size))
        self.pops = Population(self.population_size)
        # all the initialized population should be saved
        save_populations(gen_no = -1, pops = self.pops)

    def evaluate_fitness(self, gen_no):
        print("evaluate fintesss")
        # evaluate = Evaluate(self.pops, self.train_data, self.train_label, self.validate_data, self.validate_label, self.number_of_channel, self.epochs, self.batch_size, self.train_data_length, self.validate_data_length)
        # evaluate.parse_population(gen_no)
        # all theinitialized population should be saved
        E = Evaluate()
        E.parse_population(self.pops)
        E.update_population_fitness(self.pops)
        save_populations(gen_no = gen_no, pops = self.pops)
        print(self.pops)

    def recombinate(self, gen_no):
        print("mutation and crossover...")
        offspring_list = []
        for _ in range(int(self.pops.get_pop_size() / 2)):
            p1 = self.tournament_selection()
            p2 = self.tournament_selection()
            # crossover
            offset1, offset2 = self.crossover(p1, p2)
            # mutation
            offset1.mutation()
            offset2.mutation()
            offspring_list.append(offset1)
            offspring_list.append(offset2)
        offspring_pops = Population(0)
        offspring_pops.set_populations(offspring_list)
        self.offspring_pops = offspring_pops
        save_offspring(gen_no, offspring_pops)
        E = Evaluate()
        E.parse_population(offspring_pops)
        E.update_population_fitness(offspring_pops)
        # evaluate these individuals
        # evaluate = Evaluate(self.pops, self.train_data, self.train_label, self.validate_data, self.validate_label, self.number_of_channel, self.epochs, self.batch_size, self.train_data_length, self.validate_data_length)
        # evaluate.parse_population(gen_no)
        # save
        self.pops.pops.extend(offspring_pops.pops)
        save_populations(gen_no = gen_no, pops = self.pops)

    def crossover(self, p1, p2):
        offset1 = copy.deepcopy(p1)
        offset2 = copy.deepcopy(p2)

        chromosome_len = min(len(offset1), len(offset2))
        for i in range(chromosome_len):
            if flip(self.crossover_prob):
                temp = offset1[i]
                offset1[i] = offset2[i]
                offset2[1] = temp

        return offset1, offset2

    def tournament_selection(self):
        ind1_id = np.random.randint(0, self.pops.get_pop_size())
        ind2_id = np.random.randint(0, self.pops.get_pop_size())
        ind1 = self.pops.get_individual_at(ind1_id)
        ind2 = self.pops.get_individual_at(ind2_id)
        winner = self.selection(ind1, ind2)
        return winner

    def selection(self, ind1, ind2):

        # mean_threshold = 0.05
        # complexity_threhold = 100
        # if ind1.mean > ind2.mean:
        #     if ind1.mean - ind2.mean > mean_threshold:
        #         return ind1
        #     else:
        #         if ind2.complxity < (ind1.complxity-complexity_threhold):
        #             return ind2
        #         else:
        #             return ind1
        # else:
        #     if ind2.mean - ind1.mean > mean_threshold:
        #         return ind2
        #     else:
        #         if ind1.complxity < (ind2.complxity-complexity_threhold):
        #             return ind1
        #         else:
        #             return

        if ind1.fitness > ind2.fitness:
            return ind1
        else:
            return ind2

    def environmental_selection(self, gen_no):
        assert(self.pops.get_pop_size() == 2 * self.population_size)
        elitsam = 0.2
        e_count = int(np.floor(self.population_size * elitsam / 2) * 2)
        indi_list = self.pops.pops
        indi_list.sort(key = lambda x:x.fitness, reverse = True)
        elistm_list = indi_list[0 : e_count]

        left_list = indi_list[e_count :]
        np.random.shuffle(left_list)
        np.random.shuffle(left_list)

        for _ in range(self.population_size - e_count):
            i1 = randint(0, len(left_list))
            i2 = randint(0, len(left_list))
            winner = self.selection(left_list[i1], left_list[i2])
            elistm_list.append(winner)

        self.pops.set_populations(elistm_list)
        save_populations(gen_no = gen_no, pops = self.pops)
        np.random.shuffle(self.pops.pops)

    def SA_selection(self, gen_no):
        E = Evaluate()
        E.parse_population(self.offspring_pops)
        E.update_population_fitness(self.offspring_pops)
        for i in range(self.pops.get_pop_size()):
            if self.pops.get_individual_at(i).fitness < self.offspring_pops.get_individual_at(i).fitness:
                self.pops.pops[i] = self.offspring_pops.get_individual_at(i)
            if flip(np.exp(self.pops.get_individual_at(i).fitness - self.offspring_pops.get_individual_at(i)) / self.temp):
                self.pops.pops[i] = self.offspring_pops.get_individual_at(i)
        save_populations(gen_no = gen_no, pops = self.pops)
        np.random.shuffle(self.pops.pops)

    def update_best_individual(self):
        for i in range(self.pops.get_pop_size()):
            indi = self.pops.get_individual_at(i)
            if self.best_indi == None:
                self.best_indi = indi
            elif self.best_indi.fitness < indi.fitness:
                self.best_indi = indi

    def get_best_individual(self):
        return self.best_indi
