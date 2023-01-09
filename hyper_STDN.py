import numpy as np
import copy
from population import *
from utils import *
from evaluate import *

class Hyper_STDN:
    def __init__(self, crossover_prob, mutation_prob, population_size):
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.population_size = population_size
        self.best_indi = None

    def initialize_popualtion(self):
        print("initializing population with number {}...".format(self.population_size))
        self.pops = Population(self.population_size)
        # all the initialized population should be saved
        save_populations(gen_no = -1, pops = self.pops)

    def evaluate_fitness(self, gen_no):
        print("evaluate fintesss:", gen_no)

        # evaluate = Evaluate(self.pops, self.train_data, self.train_label, self.validate_data, self.validate_label, self.number_of_channel, self.epochs, self.batch_size, self.train_data_length, self.validate_data_length)
        # evaluate.parse_population(gen_no)
        # all theinitialized population should be saved

        parse_population(self.pops)
        update_population_fitness(self.pops)
        save_populations(gen_no = gen_no, pops = self.pops)
        print('gen_no:', gen_no)
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

            # here sure long term lstm be odd

            offset1.check_arg_condition()
            offset2.check_arg_condition()

            offspring_list.append(offset1)
            offspring_list.append(offset2)
        offspring_pops = Population(0)
        offspring_pops.set_population(offspring_list)
        self.offspring_pops = offspring_pops
        save_offspring(gen_no, offspring_pops)

        parse_population(offspring_pops)
        update_population_fitness(offspring_pops)
        
        # evaluate these individuals
        # evaluate = Evaluate(self.pops, self.train_data, self.train_label, self.validate_data, self.validate_label, self.number_of_channel, self.epochs, self.batch_size, self.train_data_length, self.validate_data_length)
        # evaluate.parse_population(gen_no)
        # save

        self.pops.pops.extend(offspring_pops.pops)
        save_populations(gen_no = gen_no, pops = self.pops)

    def crossover(self, p1, p2):
        offset1 = copy.deepcopy(p1)
        offset2 = copy.deepcopy(p2)

        chromosome_len = min(len(offset1.gene_list), len(offset2.gene_list))
        for i in range(chromosome_len):
            if flip(self.crossover_prob):

                temp = offset1.gene_list[i]
                offset1.gene_list[i] = offset2.gene_list[i]
                offset2.gene_list[i] = temp

        offset1.update_arg_list()
        offset2.update_arg_list()

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

        self.pops.set_population(elistm_list)
        save_populations(gen_no = gen_no, pops = self.pops)
        np.random.shuffle(self.pops.pops)
        print('In environmental_selection:')
        print('gen_no:', gen_no)
        print(self.pops)

    def update_best_individual(self):
        for i in range(self.pops.get_pop_size()):
            indi = self.pops.get_individual_at(i)
            if self.best_indi == None:
                self.best_indi = indi
            elif indi.fitness > self.best_indi.fitness:
                self.best_indi = indi

    def get_best_individual(self):
        return self.best_indi

    # def check_arg_bound(self, indi):

    #     att_lstm_num_bit_num = int(np.ceil(np.log2(indi.att_lstm_num_range[1])))
    #     long_term_lstm_seq_len_bit_num = int(np.ceil(np.log2(indi.long_term_lstm_seq_len_range[1])))
    #     short_term_lstm_seq_len_bit_num = int(np.ceil(np.log2(indi.short_term_lstm_seq_len_range[1])))
    #     nbhd_size_bit_num = int(np.ceil(np.log2(indi.nbhd_size_range[1])))
    #     cnn_nbhd_size = int(np.ceil(np.log2(indi.cnn_nbhd_size_range[1])))

    #     bit_num_list = [0, att_lstm_num_bit_num, long_term_lstm_seq_len_bit_num, short_term_lstm_seq_len_bit_num, nbhd_size_bit_num, cnn_nbhd_size]

    #     # check whether long term lstm seq len is odd 
    #     if indi.gene_list[att_lstm_num_bit_num + long_term_lstm_seq_len_bit_num - 1].unit != True:
    #         indi.gene_list[att_lstm_num_bit_num + long_term_lstm_seq_len_bit_num - 1].unit = True
    #         indi.arg_list[1] += 1
    #         print('set_long_term_lstm_seq_len correct!')

    #     start = sum(bit_num_list[: 1])
    #     end = sum(bit_num_list[: 2])
    #     att_lstm_num_gene = indi.gene_list[start : end]
    #     att_check_flag = True
    #     for temp in att_lstm_num_gene:
    #         if temp.unit == True:
    #             att_check_flag = False
    #     # That means this parameter(att_lstm_num) is zero
    #     if att_check_flag == True:
    #         for temp in range(start, end):
    #             if temp != end - 1:
    #                 indi.gene_list[temp].unit = False
    #             else:
    #                 indi.gene_list[temp].unit = True
        
    #     start = sum(bit_num_list[: 2])
    #     end = sum(bit_num_list[: 3])
    #     long_term_lstm_seq_len_gene = indi.gene_list[start : end]
    #     long_term_lstm_seq_len_check_flag = True
    #     for temp in long_term_lstm_seq_len_gene:
    #         if temp.unit == True:
    #             long_term_lstm_seq_len_check_flag = False
    #     # That means this parameter(long_term_lstm_seq_len) is zero
    #     if long_term_lstm_seq_len_check_flag == True:
    #         for temp in range(start, end):
    #             if temp != end - 1:
    #                 indi.gene_list[temp].unit = False
    #             else:
    #                 indi.gene_list[temp].unit = True
        
    #     start = sum(bit_num_list[: 3])
    #     end = sum(bit_num_list[: 4])
    #     short_term_lstm_seq_len_gene = indi.gene_list[start : end]
    #     short_term_lstm_seq_len_check_flag = True
    #     for temp in short_term_lstm_seq_len_gene:
    #         if temp.unit == True:
    #             short_term_lstm_seq_len_check_flag = False
    #     # That means this parameter(short_term_lstm_seq_len) is zero
    #     if short_term_lstm_seq_len_check_flag == True:
    #         for temp in range(start, end):
    #             if temp != end - 1:
    #                 indi.gene_list[temp].unit = False
    #             else:
    #                 indi.gene_list[temp].unit = True

