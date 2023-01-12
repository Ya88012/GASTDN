import numpy as np
import os
from STDN_control import get_fitness
import gc

def parse_population(pop):
    for i in range(pop.get_pop_size()):
        indi = pop.get_individual_at(i)
        indi.arg_list = parse_individual(indi)

def parse_individual(indi):
    
    att_lstm_num_bit_num = int(np.ceil(np.log2(indi.att_lstm_num_range[1])))
    long_term_lstm_seq_len_bit_num = int(np.ceil(np.log2(indi.long_term_lstm_seq_len_range[1])))
    short_term_lstm_seq_len_bit_num = int(np.ceil(np.log2(indi.short_term_lstm_seq_len_range[1])))
    nbhd_size_bit_num = int(np.ceil(np.log2(indi.nbhd_size_range[1])))
    cnn_nbhd_size = int(np.ceil(np.log2(indi.cnn_nbhd_size_range[1])))

    bit_num_list = [0, att_lstm_num_bit_num, long_term_lstm_seq_len_bit_num, short_term_lstm_seq_len_bit_num, nbhd_size_bit_num, cnn_nbhd_size]
    bin_str_list = []

    for i in range(1, len(bit_num_list)):
        start = sum(bit_num_list[: i])
        end = sum(bit_num_list[: i + 1])
        temp_str = ''
        for j in range(start, end):
            if indi.gene_list[j].unit == True:
                temp_str += '1'
            else:
                temp_str += '0'
        bin_str_list.append(temp_str)
    parameter_list = list(map(lambda i: int(i, 2), bin_str_list))
    return parameter_list

def update_population_fitness(pop):
    for i in range(pop.get_pop_size()):
        indi = pop.get_individual_at(i)
        update_individual_fitness(indi)

def update_individual_fitness(indi):
    gc.collect()
    (prmse, pmape), (drmse, dmape) = get_fitness(indi)
    indi.fitness = 100 - (pmape + dmape) / 2
    # indi.fitness = np.random.randint(low = 0, high = 100)
    print('indi.arg_list:', indi.arg_list)
    print('indi_fitness:', indi.fitness)