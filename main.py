from hyper_STDN import Hyper_STDN
from utils import *
from STDN_control import get_final_output
import gc

def begin_hyper_evolve(crossover_prob = 0.8, mutation_prob = 0.2, population_size = 10, total_generation_number = 10):
    gc.collect()
    H = Hyper_STDN(crossover_prob = crossover_prob, mutation_prob = mutation_prob, population_size = population_size)
    H.initialize_popualtion()
    H.evaluate_fitness(0)
    for cur_gen_no in range(total_generation_number):
        gc.collect()
        print('The {}/{} generation'.format(cur_gen_no + 1, total_generation_number))
        H.recombinate(cur_gen_no + 1)
        H.environmental_selection(cur_gen_no + 1)
        H.update_best_individual()

    final_indi = H.get_best_individual()

    print('-' * 10)
    print('final_indi.fitness:', final_indi.fitness)
    print('final_indi.arg_list:', final_indi.arg_list)
    print("att_lstm_num:", final_indi.arg_list[0])
    print("long_term_lstm_seq_len:", final_indi.arg_list[1])
    print("short_term_lstm_seq_len:", final_indi.arg_list[2])
    print("nbhd_size:", final_indi.arg_list[3])
    print("cnn_nbhd_size:", final_indi.arg_list[4])
    (prmse, pmape), (drmse, dmape) = get_final_output(final_indi)
    print(
        "Test on model {0}:\npickup rmse = {1}, pickup mape = {2}%\ndropoff rmse = {3}, dropoff mape = {4}%".format(
            'stdn', prmse, pmape * 100, drmse, dmape * 100))
    print('-' * 10)
    
# def begin_S_hyper_evolve(crossover_prob = 0.8, mutation_prob = 0.2, population_size = 10, total_generation_number = 10, initial_temp = 1, cooling_ratio = 0.9):
#     S = S_Hyper_STDN(crossover_prob, mutation_prob, population_size, initial_temp, cooling_ratio)
#     S.initialize_popualtion()
#     S.evaluate_fitness(0)
#     for cur_gen_no in range(total_generation_number):
#         print('The {}/{} generation'.format(cur_gen_no + 1, total_generation_number))
#         S.recombinate(cur_gen_no + 1)
#         # S.environmental_selection(cur_gen_no + 1)
#         S.SA_selection(cur_gen_no + 1)

if __name__ == '__main__':

    print('gc.isenabled():', gc.isenabled())
    begin_hyper_evolve(crossover_prob = 0.8, mutation_prob = 0.2, population_size = 6, total_generation_number = 5)
