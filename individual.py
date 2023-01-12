import numpy as np
from utils import *

class Gene:
    def __init__(self, unit_value):
        self.unit = unit_value

class Individual:
    def __init__(self, crossover_prob = 0.8, mutation_prob = 0.2):
        self.gene_list = []
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.fitness = -1.0
        self.arg_list = []

        self.att_lstm_num_range = [1, 4]
        self.long_term_lstm_seq_len_range = [1, 8]
        self.short_term_lstm_seq_len_range = [1, 8]
        self.nbhd_size_range = [0, 4]
        self.cnn_nbhd_size_range = [0, 4]

    def initialize(self):
        init_att_lstm_num = np.random.randint(self.att_lstm_num_range[0], self.att_lstm_num_range[1])
        init_long_term_lstm_seq_len = np.random.randint(self.long_term_lstm_seq_len_range[0], self.long_term_lstm_seq_len_range[1])
        init_short_term_lstm_seq_len = np.random.randint(self.short_term_lstm_seq_len_range[0], self.short_term_lstm_seq_len_range[1])
        init_nbhd_size = np.random.randint(self.nbhd_size_range[0], self.nbhd_size_range[1])
        init_cnn_nbhd_size = np.random.randint(self.cnn_nbhd_size_range[0], self.cnn_nbhd_size_range[1])

        while init_long_term_lstm_seq_len % 2 != 1:
            init_long_term_lstm_seq_len = np.random.randint(self.long_term_lstm_seq_len_range[0], self.long_term_lstm_seq_len_range[1])

        t_att_lstm_num_list = [Gene(bool(int(t_index))) for t_index in ('{0:0' + str(int(np.ceil(np.log2(self.att_lstm_num_range[1])))) + 'b}').format(init_att_lstm_num)]
        t_long_term_lstm_seq_len_list = [Gene(bool(int(t_index))) for t_index in ('{0:0' + str(int(np.ceil(np.log2(self.long_term_lstm_seq_len_range[1])))) + 'b}').format(init_long_term_lstm_seq_len)]
        t_short_term_lstm_seq_len_list = [Gene(bool(int(t_index))) for t_index in ('{0:0' + str(int(np.ceil(np.log2(self.short_term_lstm_seq_len_range[1])))) + 'b}').format(init_short_term_lstm_seq_len)]
        t_nbhd_size_list = [Gene(bool(int(t_index))) for t_index in ('{0:0' + str(int(np.ceil(np.log2(self.nbhd_size_range[1])))) + 'b}').format(init_nbhd_size)]
        t_cnn_nbhd_size_range = [Gene(bool(int(t_index))) for t_index in ('{0:0' + str(int(np.ceil(np.log2(self.cnn_nbhd_size_range[1])))) + 'b}').format(init_cnn_nbhd_size)]

        self.gene_list += (t_att_lstm_num_list + t_long_term_lstm_seq_len_list + t_short_term_lstm_seq_len_list + t_nbhd_size_list + t_cnn_nbhd_size_range)
        self.arg_list = [init_att_lstm_num, init_long_term_lstm_seq_len, init_short_term_lstm_seq_len, init_nbhd_size, init_cnn_nbhd_size]

        print("In Individual.py!!!!!")
        print("init_att_lstm_num:", init_att_lstm_num)
        print("init_long_term_lstm_seq_len:", init_long_term_lstm_seq_len)
        print("init_short_term_lstm_seq_len:", init_short_term_lstm_seq_len)
        print("init_nbhd_size:", init_nbhd_size)
        print("init_cnn_nbhd_size:", init_cnn_nbhd_size)
        print("self.arg_list:", self.arg_list)

    def initialize_spec(self, temp_list):
        init_att_lstm_num = temp_list[0]
        init_long_term_lstm_seq_len = temp_list[1]
        init_short_term_lstm_seq_len = temp_list[2]
        init_nbhd_size = temp_list[3]
        init_cnn_nbhd_size = temp_list[4]

        while init_long_term_lstm_seq_len % 2 != 1:
            init_long_term_lstm_seq_len = np.random.randint(self.long_term_lstm_seq_len_range[0], self.long_term_lstm_seq_len_range[1])

        t_att_lstm_num_list = [Gene(bool(int(t_index))) for t_index in ('{0:0' + str(int(np.ceil(np.log2(self.att_lstm_num_range[1])))) + 'b}').format(init_att_lstm_num)]
        t_long_term_lstm_seq_len_list = [Gene(bool(int(t_index))) for t_index in ('{0:0' + str(int(np.ceil(np.log2(self.long_term_lstm_seq_len_range[1])))) + 'b}').format(init_long_term_lstm_seq_len)]
        t_short_term_lstm_seq_len_list = [Gene(bool(int(t_index))) for t_index in ('{0:0' + str(int(np.ceil(np.log2(self.short_term_lstm_seq_len_range[1])))) + 'b}').format(init_short_term_lstm_seq_len)]
        t_nbhd_size_list = [Gene(bool(int(t_index))) for t_index in ('{0:0' + str(int(np.ceil(np.log2(self.nbhd_size_range[1])))) + 'b}').format(init_nbhd_size)]
        t_cnn_nbhd_size_range = [Gene(bool(int(t_index))) for t_index in ('{0:0' + str(int(np.ceil(np.log2(self.cnn_nbhd_size_range[1])))) + 'b}').format(init_cnn_nbhd_size)]

        self.gene_list += (t_att_lstm_num_list + t_long_term_lstm_seq_len_list + t_short_term_lstm_seq_len_list + t_nbhd_size_list + t_cnn_nbhd_size_range)
        self.arg_list = [init_att_lstm_num, init_long_term_lstm_seq_len, init_short_term_lstm_seq_len, init_nbhd_size, init_cnn_nbhd_size]

        print("In Individual.py!!!!!")
        print("init_att_lstm_num:", init_att_lstm_num)
        print("init_long_term_lstm_seq_len:", init_long_term_lstm_seq_len)
        print("init_short_term_lstm_seq_len:", init_short_term_lstm_seq_len)
        print("init_nbhd_size:", init_nbhd_size)
        print("init_cnn_nbhd_size:", init_cnn_nbhd_size)
        print("self.arg_list:", self.arg_list)

    def mutation(self):
        for i in range(len(self.gene_list)):
            if flip(self.mutation_prob):
                self.gene_list[i].unit = (not self.gene_list[i].unit)

    def __str__(self):
        str_ = []

        att_lstm_num_bit_num = int(np.ceil(np.log2(self.att_lstm_num_range[1])))
        long_term_lstm_seq_len_bit_num = int(np.ceil(np.log2(self.long_term_lstm_seq_len_range[1])))
        short_term_lstm_seq_len_bit_num = int(np.ceil(np.log2(self.short_term_lstm_seq_len_range[1])))
        nbhd_size_bit_num = int(np.ceil(np.log2(self.nbhd_size_range[1])))
        cnn_nbhd_size = int(np.ceil(np.log2(self.cnn_nbhd_size_range[1])))

        bit_num_list = [0, att_lstm_num_bit_num, long_term_lstm_seq_len_bit_num, short_term_lstm_seq_len_bit_num, nbhd_size_bit_num, cnn_nbhd_size]
        name_list = ['att_lstm_num', 'long_term_lstm_seq_len', 'short_term_lstm_seq_len', 'nbhd_size', 'cnn_nbhd_size']
        # bin_str_list = []

        for i in range(1, len(bit_num_list)):
            start = sum(bit_num_list[: i])
            end = sum(bit_num_list[: i + 1])
            temp_str = ''
            for j in range(start, end):
                if self.gene_list[j].unit == True:
                    temp_str += '1'
                else:
                    temp_str += '0'
            # bin_str_list.append(temp_str)
            str_.append(name_list[i - 1] + ': ' + temp_str)
        # parameter_list = list(map(lambda i: int(i, 2), bin_str_list))
        # return parameter_list
        return ', '.join(str_)
    
    def update_arg_list(self):
        att_lstm_num_bit_num = int(np.ceil(np.log2(self.att_lstm_num_range[1])))
        long_term_lstm_seq_len_bit_num = int(np.ceil(np.log2(self.long_term_lstm_seq_len_range[1])))
        short_term_lstm_seq_len_bit_num = int(np.ceil(np.log2(self.short_term_lstm_seq_len_range[1])))
        nbhd_size_bit_num = int(np.ceil(np.log2(self.nbhd_size_range[1])))
        cnn_nbhd_size = int(np.ceil(np.log2(self.cnn_nbhd_size_range[1])))

        bit_num_list = [0, att_lstm_num_bit_num, long_term_lstm_seq_len_bit_num, short_term_lstm_seq_len_bit_num, nbhd_size_bit_num, cnn_nbhd_size]
        bin_str_list = []

        for i in range(1, len(bit_num_list)):
            start = sum(bit_num_list[: i])
            end = sum(bit_num_list[: i + 1])
            temp_str = ''
            for j in range(start, end):
                if self.gene_list[j].unit == True:
                    temp_str += '1'
                else:
                    temp_str += '0'
            bin_str_list.append(temp_str)
        parameter_list = list(map(lambda i: int(i, 2), bin_str_list))
        self.arg_list = parameter_list

    def check_arg_condition(self):

        att_lstm_num_bit_num = int(np.ceil(np.log2(self.att_lstm_num_range[1])))
        long_term_lstm_seq_len_bit_num = int(np.ceil(np.log2(self.long_term_lstm_seq_len_range[1])))
        short_term_lstm_seq_len_bit_num = int(np.ceil(np.log2(self.short_term_lstm_seq_len_range[1])))
        nbhd_size_bit_num = int(np.ceil(np.log2(self.nbhd_size_range[1])))
        cnn_nbhd_size = int(np.ceil(np.log2(self.cnn_nbhd_size_range[1])))

        bit_num_list = [0, att_lstm_num_bit_num, long_term_lstm_seq_len_bit_num, short_term_lstm_seq_len_bit_num, nbhd_size_bit_num, cnn_nbhd_size]

        # check whether long term lstm seq len is odd 
        if self.gene_list[att_lstm_num_bit_num + long_term_lstm_seq_len_bit_num - 1].unit != True:
            self.gene_list[att_lstm_num_bit_num + long_term_lstm_seq_len_bit_num - 1].unit = True
            self.arg_list[1] += 1
            print('set_long_term_lstm_seq_len correct!')

        start = sum(bit_num_list[: 1])
        end = sum(bit_num_list[: 2])
        att_lstm_num_gene = self.gene_list[start : end]
        att_check_flag = True
        for temp in att_lstm_num_gene:
            if temp.unit == True:
                att_check_flag = False
        # That means this parameter(att_lstm_num) is zero
        if att_check_flag == True:
            for temp in range(start, end):
                if temp != end - 1:
                    self.gene_list[temp].unit = False
                else:
                    self.gene_list[temp].unit = True
        
        start = sum(bit_num_list[: 2])
        end = sum(bit_num_list[: 3])
        long_term_lstm_seq_len_gene = self.gene_list[start : end]
        long_term_lstm_seq_len_check_flag = True
        for temp in long_term_lstm_seq_len_gene:
            if temp.unit == True:
                long_term_lstm_seq_len_check_flag = False
        # That means this parameter(long_term_lstm_seq_len) is zero
        if long_term_lstm_seq_len_check_flag == True:
            for temp in range(start, end):
                if temp != end - 1:
                    self.gene_list[temp].unit = False
                else:
                    self.gene_list[temp].unit = True
        
        start = sum(bit_num_list[: 3])
        end = sum(bit_num_list[: 4])
        short_term_lstm_seq_len_gene = self.gene_list[start : end]
        short_term_lstm_seq_len_check_flag = True
        for temp in short_term_lstm_seq_len_gene:
            if temp.unit == True:
                short_term_lstm_seq_len_check_flag = False
        # That means this parameter(short_term_lstm_seq_len) is zero
        if short_term_lstm_seq_len_check_flag == True:
            for temp in range(start, end):
                if temp != end - 1:
                    self.gene_list[temp].unit = False
                else:
                    self.gene_list[temp].unit = True
        
        self.update_arg_list()

if __name__ == '__main__':
    i = Individual()
    # i.initialize_spec([3, 7, 7, 3, 3])
    i.initialize_spec([2, 3, 5, 1, 1])
    from evaluate import *
    update_individual_fitness(i)
    print("Test finish")
