from individual import *

class Population:
    def __init__(self, pop_num):
        self.pop_num = pop_num
        self.pops = []
        for i in range(pop_num):
            indi = Individual()
            indi.initialize()
            self.pops.append(indi)

    def get_individual_at(self, i):
        return self.pops[i]

    def get_pop_size(self):
        return len(self.pops)

    def set_population(self, new_pop):
        self.pops = new_pop

    def __str__(self):
        _str = []
        for i in range(self.get_pop_size()):
            _str.append(str(self.get_individual_at(i)))
        return '\n'.join(_str)