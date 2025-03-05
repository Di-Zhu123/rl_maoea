import math

from pymoo.core.infill import InfillCriterion
from aroa_op import aroaop

class AROAMating(InfillCriterion):

    def __init__(self,
                 selection,
                 crossover,
                 mutation,
                 **kwargs):

        super().__init__(**kwargs)
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.action=None
        self.t = None
        self.maxEvals = None

    '''def _do(self,  problem, pop, n_offsprings, parents=None, **kwargs):

        # how many parents need to be select for the mating - depending on number of offsprings remaining
        n_matings = math.ceil(n_offsprings / self.crossover.n_offsprings)

        # if the parents for the mating are not provided directly - usually selection will be used
        if parents is None:

            # select the parents for the mating - just an index array
            parents = self.selection(problem, pop, n_matings, n_parents=self.crossover.n_parents, **kwargs)

        # do the crossover using the parents index and the population - additional data provided if necessary
        off = self.crossover(problem, parents, **kwargs)

        # do the mutation on the offsprings created through crossover
        off = self.mutation(problem, off, **kwargs)

        return off'''
    def _do(self,  problem, pop, n_offsprings, parents=None, **kwargs):

        '''# how many parents need to be select for the mating - depending on number of offsprings remaining
        n_matings = math.ceil(n_offsprings / self.crossover.n_offsprings)

        # if the parents for the mating are not provided directly - usually selection will be used
        if parents is None:

            # select the parents for the mating - just an index array
            parents = self.selection(problem, pop, n_matings, n_parents=self.crossover.n_parents, **kwargs)

        # do the crossover using the parents index and the population - additional data provided if necessary
        off = self.crossover(problem, parents, **kwargs)

        # do the mutation on the offsprings created through crossover
        off = self.mutation(problem, off, **kwargs)'''
        
        # t, maxEvals are global variables
        off = aroaop(pop, self.t, self.maxEvals, self.action, problem.xu, problem.xl)
        off = self.mutation(problem, off)
        return off

    def mating_load_action(self, action, t, maxEvals):
        self.action = action
        self.t = t
        self.maxEvals = maxEvals
