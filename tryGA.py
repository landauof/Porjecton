import random

from deap import base
from deap import creator
from deap import tools
from Classifier import classify
import numpy as np
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

IND_SIZE = 1


def rand_atrb1():
    rand = random.random()
    while rand > 10/150:
        rand = random.random()
    return rand


def rand_atrb2():
    rand = random.random()
    while rand > 0.001:
        rand = random.random()
    return rand


def evaluate(individual):
    # Do some hard computing on the individual
    nu = individual[0]
    gamma = individual[1]
    # return accuracy of the classier
    return (classify(220, 1, nu, gamma) + 1,)


def mutate1(individual):
    individual[0] = rand_atrb1()
    individual[1] = rand_atrb2()
    return individual,


def clone1(individual):
    ind = toolbox.individual()
    ind[0] = individual[0]
    ind[1] = individual[1]
    return ind


def crossover2(individual1, individual2):
    rand = random.random()
    if rand > 0.5:
        individual1[0] = (individual2[0] + individual1[0])/2
        individual1[1] = (individual2[1] + individual1[1])/2
    else:
        individual2[0] = (individual2[0] + individual1[0])/2
        individual2[1] = (individual2[1] + individual1[1])/2


def crossover1(individual1, individual2):
    rand = random.random()
    if rand > 0.5:
        temp = individual1[0]
        individual1[0] = individual2[0]
        individual2[0] = temp
    else:
        temp = individual1[1]
        individual1[1] = individual2[1]
        individual2[1] = temp


toolbox = base.Toolbox()
# toolbox.register("attr_float", rand_atrb)
toolbox.register("individual", tools.initCycle, creator.Individual,
                 [lambda: rand_atrb1(), lambda:rand_atrb2()], n=IND_SIZE)

toolbox.register("mutate", mutate1)
toolbox.register("clone", clone1)
toolbox.register("crossover", crossover2)
toolbox.register("evaluate", evaluate)
toolbox.register("select", tools.selTournament, tournsize=4)
ind1 = toolbox.individual()
print(ind1)
print(ind1.fitness.valid)

ind1.fitness.values = evaluate(ind1)
print(ind1.fitness.valid)    # True
print(ind1.fitness)

mutant = toolbox.clone(ind1)
print(mutant)
print(ind1)
ind2, = toolbox.mutate(mutant)
del mutant.fitness.values
print(ind1)
print(ind2)

print('\n')
print('\n')

child1, child2 = [toolbox.clone(ind) for ind in (ind1, ind2)]
print(child1)
print(child2)
toolbox.crossover(child1, child2)
print(child1)
print(child2)


NGEN = 10
CXPB = 0.2
MUTPB = 0.25
i = 0
pop = []
pop_fit = []
while i < 70:
    pop.append(toolbox.individual())
    i = i + 1
    pop_fit.append(0.0)
max1 = max(pop_fit)


# while max1 < 1.85:
for g in range(NGEN):
    # Select the next generation individuals
    offspring = toolbox.select(pop, 30)
    # Clone the selected individuals
    offspring = list(map(toolbox.clone, offspring))

    # Apply crossover on the offspring
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CXPB:
            toolbox.crossover(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    # Apply mutation on the offspring
    for mutant in offspring:
        if random.random() < MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # The population is entirely replaced by the offspring
    pop[:] = offspring
    pop_fit[:] = [p.fitness.values[0] for p in pop]
    max1 = max(pop_fit)

print(pop)
print(max1)
