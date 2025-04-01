import numpy as np
from typing import List, Tuple
import pandas as pd
import random
from collections import defaultdict


class TaskAllocationGA:
    def __init__(self,
                affinity_df: pd.DataFrame, 
                duration_df: pd.DataFrame, 
                initial_population_size: int,
                generations: int):
        
        self.affinity_df = affinity_df
        self.duration_df = duration_df
        self.initial_population_size = initial_population_size
        self.tasks_size = affinity_df.shape[1]
        self.employees_size = affinity_df.shape[0]
        self.generations = generations
    
    def generate_initial_population(self) -> List[List[int]]:
        population = []

        while len(population) < self.initial_population_size:
            valid = False
            while not valid:
                chromosome = np.random.randint(0, self.employees_size, size=self.tasks_size).tolist()
                used_employees = set(chromosome)
                if len(used_employees) == self.employees_size:
                    population.append(chromosome)
                    valid = True

        return population
    
    def evaluate_fitness(self, chromosome: List[int]) -> float:
        hours_by_employee = defaultdict(int)
        tasks_by_employee = defaultdict(list)

        for task_index, employee_id in enumerate(chromosome):
            # Soma as horas e guarda as tarefas atribuídas por funcionário
            task_duration = self.duration_df.iloc[task_index]["Duracao_horas"]
            hours_by_employee[employee_id] += task_duration
            tasks_by_employee[employee_id].append(task_index)

        # Verifica restrições
        for employee in range(self.employees_size):
            num_tasks = len(tasks_by_employee.get(employee, []))
            total_hours = hours_by_employee.get(employee, 0)

            if num_tasks < 1 or num_tasks > 3:
                return 0.0
            if total_hours > 10:
                return 0.0

        # Solução válida: somar afinidades
        fitness = 0
        for task_index, employee_id in enumerate(chromosome):
            affinity = self.affinity_df.iloc[employee_id, task_index]
            fitness += affinity

        return float(fitness)

    def crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        
        assert len(parent1) == self.tasks_size and len(parent2) == self.tasks_size

        # Sorteia dois pontos de corte distintos e em ordem
        point1, point2 = sorted(random.sample(range(self.tasks_size), 2))

        # Cria os filhos com troca entre os pontos
        child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
        child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]

        return child1, child2

    def mutate(self, chromosome: List[int], mutation_rate: float = 0.05) -> List[int]:
        if random.random() < mutation_rate:
            mutated = chromosome.copy()
            task_index = random.randint(0, self.tasks_size - 1)
            current_employee = mutated[task_index]

            # Escolhe um novo funcionário diferente do atual
            possible_employees = [i for i in range(self.employees_size) if i != current_employee]
            mutated[task_index] = random.choice(possible_employees)

            return mutated

        return chromosome  
    
    def generate_next_generation(self, current_population: List[List[int]], mutation_rate: float = 0.05) -> List[List[int]]:
        offspring = []
        # Ordenar população por fitness
        evaluated_parents = [(ind, self.evaluate_fitness(ind)) for ind in current_population]
        evaluated_parents.sort(key=lambda x: x[1], reverse=True)
        current_population = [ind for ind, _ in evaluated_parents]
        # Crossover entre pares consecutivos
        for i in range(0, len(current_population), 2):
            parent1 = current_population[i]
            parent2 = current_population[(i + 1) % len(current_population)]
            child1, child2 = self.crossover(parent1, parent2)
            offspring.extend([child1, child2])

        # Aplicar mutação
        mutated = [self.mutate(child, mutation_rate) for child in offspring]

        # População temporária: pais + filhos + mutantes
        temp_population = current_population + offspring + mutated

        # Avaliar fitness de todos
        fitness_values = [self.evaluate_fitness(ind) for ind in temp_population]
        total_fitness = sum(fitness_values)

        # Fallback: população inválida (fitness zero total)
        if total_fitness == 0:
            return self.generate_initial_population()

        # Seleção por roleta usando random.choices
        next_generation = random.choices(
            population=temp_population,
            weights=fitness_values,
            k=self.initial_population_size
        )

        return next_generation
    
    def run(self, mutation_rate: float = 0.05) -> List[Tuple[np.ndarray, float]]:
        population = self.generate_initial_population()
        best_per_generation = []

        for _ in range(self.generations):
            # Avaliar todos os indivíduos válidos
            evaluated = [(ind, self.evaluate_fitness(ind)) for ind in population if self.evaluate_fitness(ind) > 0]
            if evaluated:
                # Adiciona o melhor da geração atual
                best = max(evaluated, key=lambda x: x[1])
                best_per_generation.append(best)

            # Evoluir para próxima geração
            population = self.generate_next_generation(population, mutation_rate=mutation_rate)

        # Avaliar última geração e incluir o melhor (caso ainda não esteja)
        evaluated = [(ind, self.evaluate_fitness(ind)) for ind in population if self.evaluate_fitness(ind) > 0]
        if evaluated:
            best = max(evaluated, key=lambda x: x[1])
            best_per_generation.append(best)

        # Retornar todos os melhores em ordem decrescente de fitness
        return sorted(best_per_generation, key=lambda x: x[1], reverse=True)


class PortfolioGA:
    def __init__(self,
                 expected_returns: np.ndarray,
                 variances: np.ndarray,
                 initial_population_size: int,
                 generations: int,
                 cardinality: int = 5,
                 min_allocation: float = 0.1,
                 lambda_: float = 0.5):
        """
        expected_returns: vetor (n,) com os retornos esperados dos ativos
        variances: vetor (n,) com variância individual de cada ativo
        initial_population_size: número de indivíduos por geração
        generations: número total de gerações
        cardinality: número de ativos no portfólio
        min_allocation: alocação mínima por ativo (ex: 0.1 = 10%)
        lambda_: peso do retorno na função de fitness (0.5 = equilíbrio)
        """
        self.returns = expected_returns
        self.variances = variances
        self.n_assets = len(expected_returns)
        self.pop_size = initial_population_size
        self.generations = generations
        self.cardinality = cardinality
        self.min_allocation = min_allocation
        self.lambda_ = lambda_

    def generate_initial_population(self) -> List[np.ndarray]:
        population = []
        while len(population) < self.pop_size:
            ativos = random.sample(range(self.n_assets), self.cardinality)
            pesos = np.random.dirichlet(np.ones(self.cardinality))
            if all(p >= self.min_allocation for p in pesos):
                cromossomo = np.zeros(self.n_assets)
                cromossomo[ativos] = pesos
                population.append(cromossomo)
        return population

    def evaluate_fitness(self, weights: np.ndarray) -> float:
        if not np.isclose(weights.sum(), 1.0):
            return 0.0
        if np.count_nonzero(weights) != self.cardinality:
            return 0.0
        if any(p < self.min_allocation for p in weights if p > 0):
            return 0.0

        retorno = np.dot(self.returns, weights)
        risco = np.sum((weights ** 2) * self.variances)

        return self.lambda_ * retorno - (1 - self.lambda_) * risco

    def crossover(self, p1: np.ndarray, p2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ativos1 = np.flatnonzero(p1)
        ativos2 = np.flatnonzero(p2)
        i, j = sorted(random.sample(range(self.cardinality), 2))

        # Filho 1
        mantidos1 = ativos1[i:j+1]
        restantes1 = [a for a in range(self.n_assets) if a not in mantidos1]
        novos1 = random.sample(restantes1, self.cardinality - len(mantidos1))
        ativos_f1 = list(mantidos1) + novos1
        pesos1 = np.random.dirichlet(np.ones(self.cardinality))
        filho1 = np.zeros(self.n_assets)
        filho1[ativos_f1] = pesos1

        # Filho 2
        mantidos2 = ativos2[i:j+1]
        restantes2 = [a for a in range(self.n_assets) if a not in mantidos2]
        novos2 = random.sample(restantes2, self.cardinality - len(mantidos2))
        ativos_f2 = list(mantidos2) + novos2
        pesos2 = np.random.dirichlet(np.ones(self.cardinality))
        filho2 = np.zeros(self.n_assets)
        filho2[ativos_f2] = pesos2

        return filho1, filho2

    def mutate(self, chrom: np.ndarray, mutation_rate: float = 0.05) -> np.ndarray:
        if random.random() < mutation_rate:
            ativos = np.flatnonzero(chrom)
            a, b = random.sample(list(ativos), 2)
            delta = round(random.uniform(0.01, 0.05), 4)
            if chrom[a] - delta >= self.min_allocation:
                novo = chrom.copy()
                novo[a] -= delta
                novo[b] += delta
                return novo
        return chrom

    def generate_next_generation(self, population: List[np.ndarray], mutation_rate: float = 0.05) -> List[np.ndarray]:
        fitness_scores = [self.evaluate_fitness(ind) for ind in population]
        ranked = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)
        population = [ind for ind, _ in ranked]

        offspring = []
        for i in range(0, len(population), 2):
            p1, p2 = population[i], population[(i + 1) % len(population)]
            c1, c2 = self.crossover(p1, p2)
            offspring.append(self.mutate(c1, mutation_rate))
            offspring.append(self.mutate(c2, mutation_rate))

        temp_population = population + offspring
        fitness_temp = [self.evaluate_fitness(ind) for ind in temp_population]
        total_fitness = sum(fitness_temp)

        if total_fitness == 0:
            return self.generate_initial_population()

        return random.choices(temp_population, weights=fitness_temp, k=self.pop_size)

    def run(self, mutation_rate: float = 0.05) -> List[Tuple[np.ndarray, float]]:
        population = self.generate_initial_population()
        best_individuals = []

        for _ in range(self.generations):
            evaluated = [(ind, self.evaluate_fitness(ind)) for ind in population if self.evaluate_fitness(ind) > 0]
            best_individuals.extend(evaluated)
            population = self.generate_next_generation(population, mutation_rate)

        evaluated = [(ind, self.evaluate_fitness(ind)) for ind in population if self.evaluate_fitness(ind) > 0]
        best_individuals.extend(evaluated)

        return sorted(best_individuals, key=lambda x: x[1], reverse=True)[:3]