import random
import math
import pandas as pd
import numpy as np


def compute_distance_matrix(locations):
    """Compute the Euclidean distance matrix for given locations."""
    n = len(locations)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dx = locations[i][0] - locations[j][0]
            dy = locations[i][1] - locations[j][1]
            dist[i][j] = math.hypot(dx, dy)
    return dist


def total_distance(route, dist_matrix):
    """Calculate total round-trip distance for a given route (excluding depot at index 0)."""
    # Start from depot (0) to first point
    distance = dist_matrix[0][route[0]]
    # Traverse between customer points
    for i in range(len(route) - 1):
        distance += dist_matrix[route[i]][route[i+1]]
    # Return to depot
    distance += dist_matrix[route[-1]][0]
    return distance


def init_population(dist_matrix, pop_size):
    """Initialize a population of random routes (permutations of customer indices)."""
    n = len(dist_matrix) - 1  # exclude depot
    population = []
    for _ in range(pop_size):
        perm = list(range(1, n+1))
        random.shuffle(perm)
        population.append(perm)
    return population


def tournament_selection(pop, dist_matrix, k=5):
    """Select one individual via tournament selection (size k)."""
    participants = random.sample(pop, k)
    participants.sort(key=lambda route: total_distance(route, dist_matrix))
    return participants[0]


def order_crossover(p1, p2):
    """Perform Order Crossover (OX) between two parents."""
    size = len(p1)
    a, b = sorted(random.sample(range(size), 2))
    child = [None] * size
    # Copy segment from first parent
    child[a:b+1] = p1[a:b+1]
    # Fill remaining genes from second parent in order
    fill = [gene for gene in p2 if gene not in child]
    idx = 0
    for i in range(size):
        if child[i] is None:
            child[i] = fill[idx]
            idx += 1
    return child


def swap_mutation(route, mutation_rate=0.02):
    """Randomly swap two genes in the route based on mutation rate."""
    for i in range(len(route)):
        if random.random() < mutation_rate:
            j = random.randrange(len(route))
            route[i], route[j] = route[j], route[i]


def genetic_algorithm(locations, pop_size=100, generations=500, tournament_k=5, mutation_rate=0.02):
    """Main GA loop to solve the TSP for given locations."""
    random.seed(42)
    dist_matrix = compute_distance_matrix(locations)
    population = init_population(dist_matrix, pop_size)
    best_route = min(population, key=lambda r: total_distance(r, dist_matrix))
    best_dist = total_distance(best_route, dist_matrix)

    for gen in range(generations):
        new_pop = []
        for _ in range(pop_size):
            p1 = tournament_selection(population, dist_matrix, tournament_k)
            p2 = tournament_selection(population, dist_matrix, tournament_k)
            child = order_crossover(p1, p2)
            swap_mutation(child, mutation_rate)
            new_pop.append(child)
        population = new_pop

        # Update best
        current_best = min(population, key=lambda r: total_distance(r, dist_matrix))
        current_dist = total_distance(current_best, dist_matrix)
        if current_dist < best_dist:
            best_route, best_dist = current_best, current_dist

        # Optional: progress output
        if gen % 50 == 0:
            print(f"Gen {gen}: Best dist = {best_dist:.2f}")

    # Return full route including depot at start and end
    return [0] + best_route + [0], best_dist


if __name__ == "__main__":
    coords = pd.read_excel("建模赛题附表数据.xlsx", sheet_name="附表1_坐标")
    locations = coords.sort_values("编号")[["横坐标", "纵坐标"]].values
    locations = np.delete(locations, 6, 0)
    best_route, best_distance = genetic_algorithm(locations)
    print("最佳路径节点索引：", best_route)
    print(f"总里程: {best_distance:.2f} 公里")
