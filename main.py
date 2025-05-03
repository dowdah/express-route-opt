import math
import pulp
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from adjustText import adjust_text
from itertools import product
import os


FONT_PATH = os.path.join(os.path.dirname(__file__), 'NotoSerifSC-VariableFont_wght.ttf')
DRAW_FIGURES = False
fm.fontManager.addfont(FONT_PATH)
font_name = fm.FontProperties(fname=FONT_PATH).get_name()
plt.rcParams['font.family'] = font_name
plt.rcParams['axes.unicode_minus'] = False
param_grid = {
    'ants': [20, 50],
    'iterations': [200],
    'alpha': [1.0, 1.5],
    'beta': [3.0, 5.0],
    'rho': [0.3, 0.5],
    'q': [100, 500]
}


def solve_tsp(coords, cost_coeff=None):
    """
    求解单辆车无约束TSP，使用整数线性规划（ILP）方法。
    输入:
        coords: dict, 点编号 → (x, y) 坐标
            其中 0 号点为配送中心，其余为代收点
        cost_coeff: 成本系数矩阵
    返回:
        tour: list, 最优路径的点序列（含起点和终点0）
        total_dist: float, 总路程
    """
    # 节点集合
    nodes = list(coords.keys())
    n = len(nodes)

    # 计算权重矩阵
    weight = make_weight_matrix(coords, cost_coeff)

    # 定义问题
    prob = pulp.LpProblem("TSP_MTZ", pulp.LpMinimize)

    # 决策变量 x[i,j] ∈ {0,1}
    x = pulp.LpVariable.dicts('x',
                              (nodes, nodes),
                              lowBound=0, upBound=1,
                              cat=pulp.LpBinary)

    # 辅助变量 u[i] ∈ [1, n-1]（MTZ）
    # 注意：不对 0 号点定义 u[0]
    u = pulp.LpVariable.dicts('u',
                              [i for i in nodes if i != 0],
                              lowBound=1, upBound=n-1,
                              cat=pulp.LpInteger)

    # 目标函数：最小化总距离
    prob += pulp.lpSum(weight[(i, j)] * x[i][j] for i in nodes for j in nodes)

    # 约束1：每个代收点出度=1
    for i in nodes:
        if i != 0:
            prob += pulp.lpSum(x[i][j] for j in nodes if j != i) == 1

    # 约束2：每个代收点入度=1
    for j in nodes:
        if j != 0:
            prob += pulp.lpSum(x[i][j] for i in nodes if i != j) == 1

    # 约束3：配送中心出发一次、返回一次
    prob += pulp.lpSum(x[0][j] for j in nodes if j != 0) == 1
    prob += pulp.lpSum(x[i][0] for i in nodes if i != 0) == 1

    # MTZ 消除子环约束
    # u[i] - u[j] + n * x[i,j] <= n-1, 对所有 i≠j 且 i,j≠0
    for i in nodes:
        for j in nodes:
            if i != j and i != 0 and j != 0:
                prob += u[i] - u[j] + n * x[i][j] <= n - 1

    # 求解
    solver = pulp.PULP_CBC_CMD(msg=False)  # 不打印求解过程
    prob.solve(solver)

    # 提取结果
    # 构建路径：从 0 开始，依次找到下一个节点，直到回到 0
    tour = [0]
    current = 0
    while True:
        next_nodes = [j for j in nodes if j != current and pulp.value(x[current][j]) > 0.5]
        if not next_nodes:
            break
        nxt = next_nodes[0]
        tour.append(nxt)
        current = nxt
        if current == 0:
            break

    total_dist = pulp.value(prob.objective)
    return tour, total_dist


def solve_tsp_aco(coords, cost_coeff=None, ants=10, iterations=100, alpha=1.0, beta=5.0, rho=0.5, q=100):
    """
    使用蚁群算法求解TSP。
    输入:
        coords: dict, 点编号 → (x, y) 坐标
        cost_coeff: 成本系数矩阵
        ants: 蚂蚁数量
        iterations: 迭代次数
        alpha: 信息素重要程度因子
        beta: 启发函数重要程度因子
        rho: 信息素挥发率
        q: 信息素强度常数
    返回:
        best_tour: list, 最优路径的点序列（含起点和终点0）
        best_length: float, 最短路径长度
    """
    nodes = list(coords.keys())

    # 计算权重矩阵
    weight = make_weight_matrix(coords, cost_coeff)

    # 初始化信息素矩阵，全部设为1
    pheromone = {}
    for i in nodes:
        for j in nodes:
            pheromone[(i, j)] = 1.0

    # 初始化启发因子矩阵，距离倒数（避免除零）
    heuristic = {}
    for i in nodes:
        for j in nodes:
            if weight[(i, j)] == 0:
                heuristic[(i, j)] = 0
            else:
                heuristic[(i, j)] = 1.0 / weight[(i, j)]

    best_tour = None
    best_length = float('inf')

    for iter in range(iterations):
        all_tours = []
        all_lengths = []

        for ant in range(ants):
            # 构建路径，从0开始
            unvisited = set(nodes)
            unvisited.remove(0)
            tour = [0]
            current = 0

            while unvisited:
                # 计算转移概率
                denom = 0.0
                probs = []
                for j in unvisited:
                    tau = pheromone[(current, j)] ** alpha
                    eta = heuristic[(current, j)] ** beta
                    denom += tau * eta
                for j in unvisited:
                    tau = pheromone[(current, j)] ** alpha
                    eta = heuristic[(current, j)] ** beta
                    prob = (tau * eta) / denom if denom > 0 else 0
                    probs.append((j, prob))

                # 轮盘赌选择下一个节点
                r = 0.0
                import random
                rand = random.random()
                for node, prob in probs:
                    r += prob
                    if r >= rand:
                        next_node = node
                        break
                else:
                    # 出现概率和小于1的情况，随机选择
                    next_node = random.choice(list(unvisited))

                tour.append(next_node)
                unvisited.remove(next_node)
                current = next_node

            # 回到起点0
            tour.append(0)

            # 计算路径长度
            length = 0.0
            for i in range(len(tour) - 1):
                length += weight[(tour[i], tour[i + 1])]

            all_tours.append(tour)
            all_lengths.append(length)

            # 更新迭代最优解
            if length < best_length:
                best_length = length
                best_tour = tour

        # 信息素挥发
        for key in pheromone:
            pheromone[key] *= (1 - rho)
            if pheromone[key] < 0.0001:
                pheromone[key] = 0.0001  # 防止信息素过低

        # 信息素更新，路径越短，信息素增量越大
        for k in range(ants):
            tour = all_tours[k]
            length = all_lengths[k]
            delta = q / length if length > 0 else 0
            for i in range(len(tour) - 1):
                pheromone[(tour[i], tour[i + 1])] += delta

    return best_tour, best_length


# 问题四：考虑载重限制的多车路径优化
def solve_multi_vehicle_routing_with_capacity(coords, demand, cost_coeff, capacity=4.0):
    """
    问题四：考虑载重限制的多车路径优化。
    输入:
        coords: dict, 点编号 → (x, y) 坐标
        demand: dict, 点编号 → 需求量（吨），配送中心为0点，需求量为0
        cost_coeff: 成本系数矩阵（二维数组或DataFrame）
        capacity: float, 每辆车最大载重（吨）
    返回:
        routes: list of lists，每辆车的路径序列（均以0开始和结束）
        total_cost: float, 总运输成本
    """

    nodes = list(coords.keys())
    n = len(nodes)

    # 计算加权成本
    weight = make_weight_matrix(coords, cost_coeff)

    # 初始化模型
    prob = pulp.LpProblem("Multi_Vehicle_VRP", pulp.LpMinimize)

    # 最大车辆数量：最多每个点一辆车服务
    K = len([i for i in nodes if i != 0])

    # 决策变量 x[i][j][k]：第k辆车是否走边(i,j)
    x = pulp.LpVariable.dicts("x", (range(n), range(n), range(K)), 0, 1, cat=pulp.LpBinary)
    # u[i]: 车载货量辅助变量（用于消除子环）
    u = pulp.LpVariable.dicts("u", (range(n)), 0, capacity, cat=pulp.LpContinuous)

    # y[i][k]：代收点 i 是否由第k辆车服务
    y = pulp.LpVariable.dicts("y", (range(n), range(K)), 0, 1, cat=pulp.LpBinary)

    # 目标函数：总运输成本
    prob += pulp.lpSum(weight[(i, j)] * x[i][j][k] for i in range(n) for j in range(n) for k in range(K))

    # 每个代收点必须被服务一次
    for j in range(1, n):
        prob += pulp.lpSum(y[j][k] for k in range(K)) == 1

    # 每辆车路径起点和终点都在配送中心
    for k in range(K):
        prob += pulp.lpSum(x[0][j][k] for j in range(1, n)) <= 1
        prob += pulp.lpSum(x[i][0][k] for i in range(1, n)) <= 1

    # 流量守恒约束 + 关联x和y变量
    for k in range(K):
        for h in range(1, n):
            prob += (pulp.lpSum(x[i][h][k] for i in range(n) if i != h) -
                     pulp.lpSum(x[h][j][k] for j in range(n) if j != h)) == 0
            prob += pulp.lpSum(x[i][h][k] for i in range(n) if i != h) == y[h][k]

    # 载重约束
    for k in range(K):
        prob += pulp.lpSum(demand[j] * y[j][k] for j in range(1, n)) <= capacity

    # 消除子环约束（MTZ）
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                for k in range(K):
                    prob += u[i] - u[j] + capacity * x[i][j][k] <= capacity - demand[j]

    # 求解模型
    solver = pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)

    # 提取路线
    routes = [[] for _ in range(K)]
    for k in range(K):
        current = 0
        route = [0]
        visited = set()
        while True:
            nexts = [j for j in range(n) if j != current and pulp.value(x[current][j][k]) > 0.5]
            if not nexts:
                break
            nxt = nexts[0]
            route.append(nxt)
            visited.add(nxt)
            current = nxt
            if nxt == 0:
                break
        if len(route) > 1:
            routes[k] = route

    total_cost = pulp.value(prob.objective)
    return [r for r in routes if r], total_cost


def draw_map(coords, tour=None, title="最优配送路径示意图"):
    if not DRAW_FIGURES:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    if tour:
        # 绘制路径折线
        xs = [coords[node][0] for node in tour]
        ys = [coords[node][1] for node in tour]
        ax.plot(xs, ys, marker='o', linestyle='-', label='路径')
        # 标注配送中心
        ax.scatter(coords[0][0], coords[0][1],
                   s=200, marker='*', edgecolors='k', linewidths=1.5,
                   label='配送中心')
        # 标注各代收点编号
        for node in tour:
            if node != 0:
                x, y = coords[node]
                ax.text(x, y, str(node), fontsize=12,
                        ha='right', va='bottom')
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('X 坐标', fontsize=12)
        ax.set_ylabel('Y 坐标', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.axis('equal')
        ax.legend(loc='best')
        plt.tight_layout()
        plt.show()
    else:
        # 绘制配送中心和代收点
        xs = [coords[node][0] for node in coords]
        ys = [coords[node][1] for node in coords]
        ax.scatter(xs, ys, marker='o', label='代收点')
        ax.scatter(coords[0][0], coords[0][1],
                   s=200, marker='*', edgecolors='k', linewidths=1.5,
                   label='配送中心')
        # 添加每个点的坐标标签
        texts = []
        for node, (x, y) in coords.items():
            texts.append(ax.text(x, y, f'{node}({x:.3f},{y:.3f})', fontsize=12))
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('X 坐标', fontsize=12)
        ax.set_ylabel('Y 坐标', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.axis('equal')
        ax.legend(loc='best')
        plt.tight_layout()
        plt.show()


def make_weight_matrix(coords, cost_coeff=None):
    """
    计算权重矩阵。当 cost_coeff=None 时，使用欧几里得距离作为权重；否则，用欧几里得距离乘以成本系数作为权重。
    输入:
        coords: dict, 点编号 → (x, y) 坐标
        cost_coeff: 成本系数矩阵
    返回:
        weight: dict, (i, j) → 权重
    """
    nodes = list(coords.keys())
    weight = {}
    if cost_coeff is None:
        for i in nodes:
            for j in nodes:
                xi, yi = coords[i]
                xj, yj = coords[j]
                weight[(i, j)] = math.hypot(xi - xj, yi - yj)
    else:
        for i in nodes:
            for j in nodes:
                xi, yi = coords[i]
                xj, yj = coords[j]
                euclid = math.hypot(xi - xj, yi - yj)
                weight[(i, j)] = euclid * cost_coeff[i][j]
    return weight


def evaluate_aco_params(coords, cost_coeff, param_grid):
    """
    自动评估多组ACO参数组合。
    输入:
        coords: dict, 点编号 → (x, y) 坐标
        cost_coeff: 成本系数矩阵
        param_grid: dict, 参数名称 → 参数值列表
    返回:
        results: list, 每组参数的结果列表，包含参数组合、成本和路径
    """
    keys = list(param_grid.keys())
    all_combinations = list(product(*[param_grid[k] for k in keys]))

    results = []
    for values in all_combinations:
        params = dict(zip(keys, values))
        print(f"评估参数组合: {params}")
        tour, cost = solve_tsp_aco(coords, cost_coeff=cost_coeff, **params)
        print(f"  → 成本: {cost:.2f}, 路径: {tour}")
        results.append((params, cost, tour))

    # 输出最优结果
    best = min(results, key=lambda x: x[1])
    print("\n最优组合:")
    print(f"参数: {best[0]}")
    print(f"成本: {best[1]:.2f}")
    print(f"路径: {best[2]}")
    return results


if __name__ == "__main__":
    # 读取附表1：配送中心和各代收点的坐标
    points_data = pd.read_excel("建模赛题附表数据.xlsx", sheet_name="附表1_坐标")
    points = points_data.sort_values("编号")[["横坐标", "纵坐标"]].values
    coords = {i: (x, y) for i, (x, y) in enumerate(points)}
    # 读取附表2：当日需求二值表
    delivery_flags_data = pd.read_excel("建模赛题附表数据.xlsx", sheet_name="附表2_配送情况")
    need_delivery = delivery_flags_data[delivery_flags_data["配送情况"] == 1]["代收点编号"].tolist()
    # 读取附表3：道路单位成本系数
    cost_df = pd.read_excel("建模赛题附表数据.xlsx", sheet_name="附表3_成本比例")
    cost_coeff_df = cost_df.pivot(index='起点编号', columns='终点编号', values='成本比例')
    cost_coeff_df = cost_coeff_df.sort_index().sort_index(axis=1).fillna(0)
    cost_coeff = cost_coeff_df.values
    # 读取附表4：各代收点货物量
    demand_data = pd.read_excel("建模赛题附表数据.xlsx", sheet_name="附表4_货物量")
    demand_list = demand_data["货物量（吨）"].tolist()
    demand_dict = {i: (0.0 if i == 0 else demand_list[i - 1]) for i in range(len(demand_list) + 1)}
    # 画出配送中心和代收点坐标示意图
    draw_map(coords, title="配送中心和代收点坐标示意图")

    # 保存原始全量坐标以备题3使用
    full_coords = coords.copy()

    # # 求解问题一
    # tour, dist = solve_tsp(coords)
    # print("---问题一---")
    # print(f"最优路径: {tour}")
    # print(f"总里程: {dist:.2f} 公里")
    # draw_map(coords, tour, "最优配送路径示意图-问题一")
    #
    # # 问题二：部分需求TSP
    # # 构造问题二的坐标集
    # coords_q2 = full_coords.copy()
    # for k in list(coords_q2.keys()):
    #     if k != 0 and k not in need_delivery:
    #         coords_q2.pop(k)
    # tour, dist = solve_tsp(coords_q2)
    # print("---问题二---")
    # print(f"最优路径: {tour}")
    # print(f"总里程: {dist:.2f} 公里")
    # draw_map(coords_q2, tour, "最优配送路径示意图-问题二")
    #
    # # 问题三：加权TSP
    # # print("---问题三(蚁群算法)---")
    # # evaluate_aco_params(full_coords, cost_coeff, param_grid)
    # # print(f"最优路径: {best_tour}")
    # # print(f"加权总成本: {best_cost:.2f}")
    # # draw_map(full_coords, best_tour, "最优配送路径示意图-问题三")
    # best_tour, best_cost = solve_tsp(full_coords, cost_coeff=cost_coeff)
    # print("---问题三(分支定界法)---")
    # print(f"最优路径: {best_tour}")
    # print(f"加权总成本: {best_cost:.2f}")
    # draw_map(full_coords, best_tour, "最优配送路径示意图-问题三")

    # 问题四：多车辆载重约束路径规划
    print("---问题四---")
    routes, total_cost = solve_multi_vehicle_routing_with_capacity(full_coords, demand_dict, cost_coeff)
    for idx, route in enumerate(routes):
        print(f"车辆 {idx + 1} 路径: {route}")
    print(f"总运输成本: {total_cost:.2f}")
