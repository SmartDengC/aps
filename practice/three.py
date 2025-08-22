import matplotlib.pyplot as plt
import numpy as np


# 定义适应度函数
def fitness_function(x):
    return x + 10 * np.sin(5 * x) + 7 * np.cos(4 * x)


# 遗传算法参数
POP_SIZE = 50  # 种群大小
DNA_SIZE = 17  # DNA长度（二进制编码位数，决定精度）
CROSSOVER_RATE = 0.8  # 交叉概率
MUTATION_RATE = 0.003  # 变异概率
N_GENERATIONS = 100  # 迭代次数
X_BOUND = [0, 9]  # x的取值范围


# 生成种群DNA（二进制编码）
def generate_population():
    return np.random.randint(2, size=(POP_SIZE, DNA_SIZE))


# 将DNA二进制转换为十进制数（解码）
def translate_dna(pop):
    # 将二进制转换为十进制并归一化到[0,1]范围
    binary_to_decimal = pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / (2 ** DNA_SIZE - 1)
    # 映射到X_BOUND范围
    return X_BOUND[0] + binary_to_decimal * (X_BOUND[1] - X_BOUND[0])


# 计算适应度（这里使用函数值加上一个足够大的数确保非负）
def get_fitness(pop):
    x = translate_dna(pop)
    # 因为函数可能有负值，我们加上一个足够大的数确保适应度非负
    # 找到当前种群的最小值，然后加上其绝对值再加一个小数确保正值
    raw_fitness = fitness_function(x)
    min_fitness = np.min(raw_fitness)
    if min_fitness < 0:
        fitness = raw_fitness + abs(min_fitness) + 1e-3
    else:
        fitness = raw_fitness + 1e-3  # 仍然加一个小数避免全零
    return fitness


# 选择（轮盘赌选择）
def select(pop, fitness):
    # 确保所有适应度都是正数
    fitness = np.clip(fitness, 1e-10, None)  # 防止出现零或负数
    # 按概率选择，适应度高的被选中的概率高
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=fitness / fitness.sum())
    return pop[idx]


# 交叉操作（单点交叉）
def crossover(parent, pop):
    if np.random.rand() < CROSSOVER_RATE:
        # 随机选择另一个个体
        i_ = np.random.randint(0, POP_SIZE, size=1)
        # 随机选择交叉点
        cross_points = np.random.randint(0, 2, size=DNA_SIZE).astype(bool)  # 修改为bool类型
        # 交换基因
        parent[cross_points] = pop[i_, cross_points]
    return parent


# 变异操作（位翻转）
def mutate(child):
    for point in range(DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:
            child[point] = 1 if child[point] == 0 else 0
    return child


# 主遗传算法流程
def ga():
    # 初始化种群
    pop = generate_population()

    # 记录每一代的最佳个体和适应度
    best_fitness_history = []
    best_individual_history = []

    for generation in range(N_GENERATIONS):
        # 计算适应度
        fitness = get_fitness(pop)
        raw_fitness = fitness_function(translate_dna(pop))  # 原始适应度用于记录

        # 记录当前代的最佳个体
        best_fitness = np.max(raw_fitness)
        best_fitness_history.append(best_fitness)
        best_idx = np.argmax(raw_fitness)
        best_individual = translate_dna(pop[best_idx])
        best_individual_history.append(best_individual)

        print(f"Generation {generation}: Best x = {best_individual:.6f}, y = {best_fitness:.6f}")

        # 选择
        pop = select(pop, fitness)

        # 复制当前种群用于交叉和变异
        pop_copy = pop.copy()

        # 对每个个体进行交叉和变异
        for parent in pop:
            child = crossover(parent, pop_copy)
            child = mutate(child)
            parent[:] = child

    # 最终结果
    final_fitness = fitness_function(translate_dna(pop))
    best_idx = np.argmax(final_fitness)
    best_x = translate_dna(pop[best_idx])
    best_y = final_fitness[best_idx]
    print(f"\nFinal result: x = {best_x:.6f}, y = {best_y:.6f}")

    return best_fitness_history, best_individual_history


def invoke():
    # 运行遗传算法
    best_fitness_history, best_individual_history = ga()

    # 绘制适应度变化曲线
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(N_GENERATIONS), best_fitness_history)
    plt.title("Fitness over generations")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")

    # 绘制函数曲线和搜索过程
    plt.subplot(1, 2, 2)
    x = np.linspace(X_BOUND[0], X_BOUND[1], 200)
    y = fitness_function(x)
    plt.plot(x, y)
    plt.scatter(best_individual_history, best_fitness_history, c='red', s=20, alpha=0.3)
    plt.title("Function and search process")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    invoke()
    # x = 7.8567  # 24.85536
    # print(fitness_function(x))
