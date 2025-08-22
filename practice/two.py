import copy
import math
import random

import matplotlib.pyplot as plt

PI = 3.1415926
# 个体长度
CHROM_LEN = 20
# 种群大小
POP_SIZE = 40
CMIN = 0
# 最大遗传代数
MAX_GENERATION = 40
# 交叉概率
PC = 0.7
# 变异概率
PM = 0.01


# 优化函数
def F(x):
    # return math.sin(10 * PI * x) / x
    return x + 10 * math.sin(5 * x) + 7 * math.cos(4 * x)


# 解码器
def decode(chrom, lb, ub):
    # 二进制对应的十进制数
    temp = int(chrom, 2)
    # 最终解码值
    x = lb + temp * (ub - lb) / (math.pow(2, CHROM_LEN) - 1)
    return x


# 个体类
class Individual:
    def __init__(self):
        temp = []
        for _ in range(CHROM_LEN):
            temp.append(random.randint(0, 1))
        self.chrom = "".join([str(t) for t in temp])
        self.fitness = 0

    # 计算个体适应度
    def get_fitness(self, lb, ub):
        x = decode(self.chrom, lb, ub)
        value = -F(x) + CMIN
        self.fitness = max(0, value)
        return self.fitness

    def __str__(self):
        return "chrom:{}, fitness:{}".format(self.chrom, self.fitness)


# 获得当代最佳和最差个体索引
def best_and_worst(population):
    # 最佳个体索引
    best_idx = 0
    # 最差个体索引
    worst_idx = 0
    for _idx, p in enumerate(population):
        if p.fitness > population[best_idx].fitness:
            best_idx = _idx
        elif p.fitness < population[worst_idx].fitness:
            worst_idx = _idx
    return best_idx, worst_idx


# 选择（复制）操作
def select(population):
    # 新种群
    new_pop = []
    # 当代个体适应度总和
    fitness_sum = max(sum([i.fitness for i in population]), 0.0001)
    # 当代个体累计适应度占比
    cfitness = []
    # 计算相对适应度占比
    for j in range(POP_SIZE):
        cfitness.append(population[j].fitness / fitness_sum)
    # 计算累计适应度占比
    for j in range(POP_SIZE):
        if j == 0:
            continue
        cfitness[j] = cfitness[j - 1] + cfitness[j]
    # 依据累计适应度占比进行选择复制，随机数大于对应的累计适应度占比，则进行复制
    for k in range(POP_SIZE):
        index = 0
        while random.random() > cfitness[index]:
            index += 1
            # 若无法找到要复制的其他个体，则沿用当前个体
            if index >= POP_SIZE:
                index = k
                break
        new_pop.append(copy.deepcopy(population[index]))
    return new_pop


# 交叉操作
def crossover(population):
    # 随机产生个体配对索引，类似于洗牌的效果
    index = [i for i in range(POP_SIZE)]
    for i in range(POP_SIZE):
        point = random.randint(0, POP_SIZE - i - 1)
        temp = index[i]
        index[i] = index[point + i]
        index[point + i] = temp

    for i in range(0, POP_SIZE, 2):
        if random.random() > PC:
            # 随机选择交叉开始位置
            cross_start = random.randint(0, CHROM_LEN - 2) + 1
            # 需要交换的基因
            cross_gene1 = population[index[i]].chrom[cross_start:]
            cross_gene2 = population[index[i + 1]].chrom[cross_start:]
            # 交叉操作
            population[index[i]].chrom = population[index[i]].chrom[0: cross_start] + cross_gene2
            population[index[i + 1]].chrom = population[index[i + 1]].chrom[0: cross_start] + cross_gene1


# 变异操作
def mutation(population):
    for individual in population:
        # 初始化新染色体
        new_chrom_ch = [c for c in individual.chrom]
        for i in range(CHROM_LEN):
            # 随机数小于变异概率，则进行变异操作
            if random.random() < PM:
                new_chrom_ch[i] = "1" if individual.chrom[i] is "0" else "0"
        # 更新染色体
        individual.chrom = "".join(new_chrom_ch)


# 绘制结果
def draw_result(best):
    import numpy as np
    # 绘制优化函数
    x = np.linspace(1, 2, 100)
    y = [F(_x) for _x in x]
    plt.plot(x, y)
    # 绘制最优解
    best_x = decode(best.chrom, 1, 2)
    best_y = F(decode(best.chrom, 1, 2))
    plt.scatter(best_x, best_y, s=100, c='red', marker='*', zorder=2)
    plt.show()
    # plt.savefig('sga_result.png', dpi=800)


# 绘制进化过程
def draw_evolution(evolution):
    x = [i for i in range(len(evolution))]
    plt.plot(x, evolution)
    plt.show()
    # plt.savefig('sga_evolution.png', dpi=800)


def main():
    # 种群
    population = []
    # 下界
    lb = 0
    # 上界
    ub = 9
    # 初始化种群
    for _ in range(POP_SIZE):
        population.append(Individual())
    # 计算初始种群适应度
    for individual in population:
        individual.get_fitness(lb, ub)
    # 初始种群最佳和最差个体
    best_idx, worst_idx = best_and_worst(population)
    # 历史最佳个体
    current_best = population[best_idx]
    # 进化过程，每一代的最佳个体的函数值
    evolution = []
    # 循环直到最大代数
    for generation in range(MAX_GENERATION):
        # 选择复制
        population = select(population)
        # 交叉
        crossover(population)
        # 变异
        mutation(population)
        # 重新计算适应度
        for individual in population:
            individual.get_fitness(lb, ub)
        # 当代种群最佳和最差个体索引
        best_idx, worst_idx = best_and_worst(population)
        # 利用精英模型执行进化操作，用历史最佳个体代替当代的最差个体
        if population[best_idx].fitness > current_best.fitness:
            current_best = population[best_idx]
        else:
            population[worst_idx] = current_best
        # 更新进化过程
        evolution.append(round(F(decode(current_best.chrom, lb, ub)), 4))

    # 绘制进化过程
    # draw_evolution(evolution)
    # 绘制结果
    draw_result(current_best)
    # 打印最佳结果
    print("X = {}".format(round(decode(current_best.chrom, lb, ub), 4)))
    print("Y = {}".format(round(F(decode(current_best.chrom, lb, ub)), 4)))


if __name__ == "__main__":
    main()
