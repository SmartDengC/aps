import random
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


# 参数配置
class Config:
    NUM_MACHINES = 4  # 设备数量
    NUM_BRANDS = 8  # 品牌数量
    MONTH_WORKING_HOURS = 30 * 24  # 月工作小时数 (720小时)
    CHANGE_OVER_TIME = 2  # 换牌时间(小时)
    PRODUCTION_RATE = 120  # 设备生产率(箱/小时)
    POPULATION_SIZE = 100  # 种群大小
    MAX_GENERATIONS = 200  # 最大迭代次数
    MUTATION_RATE = 0.15  # 变异率
    ELITE_RATIO = 0.1  # 精英保留比例
    DEMAND = np.array([  # 品牌需求(箱)
        12000, 18000, 15000, 20000,
        16000, 22000, 19000, 25000
    ])
    BRAND_NAMES = [  # 品牌名称
        "BrandA", "BrandB", "BrandC", "BrandD",
        "BrandE", "BrandF", "BrandG", "BrandH"
    ]


# 染色体表示: [设备1序列, 设备2序列, ...]
class Chromosome:
    def __init__(self, genes: List[List[int]]):
        self.genes = genes  # 基因: 每个设备的品牌生产序列
        self.fitness = 0.0  # 适应度值
        self.total_changeover = 0  # 总换牌时间
        self.makespan = 0  # 最大完成时间
        self.utilization = np.zeros(Config.NUM_MACHINES)  # 设备利用率

    def __repr__(self):
        return f"Fit={self.fitness:.2f}, Changeover={self.total_changeover}, Makespan={self.makespan}"


# 遗传算法排产
class ProductionScheduler:
    def __init__(self):
        self.population = []
        self.best_chromosome = None

    def initialize_population(self) -> None:
        """初始化种群"""
        self.population = []
        for _ in range(Config.POPULATION_SIZE):
            genes = []
            remaining_demand = Config.DEMAND.copy()
            brand_list = list(range(Config.NUM_BRANDS))

            # 为每台设备分配生产序列
            for machine_idx in range(Config.NUM_MACHINES):
                seq = []
                # 随机选择品牌，但确保有需求
                valid_brands = [b for b in brand_list if remaining_demand[b] > 0]

                # 即使没有有效品牌也创建空序列，确保每台机器都有对应序列
                if valid_brands:
                    # 创建该设备的品牌序列
                    while valid_brands:
                        brand = random.choice(valid_brands)
                        seq.append(brand)
                        valid_brands.remove(brand)
                        # 减少该品牌需求(仅表示分配，实际生产量在解码时计算)
                        remaining_demand[brand] = max(0, remaining_demand[brand] -
                                                      Config.PRODUCTION_RATE * Config.MONTH_WORKING_HOURS // Config.NUM_MACHINES)
                genes.append(seq)
            self.population.append(Chromosome(genes))

    def decode_schedule(self, chrom: Chromosome) -> Tuple[float, float, np.ndarray]:
        """解码染色体，计算适应度相关指标"""
        total_changeover = 0
        machine_times = np.zeros(Config.NUM_MACHINES)
        machine_utilization = np.zeros(Config.NUM_MACHINES)
        produced = np.zeros(Config.NUM_BRANDS)

        # 计算每台设备的生产时间和换牌时间
        for machine, seq in enumerate(chrom.genes):
            if not seq:
                continue

            # 第一个品牌的生产时间
            brand = seq[0]
            production_time = min(Config.DEMAND[brand],
                                  Config.PRODUCTION_RATE * Config.MONTH_WORKING_HOURS) / Config.PRODUCTION_RATE
            machine_time = production_time
            produced[brand] += production_time * Config.PRODUCTION_RATE

            # 后续品牌
            last_brand = brand
            for brand in seq[1:]:
                # 换牌时间
                if brand != last_brand:
                    machine_time += Config.CHANGE_OVER_TIME
                    total_changeover += Config.CHANGE_OVER_TIME

                # 生产时间
                remaining_demand = max(0, Config.DEMAND[brand] - produced[brand])
                production_time = min(remaining_demand,
                                      Config.PRODUCTION_RATE * (
                                              Config.MONTH_WORKING_HOURS - machine_time)) / Config.PRODUCTION_RATE
                machine_time += production_time
                produced[brand] += production_time * Config.PRODUCTION_RATE
                last_brand = brand

            machine_times[machine] = machine_time
            machine_utilization[machine] = machine_time / Config.MONTH_WORKING_HOURS

        # 计算未完成量惩罚
        unfulfilled = np.sum(np.maximum(0, Config.DEMAND - produced))
        penalty = unfulfilled / 1000  # 惩罚系数

        # 计算最大完成时间
        makespan = np.max(machine_times)

        # 平衡负载惩罚(标准差)
        load_balance_penalty = np.std(machine_utilization) * 50

        # 适应度 = 换牌时间 + 最大完成时间 + 未完成惩罚 + 负载不平衡惩罚
        fitness = (total_changeover + makespan + penalty + load_balance_penalty)

        # 保存计算结果
        chrom.total_changeover = total_changeover
        chrom.makespan = makespan
        chrom.utilization = machine_utilization

        return fitness, unfulfilled, load_balance_penalty

    def evaluate_population(self) -> None:
        """评估种群中每个个体的适应度"""
        for chrom in self.population:
            fitness, _, _ = self.decode_schedule(chrom)
            chrom.fitness = 1 / fitness  # 最小化问题转换为最大化适应度

    def select_parents(self) -> List[Chromosome]:
        """锦标赛选择父代"""
        parents = []
        tournament_size = max(2, Config.POPULATION_SIZE // 10)

        for _ in range(Config.POPULATION_SIZE):
            tournament = random.sample(self.population, tournament_size)
            winner = max(tournament, key=lambda x: x.fitness)
            parents.append(winner)

        return parents

    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> Chromosome:
        """顺序交叉(OX)"""
        child_genes = []

        for machine in range(Config.NUM_MACHINES):
            seq1 = parent1.genes[machine]
            seq2 = parent2.genes[machine]

            if not seq1 or not seq2:
                child_genes.append(seq1 or seq2)
                continue

            # 选择交叉点
            start = random.randint(0, len(seq1) - 1)
            end = random.randint(start + 1, len(seq1))

            # 创建子代序列
            child_seq = [-1] * len(seq1)
            # 复制父代1的片段
            child_seq[start:end] = seq1[start:end]

            # 从父代2填充剩余位置
            ptr = 0
            for i in range(len(seq2)):
                if seq2[i] not in child_seq:
                    while ptr < len(child_seq) and child_seq[ptr] != -1:
                        ptr += 1
                    if ptr >= len(child_seq):
                        break
                    child_seq[ptr] = seq2[i]

            child_genes.append(child_seq)

        return Chromosome(child_genes)

    def mutate(self, chrom: Chromosome) -> Chromosome:
        """交换变异"""
        new_genes = [seq[:] for seq in chrom.genes]

        for machine_seq in new_genes:
            if len(machine_seq) > 1 and random.random() < Config.MUTATION_RATE:
                # 随机选择两个位置交换
                i, j = random.sample(range(len(machine_seq)), 2)
                machine_seq[i], machine_seq[j] = machine_seq[j], machine_seq[i]

        return Chromosome(new_genes)

    def create_new_generation(self, parents: List[Chromosome]) -> None:
        """创建新一代种群"""
        # 保留精英
        elite_size = int(Config.POPULATION_SIZE * Config.ELITE_RATIO)
        elite = sorted(self.population, key=lambda x: x.fitness, reverse=True)[:elite_size]

        new_population = elite.copy()

        # 生成后代
        while len(new_population) < Config.POPULATION_SIZE:
            parent1, parent2 = random.sample(parents, 2)
            child = self.crossover(parent1, parent2)

            if random.random() < Config.MUTATION_RATE:
                child = self.mutate(child)

            new_population.append(child)

        self.population = new_population

    def optimize(self) -> Chromosome:
        """执行优化"""
        self.initialize_population()
        self.evaluate_population()
        best_fitness = -float('inf')

        # 记录收敛过程
        fitness_history = []
        changeover_history = []
        makespan_history = []

        for gen in range(Config.MAX_GENERATIONS):
            parents = self.select_parents()
            self.create_new_generation(parents)
            self.evaluate_population()

            # 更新最佳染色体
            current_best = max(self.population, key=lambda x: x.fitness)
            if current_best.fitness > best_fitness:
                best_fitness = current_best.fitness
                self.best_chromosome = current_best

            # 记录统计信息
            fitness_history.append(1 / current_best.fitness)  # 记录实际目标函数值
            changeover_history.append(current_best.total_changeover)
            makespan_history.append(current_best.makespan)

            if gen % 20 == 0:
                print(f"Generation {gen}: Best Fitness={1 / current_best.fitness:.2f}, "
                      f"Changeover={current_best.total_changeover}, Makespan={current_best.makespan:.1f}")

        # 绘制收敛曲线
        plt.figure(figsize=(12, 8))
        plt.subplot(3, 1, 1)
        plt.plot(fitness_history, 'b-')
        plt.title('Optimization Convergence')
        plt.ylabel('Total Cost')

        plt.subplot(3, 1, 2)
        plt.plot(changeover_history, 'g-')
        plt.ylabel('Changeover Time')

        plt.subplot(3, 1, 3)
        plt.plot(makespan_history, 'r-')
        plt.ylabel('Makespan')
        plt.xlabel('Generations')

        plt.tight_layout()
        plt.savefig('convergence.png')
        plt.show()

        return self.best_chromosome

    def print_schedule(self, chrom: Chromosome) -> None:
        """打印生产计划"""
        print("\n" + "=" * 60)
        print("Optimal Production Schedule")
        print("=" * 60)

        total_production = np.zeros(Config.NUM_BRANDS)

        for machine, seq in enumerate(chrom.genes):
            print(f"\nMachine {machine + 1} Schedule (Utilization: {chrom.utilization[machine] * 100:.1f}%):")
            if not seq:
                print("  No production assigned")
                continue

            current_time = 0
            last_brand = None

            for brand in seq:
                # 换牌时间
                if last_brand is not None and brand != last_brand:
                    print(f"  Changeover: {Config.CHANGE_OVER_TIME} hours")
                    current_time += Config.CHANGE_OVER_TIME

                # 计算生产时间
                remaining_demand = max(0, Config.DEMAND[brand] - total_production[brand])
                production_time = min(remaining_demand,
                                      Config.PRODUCTION_RATE * (
                                              Config.MONTH_WORKING_HOURS - current_time)) / Config.PRODUCTION_RATE
                production_qty = production_time * Config.PRODUCTION_RATE
                total_production[brand] += production_qty
                current_time += production_time

                print(f"  {Config.BRAND_NAMES[brand]}: {production_qty:.0f} cases "
                      f"({production_time:.1f} hours)")

                last_brand = brand

        print("\n" + "-" * 60)
        print("Production Summary:")
        for brand in range(Config.NUM_BRANDS):
            fulfillment = total_production[brand] / Config.DEMAND[brand] * 100
            print(f"{Config.BRAND_NAMES[brand]}: {total_production[brand]:.0f}/"
                  f"{Config.DEMAND[brand]} ({fulfillment:.1f}%)")

        print(f"\nTotal Changeover Time: {chrom.total_changeover} hours")
        print(f"Makespan: {chrom.makespan:.1f} hours")
        print("=" * 60)


# 主函数
if __name__ == "__main__":
    scheduler = ProductionScheduler()
    best_schedule = scheduler.optimize()
    scheduler.print_schedule(best_schedule)
