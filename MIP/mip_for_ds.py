"""
烟草行业卷包智能排产混合整数规划模型
支持硬约束、软约束及多目标策略
"""

import pulp
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


# -------------------- 数据定义 --------------------
@dataclass
class Machine:
    idx: int
    name: str
    init_brand: int  # 初始生产牌号
    # 其他属性可扩展


@dataclass
class Brand:
    idx: int
    name: str


@dataclass
class Material:
    idx: int
    name: str
    initial_inventory: float


@dataclass
class Order:
    idx: int
    brand: int
    quantity: float
    deadline: int  # 最晚允许生产的时段索引 (0-based)
    priority: int  # 1-高，2-中，3-低


@dataclass
class Period:
    idx: int
    day: int  # 所属天
    is_peak: bool  # 是否高电价时段
    length: float  # 时段长度 (分钟)


# -------------------- 模型构建器 --------------------
class TobaccoSchedulingModel:
    def __init__(self,
                 machines: List[Machine],
                 brands: List[Brand],
                 materials: List[Material],
                 orders: List[Order],
                 periods: List[Period],
                 speed: Dict[Tuple[int, int], float],  # (m,i) -> 件/分钟
                 setup_time: Dict[Tuple[int, int, int], float],  # (m,j,i) -> 分钟
                 bom: Dict[Tuple[int, int], float],  # (material, brand) -> 每件消耗量
                 machine_available: Dict[Tuple[int, int], bool] = None,  # (m,t) -> bool
                 allowed_brands: Dict[Tuple[int, int], bool] = None,  # (m,i) -> bool
                 # 软约束参数
                 max_switch_per_day: int = 2,
                 inventory_target: Dict[int, Tuple[float, float]] = None,  # (i) -> (min, max)
                 # 策略权重 (根据策略动态设置)
                 weight_delivery: float = 1.0,
                 weight_switch: float = 1.0,
                 weight_efficiency: float = 1.0,
                 weight_energy: float = 1.0,
                 weight_inventory_balance: float = 1.0,
                 weight_balance_load: float = 1.0):

        # 集合
        self.M = len(machines)
        self.I = len(brands)
        self.T = len(periods)
        self.D = max(p.day for p in periods) + 1  # 天数
        self.orders = orders
        self.periods = periods
        self.machines = machines
        self.brands = brands
        self.materials = materials

        # 参数
        self.speed = speed
        self.setup_time = setup_time
        self.bom = bom
        self.machine_available = machine_available or {(m, i): True for m in range(self.M) for i in range(self.T)}
        self.allowed_brands = allowed_brands or {(m, i): True for m in range(self.M) for i in range(self.I)}

        # 预处理：每天时段索引
        self.day_periods = defaultdict(list)
        for t, p in enumerate(periods):
            self.day_periods[p.day].append(t)

        # 软约束参数
        self.max_switch_per_day = max_switch_per_day
        self.inventory_target = inventory_target

        # 策略权重
        self.w_delivery = weight_delivery
        self.w_switch = weight_switch
        self.w_efficiency = weight_efficiency
        self.w_energy = weight_energy
        self.w_inv_bal = weight_inventory_balance
        self.w_balance = weight_balance_load

        # 大M
        self.bigM = {
            (m, i, t): self.speed.get((m, i), 0) * periods[t].length
            for m in range(self.M) for i in range(self.I) for t in range(self.T)
            if self.allowed_brands.get((m, i), False) and self.machine_available.get((m, t), False)
        }

        # 订单相关：每个品牌总需求量按时段分解（使用订单交期限制）
        # 需求按品牌聚合至时段，仅允许在交期之前生产
        self.demand = defaultdict(float)  # (i,t)
        self.order_brand_list = defaultdict(list)
        for o in orders:
            # 假设订单需求可以分配到 ≤ deadline 的任意时段（由模型决定）
            # 此处我们只记录品牌-时段是否有需求能力，具体分配由模型决定
            self.order_brand_list[o.brand].append(o)

        self.problem = pulp.LpProblem("Tobacco_Scheduling", pulp.LpMinimize)
        self._build_variables()
        self._build_constraints()
        self._build_objective()

    def _build_variables(self):
        M, I, T = self.M, self.I, self.T
        # 生产量
        self.x = pulp.LpVariable.dicts("x", (range(M), range(I), range(T)), lowBound=0, cat='Continuous')
        # 指派
        self.y = pulp.LpVariable.dicts("y", (range(M), range(I), range(T)), cat='Binary')
        # 换牌指示
        self.z = pulp.LpVariable.dicts("z", (range(M), range(I), range(I), range(T)), cat='Binary')
        # 库存 (时段末)
        self.inv = pulp.LpVariable.dicts("inv", (range(I), range(T + 1)), lowBound=0, cat='Continuous')
        # 物料消耗库存 (时段末)
        self.mat_inv = pulp.LpVariable.dicts("mat_inv", (range(len(self.materials)), range(T + 1)), lowBound=0,
                                             cat='Continuous')
        # 每日换牌次数
        self.daily_switch = pulp.LpVariable.dicts("daily_switch", (range(M), range(self.D)), lowBound=0, cat='Integer')
        # 超过最大换牌次数的正偏差
        self.switch_slack = pulp.LpVariable.dicts("switch_slack", (range(M), range(self.D)), lowBound=0,
                                                  cat='Continuous')
        # 库存超出目标的偏差
        self.inv_over = pulp.LpVariable.dicts("inv_over", (range(I), range(T + 1)), lowBound=0, cat='Continuous')
        self.inv_under = pulp.LpVariable.dicts("inv_under", (range(I), range(T + 1)), lowBound=0, cat='Continuous')
        # 设备闲置时间（用于均衡负荷）
        self.idle_time = pulp.LpVariable.dicts("idle_time", (range(M), range(T)), lowBound=0, cat='Continuous')
        # 最大闲置时间（用于均衡负荷 min-max）
        self.max_idle = pulp.LpVariable("max_idle", lowBound=0, cat='Continuous')
        # 订单延迟指标（若交期后生产则惩罚）
        # 简化：交期之后不允许生产，所以不产生延迟，但优先级可通过权重体现在目标中
        # 可增加订单完成时间变量，这里省略，改用需求优先级加权

    def _build_constraints(self):
        M, I, T = self.M, self.I, self.T
        D = self.D
        periods = self.periods

        # ========== 硬约束 ==========
        # 1. 单次指派
        for m in range(M):
            for t in range(T):
                if self.machine_available.get((m, t), False):
                    self.problem += pulp.lpSum(
                        self.y[m][i][t] for i in range(I) if self.allowed_brands.get((m, i), False)) <= 1
                else:
                    # 不可用时段所有指派为0
                    for i in range(I):
                        self.problem += self.y[m][i][t] == 0

        # 2. 产量与指派关联，且遵守产能上限
        for m in range(M):
            for i in range(I):
                if not self.allowed_brands.get((m, i), False):
                    for t in range(T):
                        self.problem += self.x[m][i][t] == 0
                        self.problem += self.y[m][i][t] == 0
                    continue
                for t in range(T):
                    if not self.machine_available.get((m, t), False):
                        self.problem += self.x[m][i][t] == 0
                        continue
                    # 产量不能超过最大产能 (速率 * 可用时长)
                    max_prod = self.bigM.get((m, i, t), 0)
                    self.problem += self.x[m][i][t] <= max_prod * self.y[m][i][t]

        # 3. 时间容量约束 (生产时间 + 换牌时间 <= 可用时长)
        for m in range(M):
            for t in range(T):
                if not self.machine_available.get((m, t), False):
                    continue
                prod_time = pulp.lpSum(self.x[m][i][t] / self.speed[(m, i)]
                                       for i in range(I) if self.allowed_brands.get((m, i), False))
                setup_time_total = pulp.lpSum(self.setup_time.get((m, j, i), 0) * self.z[m][j][i][t]
                                              for j in range(I) for i in range(I)
                                              if j != i and self.allowed_brands.get((m, j),
                                                                                    False) and self.allowed_brands.get(
                    (m, i), False))
                self.problem += prod_time + setup_time_total <= periods[t].length

                # 记录闲置时间
                idle_expr = periods[t].length - prod_time - setup_time_total
                self.problem += self.idle_time[m][t] == idle_expr

        # 4. 换牌逻辑
        # 构建 y0
        y0 = {}
        for m, mach in enumerate(self.machines):
            for i in range(I):
                y0[m, i] = 1 if i == mach.init_brand else 0

        for m in range(M):
            for t in range(T):
                if not self.machine_available.get((m, t), False):
                    for j in range(I):
                        for i in range(I):
                            self.problem += self.z[m][j][i][t] == 0
                    continue
                for j in range(I):
                    if not self.allowed_brands.get((m, j), False):
                        continue
                    for i in range(I):
                        if not self.allowed_brands.get((m, i), False):
                            continue
                        if j == i:
                            self.problem += self.z[m][j][i][t] == 0  # 同品牌不视为换牌
                            continue
                        y_prev = y0[m, j] if t == 0 else self.y[m][j][t - 1]
                        self.problem += self.z[m][j][i][t] >= y_prev + self.y[m][i][t] - 1
                        self.problem += self.z[m][j][i][t] <= y_prev
                        self.problem += self.z[m][j][i][t] <= self.y[m][i][t]

        # 5. 每日换牌次数统计
        for m in range(M):
            for d in range(D):
                day_switches = []
                for t in self.day_periods[d]:
                    if not self.machine_available.get((m, t), False):
                        continue
                    # 每个时段换牌总数
                    day_switches.append(pulp.lpSum(self.z[m][j][i][t]
                                                   for j in range(I) for i in range(I) if j != i))
                if day_switches:
                    self.problem += self.daily_switch[m][d] == pulp.lpSum(day_switches)
                else:
                    self.problem += self.daily_switch[m][d] == 0
                # 软约束松弛
                self.problem += self.daily_switch[m][d] - self.switch_slack[m][d] <= self.max_switch_per_day

        # 6. 库存平衡（基于订单需求分配）
        # 需求分配：每个订单的需求必须被满足，且只能在交期及之前生产
        # 引入变量 order_production[o][t] 订单o在时段t的产量，需满足求和=订单量
        # 并限制 t <= deadline
        self.order_x = {}
        for o in self.orders:
            self.order_x[o.idx] = {}
            for t in range(T):
                if t <= o.deadline:
                    self.order_x[o.idx][t] = pulp.LpVariable(f"order_x_{o.idx}_{t}", lowBound=0, cat='Continuous')
                else:
                    self.order_x[o.idx][t] = None
            self.problem += pulp.lpSum(self.order_x[o.idx][t] for t in range(T) if t <= o.deadline) == o.quantity

        # 品牌总产量 = 该品牌各订单分配量之和
        for i in range(I):
            for t in range(T):
                total_prod = pulp.lpSum(self.x[m][i][t] for m in range(M) if self.allowed_brands.get((m, i), False))
                order_prod_sum = pulp.lpSum(self.order_x[o.idx][t] for o in self.order_brand_list[i] if t <= o.deadline)
                self.problem += total_prod == order_prod_sum

        # 库存递推
        for i in range(I):
            self.problem += self.inv[i][0] == 0  # 假设初始库存为0，可改为参数
            for t in range(1, T + 1):
                prod_t = pulp.lpSum(self.x[m][i][t - 1] for m in range(M) if self.allowed_brands.get((m, i), False))
                # 需求 = 该时段需满足的订单总产量（即分配给该时段的订单量）
                demand_t = pulp.lpSum(
                    self.order_x[o.idx][t - 1] for o in self.order_brand_list[i] if t - 1 <= o.deadline)
                self.problem += self.inv[i][t] == self.inv[i][t - 1] + prod_t - demand_t

        # 7. 物料约束
        for mat in range(len(self.materials)):
            self.problem += self.mat_inv[mat][0] == self.materials[mat].initial_inventory
            for t in range(1, T + 1):
                consumption = pulp.lpSum(
                    self.bom.get((mat, i), 0) * self.x[m][i][t - 1]
                    for m in range(M) for i in range(I)
                )
                self.problem += self.mat_inv[mat][t] == self.mat_inv[mat][t - 1] - consumption

        # 8. 库存目标偏离
        if self.inventory_target:
            for i in range(I):
                if i in self.inventory_target:
                    low, high = self.inventory_target[i]
                    for t in range(1, T + 1):
                        self.problem += self.inv[i][t] - self.inv_over[i][t] + self.inv_under[i][t] == (
                                    low + high) / 2  # 以中点为目标
                        self.problem += self.inv_over[i][t] >= self.inv[i][t] - high
                        self.problem += self.inv_under[i][t] >= low - self.inv[i][t]

        # 9. 均衡负荷：计算最大闲置时间
        for m in range(M):
            for t in range(T):
                self.problem += self.max_idle >= self.idle_time[m][t]

    def _build_objective(self):
        M, I, T = self.M, self.I, self.T
        # 1. 交期保证：订单优先级加权（高优先级订单尽量安排在早期时段）
        delivery_cost = 0
        for o in self.orders:
            w = {1: 10, 2: 3, 3: 1}.get(o.priority, 1)
            for t in range(T):
                if t <= o.deadline:
                    # 越晚生产惩罚越大
                    delivery_cost += w * (t + 1) * self.order_x[o.idx][t]
        self.problem += self.w_delivery * delivery_cost

        # 2. 换牌成本
        switch_cost = pulp.lpSum(self.setup_time.get((m, j, i), 0) * self.z[m][j][i][t]
                                 for m in range(M) for j in range(I) for i in range(I) for t in range(T) if j != i)
        # 每日换牌超额惩罚
        switch_penalty = pulp.lpSum(self.switch_slack[m][d] for m in range(M) for d in range(self.D))
        self.problem += self.w_switch * (switch_cost + 100 * switch_penalty)

        # 3. 效率优先：最小化闲置时间
        total_idle = pulp.lpSum(self.idle_time[m][t] for m in range(M) for t in range(T))
        self.problem += self.w_efficiency * total_idle

        # 4. 节能策略：高电价时段惩罚生产活动
        energy_cost = 0
        for t in range(T):
            if self.periods[t].is_peak:
                factor = 1.5
            else:
                factor = 0.5
            energy_cost += factor * pulp.lpSum(self.x[m][i][t] for m in range(M) for i in range(I))
        self.problem += self.w_energy * energy_cost

        # 5. 库存平衡偏好
        inv_balance_cost = pulp.lpSum(self.inv_over[i][t] + self.inv_under[i][t]
                                      for i in range(I) for t in range(1, T + 1)
                                      if self.inventory_target and i in self.inventory_target)
        self.problem += self.w_inv_bal * inv_balance_cost

        # 6. 均衡生产：最小化最大闲置时间 (min-max)
        self.problem += self.w_balance * self.max_idle

    def solve(self, time_limit=60, msg=True):
        self.problem.solve(pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit))
        return pulp.LpStatus[self.problem.status]

    def print_solution(self):
        print("求解状态:", pulp.LpStatus[self.problem.status])
        print("目标值:", pulp.value(self.problem.objective))
        print("\n========== 生产计划 ==========")
        for m in range(self.M):
            print(f"机组 {m}:")
            for t in range(self.T):
                for i in range(self.I):
                    if pulp.value(self.x[m][i][t]) > 1e-4:
                        print(f"  时段{t}(第{self.periods[t].day}天): 品牌{i} 产量 {pulp.value(self.x[m][i][t]):.2f}")
        print("\n========== 订单分配 ==========")
        for o in self.orders:
            for t in range(T):
                if t <= o.deadline and pulp.value(self.order_x[o.idx][t]) > 1e-4:
                    print(f"订单{o.idx}(品牌{o.brand}) 时段{t}: {pulp.value(self.order_x[o.idx][t]):.2f}")
        print("\n========== 库存轨迹 ==========")
        for i in range(self.I):
            seq = [pulp.value(self.inv[i][t]) for t in range(self.T + 1)]
            print(f"品牌{i}: {seq}")


# -------------------- 示例运行与策略配置 --------------------
if __name__ == "__main__":
    # 生成小规模测试数据
    machines = [Machine(0, "M1", 0), Machine(1, "M2", 1), Machine(2, "M3", 2)]
    brands = [Brand(0, "A"), Brand(1, "B"), Brand(2, "C")]
    materials = [Material(0, "烟叶", 5000), Material(1, "滤棒", 3000)]

    # 时段: 2天，每天3个时段 (早、中、晚)，晚高峰电价高
    periods = [
        Period(0, 0, False, 240), Period(1, 0, False, 240), Period(2, 0, True, 240),
        Period(3, 1, False, 240), Period(4, 1, False, 240), Period(5, 1, True, 240)
    ]

    # 产能参数
    speed = {
        (0, 0): 5.0, (0, 1): 4.5, (0, 2): 4.0,
        (1, 0): 4.8, (1, 1): 5.2, (1, 2): 4.3,
        (2, 0): 4.0, (2, 1): 4.0, (2, 2): 5.0
    }
    setup_time = {}
    for m in range(3):
        for j in range(3):
            for i in range(3):
                if j == i:
                    setup_time[(m, j, i)] = 0
                else:
                    setup_time[(m, j, i)] = np.random.uniform(20, 60)

    # 物料清单: 品牌对物料消耗
    bom = {(0, 0): 1.2, (0, 1): 1.0, (0, 2): 0.9,
           (1, 0): 0.8, (1, 1): 1.1, (1, 2): 1.0}

    # 订单 (品牌, 数量, 交期时段, 优先级)
    orders = [
        Order(0, 0, 200, 3, 1),  # 高优先，较早交期
        Order(1, 1, 180, 5, 2),
        Order(2, 2, 150, 5, 1),
        Order(3, 0, 100, 5, 3)
    ]

    # 设备可用性：全部可用
    machine_available = None
    # 品牌兼容性：全部兼容
    allowed_brands = None

    # 库存目标 (min, max)
    inventory_target = {0: (20, 80), 1: (10, 60), 2: (15, 50)}

    # ===== 策略定义：通过权重调整 =====
    strategies = {
        "交期优先": {"w_del": 100, "w_switch": 1, "w_eff": 1, "w_energy": 1, "w_inv_bal": 1, "w_balance": 1},
        "成本优先": {"w_del": 1, "w_switch": 50, "w_eff": 5, "w_energy": 20, "w_inv_bal": 10, "w_balance": 1},
        "效率优先": {"w_del": 1, "w_switch": 1, "w_eff": 100, "w_energy": 1, "w_inv_bal": 1, "w_balance": 20},
        "均衡生产": {"w_del": 1, "w_switch": 1, "w_eff": 1, "w_energy": 1, "w_inv_bal": 1, "w_balance": 100},
        "节能策略": {"w_del": 1, "w_switch": 1, "w_eff": 1, "w_energy": 100, "w_inv_bal": 1, "w_balance": 1},
    }

    # 选择策略运行
    chosen = "均衡生产"
    params = strategies[chosen]
    model = TobaccoSchedulingModel(
        machines, brands, materials, orders, periods,
        speed, setup_time, bom,
        inventory_target=inventory_target,
        weight_delivery=params["w_del"],
        weight_switch=params["w_switch"],
        weight_efficiency=params["w_eff"],
        weight_energy=params["w_energy"],
        weight_inventory_balance=params["w_inv_bal"],
        weight_balance_load=params["w_balance"]
    )

    status = model.solve(time_limit=30, msg=False)
    print(f"执行策略: {chosen}")
    model.print_solution()