#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file test_job_scheduling_ortools.py
@brief 任务调度问题求解示例 - OR-Tools CP-SAT版本
@details 用于验证 CPlab 的任务调度问题求解结果

问题描述：
- 机器: M1, M2
- 任务: J1, J2, J3
- 加工工时:
  J1: M1=3, M2=2
  J2: M1=2, M2=4
  J3: M1=4, M2=3
- 约束：
  1. 每个任务都需要在 M1 和 M2 上加工
  2. 任务 J2 要先在 M1 加工，再在 M2 加工（flow shop）
  3. 同一时刻一台机器只能加工一个任务
  4. 任意两个任务之间需要有一定的休息间隔 5
- 目标: 最小化所有任务完成时间的最大值（makespan）

@example test_job_scheduling_ortools.py
"""
try:
    from ortools.sat.python import cp_model
except Exception as e:
    print("ortools 未安装或不可用。请先运行: pip install ortools")
    print("错误:", e)
    raise

# 任务在机器上的加工时间（与 CPlab 中的定义一致）
DURATIONS = [
    [3, 2],  # J1: M1=3, M2=2
    [2, 4],  # J2: M1=2, M2=4
    [4, 3]   # J3: M1=4, M2=3
]
BUFFER = 5  # 任务之间的休息间隔
NUM_JOBS = 3
NUM_MACHINES = 2

def print_solution(m1_starts, m2_starts, makespan):
    """
    @brief 打印解
    @param m1_starts 任务在M1上的开始时间列表
    @param m2_starts 任务在M2上的开始时间列表
    @param makespan 最大完成时间
    """
    print("\n========== 任务调度方案 ==========")
    print("任务 | M1开始 | M1结束 | M2开始 | M2结束 ")
    print("-----|--------|--------|--------|--------|----------")
    for i in range(NUM_JOBS):
        m1_start = m1_starts[i]
        m1_end = m1_start + DURATIONS[i][0]
        m2_start = m2_starts[i]
        m2_end = m2_start + DURATIONS[i][1]
        print(f"J{i+1}  |{m1_start:7d} |{m1_end:7d} |{m2_start:7d} |{m2_end:7d}")
    print(f"\nMakespan (最大完成时间): {makespan}")

def solve_job_scheduling():
    """
    @brief 求解任务调度问题
    """
    print("==========================================")
    print("任务调度问题求解 - OR-Tools CP-SAT版本")
    print("==========================================\n")
    print("问题参数:")
    print(f"- 机器数: {NUM_MACHINES} (M1, M2)")
    print(f"- 任务数: {NUM_JOBS} (J1, J2, J3)")
    print("- 任务加工时间:")
    for i in range(NUM_JOBS):
        print(f"  J{i+1}: M1={DURATIONS[i][0]}, M2={DURATIONS[i][1]}")
    print(f"- 任务间隔: {BUFFER}\n")
    # 构建模型
    model = cp_model.CpModel()
    # 使用interval变量建模：每个任务在每台机器上的加工区间
    # interval变量自动包含开始时间、长度和结束时间
    m1_intervals = {}
    m2_intervals = {}
    m1_starts = {}
    m1_ends = {}
    m2_starts = {}
    m2_ends = {}
    # 定义时间范围的上界
    horizon = 1000
    for i in range(NUM_JOBS):
        # 创建任务i在M1上的开始时间变量
        m1_start = model.NewIntVar(0, horizon, f"M1_Start_J{i+1}")
        m1_end = model.NewIntVar(0, horizon, f"M1_End_J{i+1}")
        m1_starts[i] = m1_start
        m1_ends[i] = m1_end
        # 创建任务i在M1上的interval变量
        m1_intervals[i] = model.NewIntervalVar(
            m1_start, DURATIONS[i][0], m1_end, f"M1_Interval_J{i+1}"
        )
        # 创建任务i在M2上的开始时间变量
        m2_start = model.NewIntVar(0, horizon, f"M2_Start_J{i+1}")
        m2_end = model.NewIntVar(0, horizon, f"M2_End_J{i+1}")
        m2_starts[i] = m2_start
        m2_ends[i] = m2_end
        # 创建任务i在M2上的interval变量
        m2_intervals[i] = model.NewIntervalVar(
            m2_start, DURATIONS[i][1], m2_end, f"M2_Interval_J{i+1}"
        )
    # 约束1: 每个任务都需要在 M1 和 M2 上加工
    # 这个约束已经通过创建interval变量隐式满足
    # 约束2: 任务 J2 要先在 M1 加工，再在 M2 加工（flow shop）
    # M2的开始时间 >= M1的结束时间
    model.Add(m1_ends[1] <= m2_starts[1])
    # 约束2.5: 对于没有指定顺序的任务（J1和J3），不能同时在M1与M2上加工
    # 使用NoOverlap确保同一个任务在两个机器上的运行时间不重合
    for i in [0, 2]:  # J1和J3
        model.AddNoOverlap([m1_intervals[i], m2_intervals[i]])
    # 约束3: 同一时刻一台机器只能加工一个任务（不重叠约束）
    # 使用NoOverlap约束确保不重叠
    # 对于机器M1
    m1_interval_list = [m1_intervals[i] for i in range(NUM_JOBS)]
    model.AddNoOverlap(m1_interval_list)
    # 对于机器M2
    m2_interval_list = [m2_intervals[i] for i in range(NUM_JOBS)]
    model.AddNoOverlap(m2_interval_list)
    # 约束4: 任务之间的休息间隔BUFFER
    # 对于机器M1，任意两个任务之间需要间隔BUFFER
    for i in range(NUM_JOBS):
        for j in range(i + 1, NUM_JOBS):
            # 创建布尔变量表示任务i是否在任务j之前
            before_ij = model.NewBoolVar(f"before_M1_{i}_{j}")
            # 如果 before_ij == 1，则任务i在任务j之前，且需要间隔BUFFER
            model.Add(m1_ends[i] + BUFFER <= m1_starts[j]).OnlyEnforceIf(before_ij)
            # 如果 before_ij == 0，则任务j在任务i之前，且需要间隔BUFFER
            model.Add(m1_ends[j] + BUFFER <= m1_starts[i]).OnlyEnforceIf(before_ij.Not())
    # 对于机器M2，类似处理
    for i in range(NUM_JOBS):
        for j in range(i + 1, NUM_JOBS):
            before_ij = model.NewBoolVar(f"before_M2_{i}_{j}")
            model.Add(m2_ends[i] + BUFFER <= m2_starts[j]).OnlyEnforceIf(before_ij)
            model.Add(m2_ends[j] + BUFFER <= m2_starts[i]).OnlyEnforceIf(before_ij.Not())
    # 目标：最小化makespan（所有任务完成时间的最大值）
    # makespan = max(max(end_of(m1_intervals[i]), end_of(m2_intervals[i])) for i in range(NUM_JOBS))
    # 任务的完成时间 = max(M1结束时间, M2结束时间)
    makespan = model.NewIntVar(0, horizon, "makespan")
    # 约束：makespan >= 每个任务在M1和M2上的结束时间
    for i in range(NUM_JOBS):
        # makespan >= M1的结束时间
        model.Add(makespan >= m1_ends[i])
        # makespan >= M2的结束时间
        model.Add(makespan >= m2_ends[i])
    # 最小化makespan
    model.Minimize(makespan)
    # 求解
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60.0
    solver.parameters.log_search_progress = False
    status = solver.Solve(model)
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        # 获取求解时间
        solve_time = solver.WallTime()
        # 提取解：从变量中获取开始时间
        m1_sol = []
        m2_sol = []
        for i in range(NUM_JOBS):
            m1_sol.append(solver.Value(m1_starts[i]))
            m2_sol.append(solver.Value(m2_starts[i]))
        makespan_value = solver.Value(makespan)
        # 输出结论
        print(f"\n✅ 求解完成！")
        print(f"求解时间: {solve_time:.4f} 秒")
        print(f"求解状态: {'OPTIMAL' if status == cp_model.OPTIMAL else 'FEASIBLE'}")
        # 打印解
        print_solution(m1_sol, m2_sol, makespan_value)
    else:
        print("\n❌ 未找到可行解")
        print(f"求解状态: {status}")

if __name__ == "__main__":
    try:
        solve_job_scheduling()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
