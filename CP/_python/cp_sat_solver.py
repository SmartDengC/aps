# https://developers.google.com/optimization/cp/cp_solver
# pip install ortools

from ortools.sat.python import cp_model

def simple_sat_program():
    # 创建约束规划模型
    model = cp_model.CpModel()

    # 定义变量的取值范围（0到2）
    num_vals = 3

    # 创建三个整数变量，每个变量的取值范围为 [0, num_vals-1]
    x = model.new_int_var(0, num_vals - 1, "x")
    y = model.new_int_var(0, num_vals - 1, "y")
    z = model.new_int_var(0, num_vals - 1, "z")

    # 添加约束条件：x 和 y 不能相等
    model.add(x != y)

    # 创建求解器实例
    solver = cp_model.CpSolver()

    # 求解模型
    status = solver.solve(model)

    # 检查求解状态并输出结果
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        # OPTIMAL: 找到最优解，FEASIBLE: 找到可行解
        print(f"x = {solver.value(x)}")
        print(f"y = {solver.value(y)}")
        print(f"z = {solver.value(z)}")
    else:
        print("No solution found.")

def simple_sat_program_with_solution_printer():
    from solution_printer import VarArraySolutionPrinter
    model = cp_model.CpModel()
    num_vals = 3
    x = model.new_int_var(0, num_vals - 1, "x")
    y = model.new_int_var(0, num_vals - 1, "y")
    z = model.new_int_var(0, num_vals - 1, "z")

    model.add(x != y)

    solver = cp_model.CpSolver()
    solution_printer = VarArraySolutionPrinter([x, y, z])
    # Enumerate all solutions.
    solver.parameters.enumerate_all_solutions = True
    # Solve.
    status = solver.solve(model, solution_printer)

    print(f"Status = {solver.status_name(status)}")
    print(f"Number of solutions found: {solution_printer.solution_count}")

    # 检查求解状态并输出结果
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        # OPTIMAL: 找到最优解，FEASIBLE: 找到可行解
        print(f"x = {solver.value(x)}")
        print(f"y = {solver.value(y)}")
        print(f"z = {solver.value(z)}")
    else:
        print("No solution found.")

simple_sat_program_with_solution_printer()

