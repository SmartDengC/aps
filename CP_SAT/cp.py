"""Cryptarithmetic puzzle.

First attempt to solve equation CP + IS + FUN = TRUE
where each letter represents a unique digit.

This problem has 72 different solutions in base 10.
"""
from ortools.sat.python import cp_model



class VarArraySolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self, variables: list[cp_model.IntVar]):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.__solution_count = 0

    def on_solution_callback(self) -> None:
        self.__solution_count += 1
        for v in self.__variables:
            print(f"{v}={self.value(v)}", end=" ")
        print()

    @property
    def solution_count(self) -> int:
        return self.__solution_count


def main() -> None:
    """solve the CP+IS+FUN==TRUE cryptarithm."""
    # Constraint programming engine
    model = cp_model.CpModel()

    base = 10

    c = model.new_int_var(1, base - 1, "C")
    p = model.new_int_var(0, base - 1, "P")
    i = model.new_int_var(1, base - 1, "I")
    s = model.new_int_var(0, base - 1, "S")
    f = model.new_int_var(1, base - 1, "F")
    u = model.new_int_var(0, base - 1, "U")
    n = model.new_int_var(0, base - 1, "N")
    t = model.new_int_var(1, base - 1, "T")
    r = model.new_int_var(0, base - 1, "R")
    e = model.new_int_var(0, base - 1, "E")

    # We need to group variables in a list to use the constraint AllDifferent.
    letters = [c, p, i, s, f, u, n, t, r, e]

    # Verify that we have enough digits.
    assert base >= len(letters)

    # Define constraints.
    model.add_all_different(letters)

    # CP + IS + FUN = TRUE
    model.add(
        c * base + p + i * base + s + f * base * base + u * base + n
        == t * base * base * base + r * base * base + u * base + e
    )

    # Creates a solver and solves the model.
    solver = cp_model.CpSolver()
    solution_printer = VarArraySolutionPrinter(letters)
    # Enumerate all solutions.
    solver.parameters.enumerate_all_solutions = True
    # Solve.
    status = solver.solve(model, solution_printer)

    # Statistics.
    print("\nStatistics")
    print(f"  status   : {solver.status_name(status)}")
    print(f"  conflicts: {solver.num_conflicts}")
    print(f"  branches : {solver.num_branches}")
    print(f"  wall time: {solver.wall_time} s")
    print(f"  sol found: {solution_printer.solution_count}")


if __name__ == "__main__":
    main()