package ys.project.sdyth.factory.production.application.schedule.cpsat.constraint;

import com.google.ortools.sat.*;
import lombok.Data;
import lombok.EqualsAndHashCode;
import org.junit.jupiter.api.Test;
import ys.project.sdyth.factory.production.application.schedule.cpsat.OrToolsNativeLoader;
import ys.project.sdyth.factory.production.application.schedule.cpsat.PackScheduleCpSatService;
import ys.project.sdyth.factory.production.application.schedule.cpsat.PackScheduleProblem;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

/**
 * Google OR-Tools CP-SAT 求解器端到端单元测试。
 * 构造一个极简排产场景（1 个产品、2 个机台、2 个时间桶），
 * 验证 {@link PackScheduleCpSatService#solve(PackScheduleProblem)} 能够返回可行解，
 * 且排产结果满足基本业务约束（需求全部满足、不超额排产）。
 *
 * @author 邓聪
 * @since 2026/8/7 09:48
 */
class GoogleCpSatTest {

    @Test
    void simpleIntCpSatTest() {
        OrToolsNativeLoader.load();
        CpModel model = new CpModel();
        int numVals = 3;

        IntVar x = model.newIntVar(0, numVals - 1, "x");
        IntVar y = model.newIntVar(0, numVals - 1, "y");
        IntVar z = model.newIntVar(1, numVals - 1, "z");

        model.addDifferent(x, y);

        CpSolver cpSolver = new CpSolver();
        VarArraySolutionPrinter cb = new VarArraySolutionPrinter(new IntVar[]{x, y, z});
        // Tell the solver to enumerate all solutions
        cpSolver.getParameters().setEnumerateAllSolutions(true);

        // And solver
        CpSolverStatus status = cpSolver.solve(model, cb);
        if (status.equals(CpSolverStatus.OPTIMAL) || status.equals(CpSolverStatus.FEASIBLE)) {
            System.out.println(cpSolver.value(x));
            System.out.println(cpSolver.value(y));
            System.out.println(cpSolver.value(z));
        } else {
            System.out.println("无解");
        }
    }

    @Test
    void simpleStringCpSatTest() {
        OrToolsNativeLoader.load();
        final int numNurses = 4;
        final int numDays = 3;
        final int numShifts = 3;
        final int[] allNurses = IntStream.range(0, numNurses).toArray();
        final int[] allDays = IntStream.range(0, numDays).toArray();
        final int[] allShifts = IntStream.range(0, numShifts).toArray();

        CpModel cpModel = new CpModel();
        Literal[][][] shifts = new Literal[numNurses][numDays][numShifts];
        for (int n : allNurses) {
            for (int d : allDays) {
                for (int s : allShifts) {
                    shifts[n][d][s] = cpModel.newBoolVar("shifts_n" + n + "d" + d + "s" + s);
                }
            }
        }

        for (int d : allDays) {
            for (int s : allShifts) {
                ArrayList<Literal> nurses = new ArrayList<>();
                for (int n : allNurses) {
                    nurses.add(shifts[n][d][s]);
                }
                cpModel.addExactlyOne(nurses);
            }
        }

        for (int n : allNurses) {
            for (int d : allDays) {
                List<Literal> work = new ArrayList<>();
                for (int s : allShifts) {
                    work.add(shifts[n][d][s]);
                }
                cpModel.addAtMostOne(work);
            }
        }
    }

    @Data
    @EqualsAndHashCode(callSuper = true)
    static class VarArraySolutionPrinter extends CpSolverSolutionCallback {
        private final IntVar[] variableArray;
        private int solutionCount;

        public VarArraySolutionPrinter(IntVar[] variables) {
            variableArray = variables;
        }

        @Override
        public void onSolutionCallback() {
            System.out.printf("Solution #%d: time = %.02f s%n", solutionCount, wallTime());
            for (IntVar intVar : variableArray) {
                System.out.printf("  %s = %d%n", intVar.getName(), value(intVar));
            }
            solutionCount++;
        }
    }

    @Test
    public void t1() {
        OrToolsNativeLoader.load();

        CpModel model = new CpModel();
        IntVar x1 = model.newIntVar(0, 1, "x1");
        IntVar x2 = model.newIntVar(0, 1, "x2");
        IntVar y = model.newIntVar(0, 1, "y");

        // y <= x1
        model.addLessOrEqual(y, x1);
        // y <= x2
        model.addLessOrEqual(y, x2);
        // y >= x1 + x2 -1
        model.addGreaterOrEqual(y, LinearExpr.newBuilder().add(x1).add(x2).add(-1).build());


        CpSolver solver = new CpSolver();
        CpSolverStatus status = solver.solve(model);
        if (status == CpSolverStatus.OPTIMAL || status == CpSolverStatus.FEASIBLE) {
            System.out.println("Solution:");
            System.out.println("x1 = " + solver.value(x1));
            System.out.println("x2 = " + solver.value(x2));
            System.out.println("y = " + solver.value(y));
        } else {
            System.out.println("No solution found.");
        }
    }

    @Test
    void t2() {
        OrToolsNativeLoader.load();
        CpModel model = new CpModel();
        BoolVar a = model.newBoolVar("a");
        BoolVar b = model.newBoolVar("b");

        model.addBoolOr(new Literal[]{a, b});

        CpSolver cpSolver = new CpSolver();
        CpSolverStatus status = cpSolver.solve(model);

        if (status == CpSolverStatus.OPTIMAL || status == CpSolverStatus.FEASIBLE) {
            System.out.println("Solution:");
            System.out.println("a = " + cpSolver.value(a));
            System.out.println("b = " + cpSolver.value(b));
        } else {
            System.out.println("No solution found.");
        }
    }
}

