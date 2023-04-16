import numpy as np
import pyomo.environ as pyo
import time

solver_lib = 'gurobi'

class PyomoSolver:
    def __init__(self, N, M, C = None, D = None):
        self.N = N
        self.M = M
        if C is None or D is None:
            self.dynamic = True

            model = pyo.ConcreteModel()

            model.Users = pyo.RangeSet(0, N - 1)
            model.Items = pyo.RangeSet(0, M - 1)

            # the next line declares a variable
            model.x = pyo.Var(model.Users, model.Items, domain=pyo.NonNegativeReals, bounds=(0, 1))

            model.C = pyo.Param(model.Items, mutable=True)
            model.D = pyo.Param(model.Users, mutable=True)

            def cap_constraint_rule(m, i):
                return sum(m.x[u, i] for u in range(N)) <= m.C[i]

            def dem_constraint_rule(m, u):
                return sum(m.x[u, i] for i in range(M)) <= m.D[u]

            model.CapConstraint = pyo.Constraint(model.Items, rule=cap_constraint_rule)
            model.DemConstraint = pyo.Constraint(model.Users, rule=dem_constraint_rule)

            model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

            model.Theta = pyo.Param(model.Users, model.Items, mutable=True)

            for u in range(N):
                for i in range(M):
                    model.Theta[u, i] = 1

            def obj_expression(m):
                return -1 * pyo.summation(m.Theta, m.x)

            model.OBJ = pyo.Objective(rule=obj_expression)

            self.model = model
            self.pyomo_opt = pyo.SolverFactory(solver_lib)
        else:
            self.dynamic = False

            model = pyo.ConcreteModel()

            model.Users = pyo.RangeSet(0, N - 1)
            model.Items = pyo.RangeSet(0, M - 1)

            # the next line declares a variable
            model.x = pyo.Var(model.Users, model.Items, domain=pyo.NonNegativeReals, bounds=(0, 1))

            def cap_constraint_rule(m, i):
                return sum(m.x[u, i] for u in range(N)) <= C[i]

            def dem_constraint_rule(m, u):
                return sum(m.x[u, i] for i in range(M)) <= D[u]

            model.CapConstraint = pyo.Constraint(model.Items, rule=cap_constraint_rule)
            model.DemConstraint = pyo.Constraint(model.Users, rule=dem_constraint_rule)

            model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

            model.Theta = pyo.Param(model.Users, model.Items, mutable=True)

            for u in range(N):
                for i in range(M):
                    model.Theta[u, i] = 1

            def obj_expression(m):
                return -1 * pyo.summation(m.Theta, m.x)

            model.OBJ = pyo.Objective(rule=obj_expression)

            self.model = model
            self.pyomo_opt = pyo.SolverFactory(solver_lib)

    def _solve_system_static(self, R, x_prev = None, p_init=None):
        N, M = R.shape

        if p_init is not None:
            c = self.model.CapConstraint
            for index in c:
                self.model.dual[c[index]] = p_init[index]

        if x_prev is not None:
            for u in range(N):
                for i in range(M):
                    self.model.x[u, i] = x_prev[u, i]

        for u in range(N):
            for i in range(M):
                self.model.Theta[u, i] = R[u, i]

        self.pyomo_opt.solve(self.model)

        x_opt = np.zeros((N, M))

        for u in range(N):
            for i in range(M):
                x_opt[u, i] = np.int32(self.model.x[u, i].value)

        return np.int32(x_opt)

    def _solve_system_dynamic(self, R, C, D, x_prev=None, p_init=None):
        N, M = R.shape

        if p_init is not None:
            c = self.model.CapConstraint
            for index in c:
                self.model.dual[c[index]] = p_init[index]

        if x_prev is not None:
            for u in range(N):
                for i in range(M):
                    self.model.x[u, i] = x_prev[u, i]

        for u in range(N):
            for i in range(M):
                self.model.Theta[u, i] = R[u, i]

        for u in range(N):
            self.model.D[u] = D[u]

        for i in range(M):
            self.model.C[i] = C[i]

        self.pyomo_opt.solve(self.model)

        x_opt = np.zeros((N, M))

        for u in range(N):
            for i in range(M):
                x_opt[u, i] = np.int32(self.model.x[u, i].value)

        return np.int32(x_opt)

    def solve_system(self, *args, **kwargs):
        if self.dynamic:
            return self._solve_system_dynamic(*args, **kwargs)
        else:
            return self._solve_system_static(*args, **kwargs)

    def get_prices(self):
        c = self.model.CapConstraint
        prices = np.zeros(self.M)
        for index in c:
            prices[index] = np.abs(self.model.dual[c[index]])
        return prices

