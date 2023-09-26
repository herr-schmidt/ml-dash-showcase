import pyomo.environ as pyo
import numpy as np


class SVM:

    def __init__(self, solver="ipopt"):
        self.model = pyo.AbstractModel()
        self.model_instance = None
        self.solver = pyo.SolverFactory(solver)

        self.C = 1e3
        self.w = 0
        self.b = 0
        self.support_vectors = []
        self.support_vectors_Y = []
        self.lagrange_coefficients = []

    def define_training_set_cardinality(self, model):
        model.I = pyo.Param(within=pyo.PositiveIntegers)

    def define_training_set_cardinality_range_set(self, model):
        model.i = pyo.RangeSet(1, model.I)

    def define_variables(self, model):
        model.alpha = pyo.Var(model.i,
                              domain=pyo.NonNegativeReals)

    def define_parameters(self, model):
        model.y = pyo.Param(model.i)
        model.inner_products = pyo.Param(model.i,
                                         model.i)

    def define_alpha_upper_bound_constraint(self, model):
        model.alpha_upper_bound_constraint = pyo.Constraint(model.i,
                                                            rule=lambda model, i: self.alpha_upper_bound_rule(model, i))

    def alpha_upper_bound_rule(self, model, i):
        return model.alpha[i] <= self.C

    def define_alpha_y_constraint(self, model):
        model.alpha_y_constraint = pyo.Constraint(rule=lambda model: self.alpha_y_rule(model))

    def alpha_y_rule(self, model):
        return sum(model.alpha[i] * model.y[i] for i in model.i) == 0

    def objective_function(self, model):
        return sum(model.alpha[i] for i in model.i) - 0.5 * sum(model.alpha[i] * model.alpha[j] * model.y[i] * model.y[j] * model.inner_products[i, j] for i in model.i for j in model.i)

    def define_objective(self, model):
        model.objective = pyo.Objective(
            rule=self.objective_function,
            sense=pyo.maximize)

    def define_model(self):
        self.define_training_set_cardinality(self.model)
        self.define_training_set_cardinality_range_set(self.model)
        self.define_parameters(self.model)
        self.define_variables(self.model)

        self.define_alpha_upper_bound_constraint(self.model)
        self.define_alpha_y_constraint(self.model)

        self.define_objective(self.model)

    def create_model_instance(self, data):
        print("Creating model instance...")
        self.model_instance = self.model.create_instance(data)

    def solve(self):
        print("Solving instance...")
        self.model.results = self.solver.solve(self.model_instance, tee=True)
        print("\ninstance solved.")

    def train(self, X, Y):
        inner_products = {}
        i = 1
        for v1 in X:
            j = 1
            for v2 in X:
                inner_products[(i, j)] = np.dot(v1, v2)
                j += 1
            i += 1

        targets = {}
        i = 1
        for y in Y:
            targets[i] = y
            i += 1

        data_dictionary = {
            None: {
                'I': {None: len(targets)},
                'y': targets,
                'inner_products': inner_products,
            }
        }

        self.define_model()
        self.create_model_instance(data_dictionary)
        self.solve()

        self.w = 0
        self.b = 0
        self.support_vectors = []
        self.support_vectors_Y = []
        self.lagrange_coefficients = []

        for i in self.model_instance.i:
            # consider it a support vector
            if self.model_instance.alpha[i].value > 1e-6:
                self.w += self.model_instance.alpha[i].value * targets[i] * X[i - 1]
                self.support_vectors.append(X[i - 1])
                self.support_vectors_Y.append(targets[i])
                self.lagrange_coefficients.append(self.model_instance.alpha[i].value)

        # average for numerical accuracy
        for i in range(len(self.support_vectors)):
            self.b += self.support_vectors_Y[i] - np.dot(self.w, self.support_vectors[i])

        self.b = self.b / len(self.support_vectors)

    def predict(self, X):
        predictions = []
        for x in X:
            if np.dot(self.w, x) + self.b >= 0:
                predictions.append(1)
            else:
                predictions.append(-1)

        return predictions

    def score(self, X, y):
        predictions = self.predict(X)
        tp_tn = 0
        for i in range(len(predictions)):
            if predictions[i] == y[i]:
                tp_tn += 1

        return tp_tn / len(predictions)
