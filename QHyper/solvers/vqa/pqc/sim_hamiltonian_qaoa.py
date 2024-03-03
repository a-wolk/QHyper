from dataclasses import dataclass, field
import pennylane as qml
from pennylane import numpy as np
import math

import numpy.typing as npt
from QHyper.problems.base import Problem
from QHyper.optimizers.qml_gradient_descent import QmlGradientDescent
from QHyper.solvers.vqa.pqc.qml_qaoa import QML_QAOA
from QHyper.util import Operator

from .hsim import simulator, gen_neigh_oracle_generator

@dataclass
class SimHamiltonianQAOA(QML_QAOA):
    layers: int = 3
    backend: str = "lightning.qubit"
    mixer: str = 'ham_1_feasible'

    def run_opt(
        self,
        problem: Problem,
        opt_args: npt.NDArray[np.float64],
        hyper_args: npt.NDArray[np.float64]
    ) -> float:
        if self.optimizer == '':
            raise ValueError("Optimizer not provided, if you don't "
                             "want to use optimizer use qaoa instead")
        self.dev = qml.device(
            self.backend, wires=[str(x) for x in problem.variables] + [f"a{i}" for i in range(3 + 2 + len(problem.variables))])
        optimizer_instance = QmlGradientDescent(
            self.optimizer, **self.optimizer_args)

        return optimizer_instance.minimize_expval_func(
            self.get_expval_circuit(problem, list(hyper_args)), opt_args)
    
    def run_with_probs(
        self,
        problem: Problem,
        opt_args: npt.NDArray[np.float64],
        hyper_args: npt.NDArray[np.float64],
    ) -> dict[str, float]:
        self.dev = qml.device(
            self.backend, wires=[str(x) for x in problem.variables] + [f"a{i}" for i in range(3 + 2 + len(problem.variables))])
        probs = self.get_probs_func(problem, list(hyper_args))(opt_args.reshape(2, -1))
        return {
            format(result, "b").zfill(len(problem.variables)): float(prob)
            for result, prob in enumerate(probs)
        }

    def _circuit(self, problem: Problem, params: npt.NDArray[np.float64],
                 cost_operator: qml.Hamiltonian) -> None:
        print("SimHamiltonianQAOA", self.layers, self.backend)
        def ctrl_add(k):
            def func(r_qbits, control_qbits):
                for i in range(len(r_qbits)):
                    qml.ctrl(qml.RZ, control=control_qbits, control_values=[1]*len(control_qbits))(k * np.pi / (2**i), wires=r_qbits[i])
            return func
        
        def check_constraint(constraint):
            def func(r_qbits, control_qbits):
                qml.QFT(wires=r_qbits)
                for vars, coeff in constraint.lhs.items():
                    if len(vars) == 0:
                        continue
                    
                    ctrl_add(coeff)(r_qbits, control_qbits)
                qml.adjoint(qml.QFT)(wires=r_qbits)
            return func
        
        def check_constraints(y_qbits, c_flag_qbits, ancilla_qbits):
            for constraint,flag_qbit in zip(problem.constraints, c_flag_qbits):
                max_v = sum([coeff if coeff > 0 and len(vars) > 0 else 0 for vars, coeff in constraint.lhs.items()])
                min_v = sum([coeff if coeff < 0 and len(vars) > 0 else 0 for vars, coeff in constraint.lhs.items()])
                C = sum([coeff if len(vars) == 0 else 0 for vars, coeff in constraint.lhs.items()]) - constraint.rhs
                if constraint.operator == Operator.GE:
                    max_v, min_v = -min_v, -max_v
                    C = -C

                if C > 0:
                    vrange = (min_v, max_v + C)
                else:
                    vrange = (min_v + C, max_v)

                if vrange[1] <= 0 or vrange[0] >= 0:
                    print("Constraint is always satisfied")
                    continue

                neg_len = math.ceil(math.log2(-vrange[0]))
                pos_len = math.ceil(math.log2(vrange[1] + 1))
                r_len = max(neg_len, pos_len) + 1
                r_qbits = ancilla_qbits[:r_len]

                check_constraint(constraint)(r_qbits, y_qbits)
                qml.ctrl(qml.PauliX, control=r_qbits[0], control_values=1)(flag_qbit)
                qml.adjoint(check_constraint(constraint))(r_qbits, y_qbits)
        
        def validate(y_qbits, flag_qbit, ancilla_qbits) -> None:
            c_flag_qbits = ancilla_qbits[:len(problem.constraints)]
            ancilla_qbits = ancilla_qbits[len(problem.constraints):]

            check_constraints(y_qbits, c_flag_qbits, ancilla_qbits)
            qml.ctrl(qml.PauliX, control=c_flag_qbits, control_values=[1]*len(c_flag_qbits))(flag_qbit)
            qml.adjoint(check_constraints)(y_qbits, c_flag_qbits, ancilla_qbits)

        gen_neigh_oracle = gen_neigh_oracle_generator(validate)
        def qaoa_layer(gamma: list[float], beta: list[float]) -> None:
            qml.qaoa.cost_layer(gamma, cost_operator)
            simulator(gen_neigh_oracle, beta, 1)([str(v) for v in problem.variables], [f"a{i}" for i in range(3 + 2 + len(problem.variables))])
            #qml.exp(self._calculate_mixing_hamiltonian(problem), (-1j) * beta)

        #self._hadamard_layer(problem) #TODO: superposition of feasible states
        qml.QubitStateVector(self._calculate_state(problem), wires=[str(v) for v in problem.variables])
        qml.layer(qaoa_layer, self.layers, params[0], params[1])

    def _calculate_state(self, problem: Problem):
        variables_n = len(problem.variables)

        def validate(x: int) -> bool:
            x = list(map(int, bin(x)[2:].zfill(variables_n)))
            values = {str(var): x[i] for i,var in enumerate(problem.variables)}

            valid = True
            for constraint in problem.constraints:
                res = 0
                for vars, coeff in constraint.lhs.items():
                    res += coeff * np.prod([values[var] for var in vars])

                if constraint.operator == Operator.LE:
                    valid = res <= constraint.rhs
                elif constraint.operator == Operator.GE:
                    valid = res >= constraint.rhs
                
                if not valid:
                    break
            return valid

        state = np.array(list(map(validate, np.arange(2**variables_n).tolist())))
        return state / np.linalg.norm(state)

    def _calculate_mixing_hamiltonian(self, problem: Problem) -> qml.Hermitian:
        variables_n = len(problem.variables)
        matrix = np.zeros((2**variables_n, 2**variables_n), dtype=np.complex128)

        def validate(x: int) -> bool:
            x = list(map(int, bin(x)[2:].zfill(variables_n)))
            values = {str(var): x[i] for i,var in enumerate(problem.variables)}

            valid = True
            for constraint in problem.constraints:
                res = 0
                for vars, coeff in constraint.lhs.items():
                    res += coeff * np.prod([values[var] for var in vars])

                if constraint.operator == Operator.LE:
                    valid = res <= constraint.rhs
                elif constraint.operator == Operator.GE:
                    valid = res >= constraint.rhs
                
                if not valid:
                    break
            return valid

        for i in range(2**variables_n):
            if not validate(i):
                continue

            for j in range(i+1, 2**variables_n):
                if validate(j):
                    matrix[i, j] = 1
                    matrix[j, i] = 1

        return qml.Hermitian(matrix, wires=[str(v) for v in problem.variables])