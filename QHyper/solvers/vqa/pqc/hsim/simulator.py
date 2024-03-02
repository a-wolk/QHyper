import pennylane as qml
import pennylane.numpy as np
import pennylane.operation as qml_op
from .util import gen_lambda
from .color import Color

W = np.array([
    [1, 0, 0, 0],
    [0, 1/np.sqrt(2), 1/np.sqrt(2), 0],
    [0, 1/np.sqrt(2), -1/np.sqrt(2), 0],
    [0, 0, 0, 1]
])

# def E_IT(t):
#     return np.array([
#         [1, 0],
#         [0, np.exp(1j * t)]
#     ])

# class e_it(qml_op.Operation):
#     num_params = 1
#     num_wires = 1
#     par_domain = None

#     @staticmethod
#     def compute_matrix(t):
#         return np.array([
#             [1, 0],
#             [0, np.exp(1j * t)]
#         ])

@gen_lambda
def A(x_qbits, y_qbits, flag_qbit):
    for x_q, y_q in zip(x_qbits, y_qbits):
        qml.QubitUnitary(W, [x_q, y_q])
        qml.PauliX(y_q)
        qml.Toffoli([x_q, y_q, flag_qbit])
        qml.PauliX(y_q)

def simulate_max_one(t, oracle):
    def func(x_qbits, ancilla_qbits):
        y_qbits = ancilla_qbits[:len(x_qbits)]
        w_qbit = ancilla_qbits[len(x_qbits)]
        flag_qbit = ancilla_qbits[len(x_qbits)+1]

        oracle(x_qbits, y_qbits, w_qbit, ancilla_qbits[len(x_qbits)+2:])
        A(x_qbits, y_qbits, flag_qbit)()
        qml.ctrl(qml.PhaseShift(-t, wires=w_qbit), flag_qbit, 0)
        qml.ctrl(qml.PhaseShift(t, wires=w_qbit), flag_qbit, 1)
        # qml.ctrl(e_it(-t, wires=w_qbit), flag_qbit, 0)
        # qml.ctrl(e_it(t, wires=w_qbit), flag_qbit, 1)
        # qml.ctrl(qml.QubitUnitary(E_IT(-t), w_qbit), flag_qbit, 0)
        # qml.ctrl(qml.QubitUnitary(E_IT(t), w_qbit), flag_qbit, 1)
        qml.adjoint(A(x_qbits, y_qbits, flag_qbit))()
        qml.adjoint(oracle)(x_qbits, y_qbits, w_qbit, ancilla_qbits[len(x_qbits)+2:])
    return func

def gen_color_oracle(color, gen_neigh_oracle):
    j_neigh_oracle = gen_neigh_oracle(color.j)
    def func(x_qbits, y_qbits, w_qbit, ancilla_qbits):
        j_neigh_oracle(x_qbits, y_qbits, w_qbit, ancilla_qbits)
    return func

def color_simulator(t, color: Color, gen_neigh_oracle):
    color_oracle = gen_color_oracle(color, gen_neigh_oracle)
    simulator = simulate_max_one(t, color_oracle)
    def func(x_qbits, ancilla_qbits):
        simulator(x_qbits, ancilla_qbits)
    return func

def simulator(gen_neigh_oracle, t, r):
    def func(x_qbits, ancilla_qbits):
        n = len(x_qbits)
        for _ in range(r):
            for j in range(n):
                color_simulator(t/(2*r), Color(j, 0, 0), gen_neigh_oracle)(x_qbits, ancilla_qbits)
            for j in range(n-1, -1, -1):
                color_simulator(t/(2*r), Color(j, 0, 0), gen_neigh_oracle)(x_qbits, ancilla_qbits)
    return func
