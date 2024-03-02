import pennylane as qml

def dummy_validate(y_qbits, flag_qbit, ancilla_qbits):
    qml.PauliX(flag_qbit)

def gen_neigh_oracle_generator(validate):
    def gen_neigh_oracle(j):
        def neigh_oracle(x_qbits, y_qbits, w_qbit, ancilla_qbits):
            qml.PauliX(x_qbits[j])
            validate(x_qbits, w_qbit, y_qbits + ancilla_qbits)
            for x_q,y_q in zip(x_qbits, y_qbits):
                qml.ctrl(qml.PauliX(y_q), x_q, 1)
            qml.PauliX(x_qbits[j])
        return neigh_oracle
    return gen_neigh_oracle

def dummy_oracle():
    return gen_neigh_oracle_generator(dummy_validate)