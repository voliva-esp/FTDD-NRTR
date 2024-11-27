class CircuitCreator:
    def __init__(self):
        self.header = "OPENQASM 2.0;\n include \"qelib1.inc\" ;\n"
        self.circuit = ""

    def init_circuit(self, n_qubits):
        self.circuit = f"{self.header}qreg q[{n_qubits}];\n"

    def add_gate(self, gate, qubits_to_act):
        self.circuit = f"{self.circuit}{gate} q[{qubits_to_act[0]}]"
        if len(qubits_to_act) > 1:
            for i in range(1, len(qubits_to_act)):
                self.circuit = f"{self.circuit}, q[{qubits_to_act[i]}]"
        self.circuit = f"{self.circuit};\n"

    def create_small_circuit(self):
        self.init_circuit(3)
        self.add_gate('cx', [0, 1])
        self.add_gate('cx', [2, 1])
        self.add_gate('cx', [1, 0])
        return self.circuit

    def get_small_circuit_solution_close_close(self):
        result = 1.0
        return result

    def get_small_circuit_solution_close_open(self):
        result = [
            [[1., 0.], [0., 0.]],
            [[0., 0.], [0., 0.]]
        ]
        return result

    def get_small_circuit_solution_open_close(self):
        result = [
            [[1., 0.], [0., 0.]],
            [[0., 0.], [0., 0.]]
        ]
        return result

    def get_small_circuit_solution_open_open(self):
        result = [
            [
                [
                    [[[1., 0.], [0., 0.]], [[0., 0.], [0., 1.]]],
                    [[[0., 1.], [0., 0.]], [[0., 0.], [1., 0.]]]
                ],
                [
                    [[[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]]],
                    [[[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]]]
                ]
            ],
            [
                [
                    [[[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]]],
                    [[[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]]]
                ],
                [
                    [[[0., 0.], [0., 1.]], [[1., 0.], [0., 0.]]],
                    [[[0., 0.], [1., 0.]], [[0., 1.], [0., 0.]]]
                ]
            ]
        ]
        return result


