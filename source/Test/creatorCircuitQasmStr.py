"""

    This file was created and documented by Vicente Lopez (voliva@uji.es, @romOlivo) for testing purposes.

"""


class CircuitCreator:
    """
        This class is used to generate circuits in str QASM format. Also has some solutions of
        the generated circuits when are contracted in some cases. Used for testing purposes.
    """
    def __init__(self):
        self.header = "OPENQASM 2.0;\n include \"qelib1.inc\" ;\n"
        self.circuit = ""

    def init_circuit(self, n_qubits):
        """
            Create the str header of the circuit, following the QASM format. Will be common for all the circuits.
            All circuits will use only one quantum register.
        """
        self.circuit = f"{self.header}qreg q[{n_qubits}];\n"

    def add_gate(self, gate, qubits_to_act):
        """
            Add a gate to the str representation of the circuit.
            gate -> Str that represents the gate you want to apply
            qubits_to_act -> Array with the number of the qubits, in order, which interacts with the gate
        """
        self.circuit = f"{self.circuit}{gate} q[{qubits_to_act[0]}]"
        if len(qubits_to_act) > 1:
            for i in range(1, len(qubits_to_act)):
                self.circuit = f"{self.circuit}, q[{qubits_to_act[i]}]"
        self.circuit = f"{self.circuit};\n"

    def create_small_circuit(self):
        """
            Creates a very simple circuit for testing purposes
        """
        self.init_circuit(3)
        self.add_gate('cx', [0, 1])
        self.add_gate('cx', [2, 1])
        self.add_gate('cx', [1, 0])
        return self.circuit

    def get_small_circuit_solution_close_close(self):
        """
            Returns the solution of contracting the small circuit when the input and the output are closed with
            the state 0.
        """
        result = 1.0
        return result

    def get_small_circuit_solution_close_open(self):
        """
            Returns the solution of contracting the small circuit when the input is closed with
            the state 0.
        """
        result = [
            [[1., 0.], [0., 0.]],
            [[0., 0.], [0., 0.]]
        ]
        return result

    def get_small_circuit_solution_open_close(self):
        """
            Returns the solution of contracting the small circuit when the output is closed with
            the state 0.
        """
        result = [
            [[1., 0.], [0., 0.]],
            [[0., 0.], [0., 0.]]
        ]
        return result

    def get_small_circuit_solution_open_open(self):
        """
            Returns the solution of contracting the small circuit when nor the input or the output are closed
        """
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


