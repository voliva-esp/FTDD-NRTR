from qiskit import QuantumCircuit

from source.TDD_Q import simulate


class CircuitGroupRun:
    def __init__(self, path, circuit_name, contraction_method, backend="PyTDD", use_tetris=False, use_slicing=False,
                 slicing_method="max", n_indices=1, prefix_circuit_name=""):
        self.path = path
        self.backend = backend
        self.n_indices = n_indices
        self.use_tetris = use_tetris
        self.use_slicing = use_slicing
        self._circuit_name = circuit_name
        self.slicing_method = slicing_method
        self.contraction_method = contraction_method
        self.prefix_circuit_name = prefix_circuit_name

        self.is_input_closed = True
        self.is_output_closed = False

    def _run_single_circuit(self, n):
        file_name = f"{self._circuit_name}_{n}{self.prefix_circuit_name}"
        circuit = QuantumCircuit.from_qasm_file(self.path + file_name + '.qasm')
        circuit.name = self._circuit_name
        result = simulate(circuit, use_slicing=self.use_slicing, contraction_method=self.contraction_method,
                          is_input_closed=self.is_input_closed, is_output_closed=self.is_output_closed,
                          use_tetris=self.use_tetris, n_indices=self.n_indices, slicing_method=self.slicing_method,
                          backend=self.backend, handler_name="file")

    def run(self, first_n_qubits, last_n_qubits):
        for n_qubits in range(first_n_qubits, last_n_qubits):
            self._run_single_circuit(n_qubits)
        if last_n_qubits > first_n_qubits:
            self._run_single_circuit(last_n_qubits)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 6:
        raise ValueError("Incorrect usage of runCircuit file. \n Correct usage: \n   $ python runCircuit.py " +
                         "<min_n_qubits> <max_n_qubits> <contraction_method> <folder_name> <circuit_name> (<tool>)" +
                         "  These are the arguments you used: \n  " + str(sys.argv))
    min_qubits = int(sys.argv[1])
    max_qubits = int(sys.argv[2])
    contraction_method = sys.argv[3]
    folder_name = sys.argv[4]
    circuit_name = sys.argv[5]
    tool = "PyTDD"
    if len(sys.argv) == 7:
        tool = sys.argv[6]
    cgr = CircuitGroupRun(path=f"./Benchmarks/MQTbench/{folder_name}/", circuit_name=circuit_name,
                          contraction_method=contraction_method, backend=tool)
    cgr.run(min_qubits, max_qubits)


