from qiskit import QuantumCircuit

from source.TDD_Q import simulate

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 6:
        raise ValueError("Incorrect usage of runCircuit file. \n Correct usage: \n   $ python runCircuit.py " +
                         "<n_qubits> <contraction_method> <folder_name> <circuit_name> (<tool>)" +
                         "  These are the arguments you used: \n  " + str(sys.argv))
    n_qubits = int(sys.argv[1])
    contraction_method = sys.argv[2]
    folder_name = sys.argv[3]
    circuit_name = sys.argv[4]
    tool = "PyTDD"
    if len(sys.argv) == 6:
        tool = sys.argv[5]
    use_slicing = False
    is_input_closed = True
    is_output_closed = False
    use_tetris = False
    n_indices = 0
    slicing_method = "max"
    path=f"./Benchmarks/MQTbench/{folder_name}/"
    file_name = f"{circuit_name}_{n_qubits}"
    circuit = QuantumCircuit.from_qasm_file(path + file_name + '.qasm')
    circuit.name = circuit_name

    result = simulate(circuit, use_slicing=use_slicing, contraction_method=contraction_method,
                      is_input_closed=is_input_closed, is_output_closed=is_output_closed,
                      use_tetris=use_tetris, n_indices=n_indices, slicing_method=slicing_method,
                      backend=tool, handler_name="hybrid")


