from source.TDD_Q import simulate, cir_2_tn_lbl, get_real_qubit_num, add_outputs, add_inputs, apply_full_tetris, \
    calculate_path, SlicedTensorNetwork
from qiskit import QuantumCircuit
from copy import deepcopy
from source.TDD import Ini_TDD, add
from time import time

path = './../Benchmarks/Verification/'
# file_names = ["3_17_13", "3_17_13_2", "ex-1_166", "qft_15", "qft_16"]
# file_names = ["3_17_13", "3_17_13_2"]
file_names = ["test"]

# path = './../Benchmarks/MQTbench/'
# file_names = ["GRQC/inst_4x4_10_8", "GRQC/inst_4x4_12_8", "GRQC/inst_4x5_10_8", "GRQC/inst_4x5_14_8",
#              "GRQC/inst_4x5_16_8", "GRQC/inst_4x5_20_8"]
# file_names = ["GRQC/inst_4x4_10_8"]

file_name = file_names[0]

cir = QuantumCircuit.from_qasm_file(path + file_name + '.qasm')

# Read and prepare the circuit
tn, all_indices_lbl, depth = cir_2_tn_lbl(cir)
n = get_real_qubit_num(cir)

# Initialize PyTDD
Ini_TDD(all_indices_lbl)

input_state = [0] * n

t_total = 0

# Inputs and outputs are here to make the simple contractions using tetris
add_inputs(tn, input_state, n)
tn_original = deepcopy(tn)
N = 2**n
for i in range(N):
    output_state = [0] * n
    j = n - 1
    while j >= 0 and i > 0:
        t = 2**j
        if i >= t:
            output_state[j] = 1
            i -= t
        j -= 1
    print(output_state)

    tn = deepcopy(tn_original)
    add_outputs(tn, output_state, n)

    # Calculate the path
    path = calculate_path(SlicedTensorNetwork(tn, [], []), "cot")

    t_ini = time()
    # Make the contractions
    tdd = tn.cont_TN(path, False)
    t_fin = time()

    t_total += t_fin-t_ini
print(f"Time (s): {t_total}")

