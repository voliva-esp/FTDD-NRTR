from source.TDD_Q import cir_2_tn_lbl, get_real_qubit_num, add_inputs, apply_full_tetris, SlicedTensorNetwork, \
    calculate_path
from qiskit import QuantumCircuit
from source.TDD import Ini_TDD
from time import time


def make_values(n_values, iteration):
    dev_values = [0] * n_values
    for i in range(n_values - 1, -1, -1):
        if iteration >= (2 ** i):
            iteration -= 2 ** i
            dev_values[i] = 1
    return dev_values


path = './Benchmarks/Verification/'

# file_name = "qft_15"
# indices = []
# indices = ['x7_1']
# indices = ['x0_1', 'x4_1']
# indices = ['x13_0', 'x14_0', 'x6_1']

# file_name = "qft_16"
# indices = []
# indices = ['x12_0']
# indices = ['x12_0', 'x15_0']
# indices = ['x12_0', 'x13_0', 'x15_0']

# file_name = "inst_4x4_10_8"
# indices = []
# indices = ['x9_2']
# indices = ['x4_2', 'x6_2']
# indices = ['x4_2', 'x5_1', 'x6_2']

# file_name = "inst_4x4_12_8"
# indices = []
# indices = ['x10_4']
# indices = ['x10_1', 'x4_2']
# indices = ['x10_1', 'x10_4', 'x4_2']

# file_name = "qnn_indep_qiskit_12"
# indices = []
# indices = ['x3_14']
# indices = ['x7_30', 'x8_34']
# indices = ['x10_38', 'x2_10', 'x9_34']

# file_name = "qnn_indep_qiskit_13"
# indices = []
# indices = ['x9_36']
# indices = ['x10_36', 'x7_30']
# indices = ['x10_36', 'x1_6', 'x9_34']

# file_name = "qft_entangled_16"
# indices = []
# indices = ['x14_2']
# indices = ['x11_2', 'x13_2']
# indices = ['x0_1', 'x15_2', 'x1_1']

file_name = "qft_entangled_17"
# indices = []
# indices = ['x15_2']
# indices = ['x0_1', 'x2_1']
indices = ['x10_2', 'x15_2', 'x6_2']

use_tetris = True
# tool = "PyTDD"
tool = "GTN"
method = "seq"
# method = "spair"
iteration = 4

print(f" ----- {file_name} -----")
circuit = QuantumCircuit.from_qasm_file(path + file_name + '.qasm')
print(f"Slicing {len(indices)} indices which are: {indices}")

# Read and prepare the circuit
tn, all_indices, depth = cir_2_tn_lbl(circuit)
n = get_real_qubit_num(circuit)

# Add inputs
state = [0] * n
add_inputs(tn, state, n)

# Preprocess with Tetris
if use_tetris:
    tn = apply_full_tetris(tn, depth)

# Applying slicing and generate only 1 sub-circuit
n = len(indices)
values = make_values(n, iteration)
print(f"Iteration {iteration} with values {values}")
tnn = SlicedTensorNetwork(tn, indices, values)
tn = tnn.generate_tn()

# Calculate the path
path = calculate_path(tnn, method)

# Initialize PyTDD
Ini_TDD(all_indices)

# Start timer
t_ini = time()
gtn_time = None

# Make the contractions
if tool == "PyTDD":
    tdd = tn.cont_TN(path, False)
elif tool == "GTN":
    tdd, gtn_time = tn.cont_GTN(path, False)

# Calculate time spent
t_fin = time()
t_total = t_fin-t_ini
if gtn_time is not None:
    t_total = gtn_time
print(f"Time: {t_total}")

