# Fast Tensor Decision Diagram

Fast Tensor Decision Diagram (FTDD) is a novel open-source software framework for quantum circuit simulation (QCS). It uses [tensor decision diagram (TDD)](https://dl.acm.org/doi/full/10.1145/3514355) as the backend. It employs a series of tensor network level optimizations, including rank simplification and contraction ordering, as preprocessing steps for the input quantum circuit. To enhance performance, the orginal Python-based TDD backend ([PyTDD](https://github.com/Veriqc/TDD)), has been re-engineered in C++. This redesign incorporates several data structural optimizations, drawing on some of the key techniques used in [binary decision diagram implementation](https://ieeexplore.ieee.org/document/114826).

## Prerequisites

Ensure you have the following:
- Linux (e.g., Ubuntu 20.04 LTS or RHEL 8)
- Anaconda
- GCC version >= 9.1
- CMake version >= 3.10
<br>

Create and activate a conda virtual environment with Python 3.8.15:
```sh
conda create --name myenv python=3.8.15
conda activate myenv
```
<br>

Install Python dependencies:
```sh
pip install -r requirements.txt
pip install -U git+https://github.com/jcmgray/cotengra.git
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```
<br>

Install C/C++ dependencies:
```sh
conda install -c conda-forge eigen
conda install -c conda-forge graphviz
conda install -c conda-forge pybind11
```
<br>

## Building the Project

In the project root folder, run
```sh
cd /source/cpp/
mkdir build
cd build
cmake ..
make
```
<br>

## Running the Project

For a demo of FTDD, in the project root folder, run
```sh
make demo CIR=inst_4x4_10_8
```
This will simulate a 16-qubit depth-10 [Google random quantum circuit](https://github.com/sboixo/GRCS) instance using both [PyTDD](https://github.com/Veriqc/TDD) and FTDD, and compare their performance.
To run your own circuit, put the `QASM` file (e.g., `my_circ.qasm`) in `/Benchmarks/Verification`, and assign `CIR` in the above command with the circuit instance name (e.g., `my_circ`).
To further adapth FTDD for your own use cases, please feel free to customize `/TestFTDD/DemoFTDD.py`.<br>
<br>

To verify FTDD correctness, in the project root folder, run
```sh
make verify
```
This will simulate all the circuits in `/Benchmarks/Verification` using [IBM Qiskit Aer](https://github.com/Qiskit/qiskit-aer), [PyTDD](https://github.com/Veriqc/TDD), and FTDD, and compare their fidelities. Check the log file `/TestFTDD/log/VerifyFTDD.log` for results. <br>
<br>

## Reproducing FTDD Experimental Results
To reproduce the experimental results for [Google TensorNetwork](https://github.com/google/TensorNetwork), in the project root folder, run
```sh
make benchGTN
```
The logs are stored in `/BenchFTDD/log/GTN`. Experimental results are stored as `csv` files in `/BenchFTDD/data/GTN`. <br>
<br>

To reproduce the experimental results for [QMDD](https://github.com/cda-tum/mqt-ddsim), in the project root folder, run
```sh
make benchQMDD
```
The logs are stored in `/BenchQMDD/log`. Experimental results are stored as `csv` files in `/BenchQMDD/data`. <br>
<br>

To reproduce the experimental results for [PyTDD](https://github.com/Veriqc/TDD), in the project root folder, run
```sh
make benchPyTDD
```
The logs are stored in `/BenchFTDD/log/PyTDD`. Experimental results are stored as `csv` files in `/BenchFTDD/data/PyTDD`. <br>
<br>

To reproduce the experimental results for FTDD, in the project root folder, run
```sh
make benchFTDD
```
The logs are stored in `/BenchFTDD/log/FTDD`. Experimental results are stored as `csv` files in `/BenchFTDD/data/FTDD`. <br>
<br>

## Citation

If you find FTDD useful in your research, we kindly request to cite our paper:
 - Qirui Zhang, Mehdi Saligane, Hun-Seok Kim, David Blaauw, Georgios Tzimpragos and Dennis Sylvester, "[Quantum Circuit Simulation with Fast Tensor Decision Diagram](https://ieeexplore.ieee.org/document/10528748)," 2024 25th International Symposium on Quality Electronic Design (ISQED), San Francisco, CA, USA, 2024, pp. 1-8.
