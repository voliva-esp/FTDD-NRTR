# Fast Tensor Decision Diagram

This project is actively being documented. Please check back later for detailed information on dependency, usage, and other important details.

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
To run your own circuit, put the `QASM` file (e.g., `my_circ.qasm`) in `/Benchmarks/Verification`, and replace `CIR` in the above command with the circuit instance name (e.g., `my_circ`).
To further adapth FTDD for your own use cases, please feel free to customize `/TestFTDD/DemoFTDD.py`.<br>
<br>

To verify FTDD correctness, in the project root folder, run
```sh
make verify
```
This will simulate all the circuits in `/Benchmarks/Verification` using [IBM Qiskit Aer](https://github.com/Qiskit/qiskit-aer), [PyTDD](https://github.com/Veriqc/TDD), and FTDD, and compare their fidelities. Check the log file `/TestFTDD/log/VerifyFTDD.log` for results. <br>
<br>

## Reproducing FTDD Experimental Results
To reproduce the experimental results for Google TensorNetwork, in the project root folder, run
```sh
make benchGTN
```
The results are stored in `/BenchFTDD/log/GTN`. <br>
<br>

To reproduce the experimental results for [QMDD](https://github.com/cda-tum/mqt-ddsim), in the project root folder, run
```sh
make benchQMDD
```
The results are stored in `/BenchQMDD/log`. <br>
<br>

## Citation

If you find FTDD useful in your research, we kindly request to cite our paper:
 - Qirui Zhang, Mehdi Saligane, Hun-Seok Kim, David Blaauw, Georgios Tzimpragos and Dennis Sylvester, "[Quantum Circuit Simulation with Fast Tensor Decision Diagram](https://ieeexplore.ieee.org/document/10528748)," 2024 25th International Symposium on Quality Electronic Design (ISQED), San Francisco, CA, USA, 2024, pp. 1-8.
