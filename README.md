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

For a demo of FTDD run
```sh
make demo CIR=inst_4x4_10_8
```
This will simulate a 16-qubit depth-10 Google random quantum circuit instance using both PyTDD and FTDD, and compare their performance.
To run your own circuit, put the `QASM` file in `/Benchmarks/Verification`, and replace `CIR` in the above command with the circuit instance name.
<br>

## Re-producing FTDD Experimental Results

## Citation

If you find FTDD useful in your research, we kindly request to cite our paper:
 - Qirui Zhang, Mehdi Saligane, Hun-Seok Kim, David Blaauw, Georgios Tzimpragos and Dennis Sylvester, "[Quantum Circuit Simulation with Fast Tensor Decision Diagram](https://ieeexplore.ieee.org/document/10528748)," 2024 25th International Symposium on Quality Electronic Design (ISQED), San Francisco, CA, USA, 2024, pp. 1-8.
