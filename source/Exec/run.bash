#!/bin/bash
paths=('qft_entangled' 'qaoa' 'ghz')
circuit=('qft_entangled' 'qaoa_indep_qiskit' 'ghz')
minQubits=(12 18 14 30)
maxQubits=(16 19 15 31)
echo "Starting all executions..."
for method in k-ops seq
do
  echo " > Running for $method contraction method..."
  for (( i=0; i<${#paths[@]}; i++))
  do
    echo " --- Running ${circuit[i]} circuit type..."
    python3 source/Exec/runCircuit.py ${minQubits[i]} ${maxQubits[i]} $method ${paths[i]} ${circuit[i]}
    echo " --- Circuit done"
  done
  echo " > Contraction method done"
done
echo "Execution ended successfully"



