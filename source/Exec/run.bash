#!/bin/bash
tools=('PyTDD' 'GTN' 'FTDD')
paths=('qft_entangled' 'qaoa' 'ghz')
circuit=('qft_entangled' 'qaoa_indep_qiskit' 'ghz')
minQubits=(12 14 30)
maxQubits=(16 15 31)
echo "Starting all executions..."
for (( j=0; j<${#tools[@]}; j++))
do
  echo " > Running ${tools[j]} backend..."
  for method in k-ops seq
  do
    echo " --- Running for $method contraction method..."
    for (( i=0; i<${#paths[@]}; i++))
    do
      echo " # --- Running ${circuit[i]} circuit type..."
      python3 source/Exec/runCircuit.py ${minQubits[i]} ${maxQubits[i]} $method ${paths[i]} ${circuit[i]} ${tools[j]}
      echo " # --- Circuit done"
    done
    echo " --- Contraction method done"
  done
  echo "> Backend done"
done
echo "Execution ended successfully"



