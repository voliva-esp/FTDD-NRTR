#!/bin/bash
tools=('GTN' 'FTDD')
paths=('qft_indep' 'rqc' 'ghz')
circuit=('qftentangled_indep_qiskit' 'rqc' 'ghz')
minQubits=(2 1 5)
maxQubits=(3 12 6)
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



