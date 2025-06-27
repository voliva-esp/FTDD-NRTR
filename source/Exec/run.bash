#!/bin/bash
tools=('FTDD')
#paths=('qft_indep' 'rqc' 'ghz' 'qnn' 'graph_state' 'qwalk' 'real_amplitude' 'qpe' 'amplitude_estimation' 'grover')
#circuit=('qftentangled_indep_qiskit' 'rqc' 'ghz' 'qnn_indep_tket' 'graph_state' 'qwalk_d1' 'realamprandom_indep_qiskit' 'qpeexact_indep_qiskit' 'ae_indep_qiskit' 'grover-v-chain_ind>
#minQubits=(10 1 20 5 5 3 2 2 2 2)
#maxQubits=(129 25 129 50 50 15 130 130 130 17)
methods=('seq')
paths=('qft_indep' 'rqc' 'ghz' 'qwalk' 'qpe')
circuit=('qftentangled_indep_qiskit' 'rqc' 'ghz' 'qwalk_d1' 'qpeexact_indep_qiskit')
minQubits=(20 1 40 3 2)
maxQubits=(120 10 120 15 60)
echo "Starting all executions..."
for (( j=0; j<${#tools[@]}; j++))
do
  echo " > Running ${tools[j]} backend..."
  for (( l=0; l<${#methods[@]}; l++))
  do
    echo " --- Running for ${methods[l]} contraction method..."
      for (( i=0; i<${#paths[@]}; i++))
      do
        echo " ## --- Running ${circuit[i]} circuit type..."
        for (( n=${minQubits[i]}; n<${maxQubits[i]}+1; n++))
        do
          echo " >>> ## --- N: $n"
          timeout 3600 python3 source/Exec/runCircuit.py ${n} ${methods[l]} ${paths[i]} ${circuit[i]} ${tools[j]}
          exit_code=$?
          if [ "$exit_code" -ne 0 ]; then
            if [ "$exit_code" -eq 124 ]; then
              echo " ## --- Circuit failed (timeout)"
            else
              echo " ## --- Circuit failed"
            fi
            break
          else
            echo " ## --- Circuit done"
          fi
        done
    done
    echo " --- Contraction method done"
  done
  echo "> Backend done"
done
echo "Execution ended successfully"
