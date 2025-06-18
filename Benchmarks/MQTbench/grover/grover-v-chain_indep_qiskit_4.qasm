// Benchmark was created by MQT Bench on 2024-03-19
// For more information about MQT Bench, please visit https://www.cda.cit.tum.de/mqtbench/
// MQT Bench version: 1.1.0
// Qiskit version: 1.0.2

OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
h q[0];
h q[1];
h q[2];
x q[3];
cp(pi/4) q[2],q[3];
cx q[2],q[1];
cp(-pi/4) q[1],q[3];
cx q[2],q[1];
cp(pi/4) q[1],q[3];
cx q[1],q[0];
cp(-pi/4) q[0],q[3];
cx q[2],q[0];
cp(pi/4) q[0],q[3];
cx q[1],q[0];
cp(-pi/4) q[0],q[3];
u2(0,0) q[1];
cx q[2],q[0];
cp(pi/4) q[0],q[3];
u2(0,0) q[0];
u1(-pi) q[2];
ccx q[0],q[1],q[2];
u2(-pi,-pi) q[0];
u2(-pi,-pi) q[1];
u1(-pi) q[2];
cp(pi/4) q[2],q[3];
cx q[2],q[1];
cp(-pi/4) q[1],q[3];
cx q[2],q[1];
cp(pi/4) q[1],q[3];
cx q[1],q[0];
cp(-pi/4) q[0],q[3];
cx q[2],q[0];
cp(pi/4) q[0],q[3];
cx q[1],q[0];
cp(-pi/4) q[0],q[3];
u2(0,0) q[1];
cx q[2],q[0];
cp(pi/4) q[0],q[3];
u2(0,0) q[0];
u1(-pi) q[2];
ccx q[0],q[1],q[2];
u2(-pi,-pi) q[0];
u2(-pi,-pi) q[1];
u1(-pi) q[2];
