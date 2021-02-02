# QOSF_Screening_Task
___________________________________________________________________
This is an implementation of a second screening task given by QOSF in order to enroll in a Quantum Mentorship Program

___________________________________________________________________
DESCRIPTION

This task utilizes the bit flip and the sign flip code to error-correct a circuit in which an "error gate" is applied. It either applies a unity, x or Z matrices, with finite probabilities, respectively. The goal is to obtain the desired output state after some error correction circuits. 

___________________________________________________________________
SOLUTION

Since the initial state, after the Hadamard gate, is |+0>, we can see that applying the "error gate" to the first qubit (+) may result in a sign flip, whilst applying an "error gate" to the second qubit (0) may result in a bit flip. 

Therefore, the sign flip error correction circuit will be applied to correct the possible sign flip of the upper qubit (while remembering to apply the Hadamard gate again at the end to get from the computational basis back to |+>), and the bit flip error correction circuit will be applied to correct the possible bit flip of the lower one. 

Two ancilla qubits will be initialized to |0> for each erorr correction circuit, and then discarded. 

