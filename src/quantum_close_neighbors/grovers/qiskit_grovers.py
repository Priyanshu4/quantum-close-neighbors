from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.circuit.library import GroverOperator
from qiskit.circuit.library import PhaseOracle
from qiskit.circuit.classicalfunction import classical_function
import numpy as np
from typing import Set
from tqdm import tqdm
from quantum_close_neighbors.grovers import AbstractGroverResult
from quantum_close_neighbors.grovers import Grovers
from abc import abstractmethod, ABC

def randomly_marked_elements(n_elements: int, n_marked_elements: int) -> Set[int]:
    """ Randomly choose n_marked_elements from n_elements.
    """
    return set(np.random.choice(n_elements, n_marked_elements, replace=False))


class QiskitGrovers(Grovers, ABC):

    def __init__(self, n_elements: int, marked_elements: Set[int]):
        self.n_elements = n_elements
        self.marked_elements = marked_elements
        self.n_qubits = int(np.ceil(np.log2(n_elements)))

        self.seen_elements = set()
        self.last_seen_element = None

        self.oracle = self.build_oracle_circuit()
        self.grover_op = GroverOperator(oracle=self.oracle)


    @classmethod
    def init_with_random_marked_elements(cls, n_elements: int, n_marked_elements: int):
        marked_elements = randomly_marked_elements(n_elements, n_marked_elements)
        return cls(n_elements, marked_elements)


    def run(self, num_iterations: int, mark_result_seen: bool = True) -> AbstractGroverResult:
        """ Run Grover's algorithm for a given number of iterations.
        """
        grover_circuit = self.build_grover_circuit(num_iterations)
       
        # Run the circuit
        backend = Aer.get_backend('qasm_simulator')
        compiled_circuit = transpile(grover_circuit, backend, optimization_level=0)
        job = backend.run(compiled_circuit, shots=1)
        result = job.result()
        counts = result.get_counts()
        most_common = max(counts, key=counts.get)

        # Convert the result to an integer
        element = int(most_common, 2)

        # Check if the element is marked
        if element in self.marked_elements:
            if element in self.seen_elements:
                result = AbstractGroverResult.MARKED_SEEN
            else:
                result = AbstractGroverResult.MARKED_UNSEEN
        else:
            result = AbstractGroverResult.UNMARKED

        # Mark the element as seen
        if mark_result_seen:
            self.last_seen_element = element
            self.seen_elements.add(element)

        return result
    
    def estimate_p_success(self, num_iterations: int, num_samples: int, with_tqdm = False) -> float:
        """ Estimate the probability of success after num_iterations using num_samples samples.
        """
        marked = 0
        unmarked = 0

        iterator = range(num_samples)
        if with_tqdm:
            iterator = tqdm(iterator)

        for _ in iterator:
            result = self.run(num_iterations, mark_result_seen=False)

            if result.is_marked():
                marked += 1
            else:
                unmarked += 1

        return marked / (marked + unmarked)

    def remove_seen_element_from_oracle(self):
        """ Remove the most recently seen element from the oracle.
            This means that the element is no longer marked.
        """
        if self.last_seen_element is not None:
            self.marked_elements.remove(self.last_seen_element)
            self.last_seen_element = None

        # rebuild oracle
        self.oracle = self.build_oracle_circuit()
        self.grover_op = GroverOperator(oracle=self.oracle)

    def remove_all_seen_elements_from_oracle(self):
        """ Remove all seen elements from the oracle.
            This means that those elements are no longer marked.
        """
        self.marked_elements -= self.seen_elements
        self.seen_elements = set()

        # rebuild oracle
        self.oracle = self.build_oracle_circuit()
        self.grover_op = GroverOperator(oracle=self.oracle)
        
    def build_grover_circuit(self, num_iterations: int):
        
        qc = QuantumCircuit(self.n_qubits)
        qc.h(range(self.n_qubits))
        qc.compose(self.grover_op.power(num_iterations), inplace=True)
        qc.measure_all()

        return qc


    @abstractmethod
    def build_oracle_circuit(self):
        pass





class QiskitGroversPhaseOracle(QiskitGrovers):

    # Function to convert marked elements into a Boolean expression
    @staticmethod
    def generate_boolean_expression(num_qubits, marked_elements):

        if len(marked_elements) == 0:
            raise ValueError("No marked elements provided")

        expressions = []

        for element in marked_elements:
            bitstring = bin(element)
            # find the 'b' and only get string after that
            if 'b' in bitstring:
                bitstring = bitstring[bitstring.index('b')+1:]
            
            # Pad with zeros to get the full bitstring
            bitstring = bitstring.zfill(num_qubits)
            assert len(bitstring) == num_qubits

            conditions = []
            for i, bit in reversed(list(enumerate(bitstring))):
                var = f'q{i}'  # Use q0, q1, q2, ..., qN as variable names
                if bit == '1':
                    conditions.append(f"{var}")
                else:
                    conditions.append(f"~{var}")
            
            # Combine conditions for the bitstring
            expression = '(' + ' & '.join(conditions) + ')'
            expressions.append(expression)

        # Combine all expressions using OR
        full_expression = ' | '.join(expressions)
        return full_expression

    def run(self, num_iterations: int, mark_result_seen: bool = True) -> AbstractGroverResult:
        """ Run Grover's algorithm for a given number of iterations.
        """
        if num_iterations == 0:
            # draw a random element
            element = np.random.randint(0, self.n_elements)

            if element in self.marked_elements:
                if element in self.seen_elements:
                    result = AbstractGroverResult.MARKED_SEEN
                else:
                    result = AbstractGroverResult.MARKED_UNSEEN
            else:
                result = AbstractGroverResult.UNMARKED

            if mark_result_seen:
                self.last_seen_element = element
                self.seen_elements.add(element)

            return result
        
        return super().run(num_iterations, mark_result_seen)

    def build_oracle_circuit(self):
        if len(self.marked_elements) == 0:
            return QuantumCircuit(self.n_qubits) # Empty circuit

        oracle = PhaseOracle(self.generate_boolean_expression(self.n_qubits, self.marked_elements))
        return transpile(oracle, backend = Aer.get_backend('qasm_simulator'))


class QiskitGroversStaticPhaseOracle(QiskitGroversPhaseOracle):

    def __init__(self, n_elements: int, marked_elements: Set[int]):
        super().__init__(n_elements, marked_elements)
        self.marked_elements = list(marked_elements)

        # cached circuits for a 
        self.prebuilt_circuits = dict()

    def run(self, num_iterations: int, mark_result_seen: bool = True) -> AbstractGroverResult:
        """ Run Grover's algorithm for a given number of iterations.
        """
        backend = Aer.get_backend('qasm_simulator')
        
        if num_iterations not in self.prebuilt_circuits:
            grover_circuit = self.build_grover_circuit(num_iterations)
            compiled_circuit = transpile(grover_circuit, backend, optimization_level=0)
            self.prebuilt_circuits[num_iterations] = compiled_circuit
        else:
            compiled_circuit = self.prebuilt_circuits[num_iterations]
       
        # Run the circuit
        job = backend.run(compiled_circuit, shots=1)
        result = job.result()
        counts = result.get_counts()
        most_common = max(counts, key=counts.get)

        # Convert the result to an integer
        element = int(most_common, 2)

        # Check if the element is marked
        if element in self.marked_elements:
            if element in self.seen_elements:
                result = AbstractGroverResult.MARKED_SEEN
            else:
                result = AbstractGroverResult.MARKED_UNSEEN
        else:
            result = AbstractGroverResult.UNMARKED

        # Mark the element as seen
        if mark_result_seen:
            self.last_seen_element = element
            self.seen_elements.add(element)

        return result
    
    def remove_all_seen_elements_from_oracle(self):
        return NotImplementedError("This method is not supported for this implementation")
    
    def remove_seen_element_from_oracle(self):
        return NotImplementedError("This method is not supported for this implementation")