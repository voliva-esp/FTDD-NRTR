"""
    Code write by Vicente Lopez (voliva@uji.es). Comments and annotations will be marked with @romOlivo
"""


def generate_ftdd_data():
    """
        romOlivo: This method gets and process all the metrics from the cTDD (FTDD) package. Used for a simple way
                  to analyze and check all the metrics.
    """
    import source.cpp.build.cTDD as cTDD
    # Unique Table Metrics
    unique_hits = int(cTDD.get_count().split("\n")[0].split(" / ")[0].split("node: ")[1])
    unique_calls = int(cTDD.get_count().split("\n")[0].split(" / ")[1])
    unique_hit_ratio = float(cTDD.get_count().split("\n")[0].split(" / ")[2])
    # Addition Computed Table Metrics
    add_hits = int(cTDD.get_count().split("\n")[1].split(" / ")[0].split("add: ")[1])
    add_calls = int(cTDD.get_count().split("\n")[1].split(" / ")[1])
    add_hit_ratio = float(cTDD.get_count().split("\n")[1].split(" / ")[2])
    add_collisions = int(cTDD.get_count().split("\n")[1].split(" / ")[3])
    # Contraction Computed Table Metrics
    cont_hits = int(cTDD.get_count().split("\n")[2].split(" / ")[0].split("cont: ")[1])
    cont_calls = int(cTDD.get_count().split("\n")[2].split(" / ")[1])
    cont_hit_ratio = float(cTDD.get_count().split("\n")[2].split(" / ")[2])
    cont_collisions = int(cTDD.get_count().split("\n")[2].split(" / ")[3])
    # Other characteristics
    gc = int(cTDD.get_count().split("\n")[3].split(": ")[2])
    n_nodes_final = int(cTDD.get_count().split("\n")[3].split(": ")[1].split(', ')[0])
    return {
        "gc": gc,
        "add_hits": add_hits,
        "cont_hits": cont_hits,
        "add_calls": add_calls,
        "cont_calls": cont_calls,
        "unique_hits": unique_hits,
        "unique_calls": unique_calls,
        "n_nodes_final": n_nodes_final,
        "add_hit_ratio": add_hit_ratio,
        "add_collisions": add_collisions,
        "cont_hit_ratio": cont_hit_ratio,
        "cont_collisions": cont_collisions,
        "unique_hit_ratio": unique_hit_ratio,
    }


class OutputHandler:
    """
        romOlivo: This class was added with the objective to help handler all the information of the simulation
                  process, helping in changing to print the values for punctual test and writing the results in
                  a more formal output.
    """
    def __init__(self, tool, cont_method, circuit=None, sep="#"):
        """
            Input variables:
            tool ---------> Name of the tool used to simulate
            cont_method --> Contraction method used
            circuit ------> Circuit in the form of 'QuantumCircuit' class of qiskit
            sep ----------> Separator for the file
        """
        self.using_slicing = False
        self.cont_method = cont_method
        self.circuit = circuit
        self.tool = tool
        self.sep = sep

    def print_init(self, n_slices=0):
        pass

    def print_time_result(self, time, it=-1):
        pass

    def end_printing(self):
        pass


class PrintOutputHandler(OutputHandler):
    def __init__(self, tool, cont_method, circuit=None, sep="#"):
        """
            Input variables:
            tool -------> Name of the tool used to simulate
            cont_method --> Contraction method used
            circuit ----> Circuit in the form of 'QuantumCircuit' class of qiskit
            sep --------> Separator for the file
        """
        super().__init__(tool, cont_method, circuit, sep)

    def print_init(self, n_slices=0):
        text = "Simulating circuit"
        if n_slices > 0:
            self.using_slicing = True
            text = f"{text} using slicing with {n_slices} slices"
        if self.circuit is not None:
            text = f"Circuit {self.circuit.name} {self.sep} {text}"
        print(text)

    def print_time_result(self, time, it=-1):
        text = f"Time: {time}"
        if self.using_slicing:
            text = f"Iter: {it} {self.sep} {text}"
        print(text)

    def end_printing(self):
        print("Simulation done correctly")


class FileOutputHandler(OutputHandler):
    def __init__(self, tool, cont_method, circuit=None, sep="#", file_name=None):
        """
            Input variables:
            tool -------> Name of the tool used to simulate
            cont_method --> Contraction method used
            circuit ----> Circuit in the form of 'QuantumCircuit' class of qiskit
            sep --------> Separator for the file
            file_name --> Name of the file to write in
        """
        super().__init__(tool, cont_method, circuit, sep)
        self.file_name = file_name
        self.path = "./source/output"
        self.file = None

    def create_file_name(self, n_slices):
        file_name = "test"
        if self.circuit is not None:
            file_name = self.circuit.name
        if self.using_slicing:
            file_name = f"{file_name}{self.sep}slc{self.sep}{n_slices}"
        file_name = f"{self.tool}{self.sep}{file_name}"
        self.file_name = f"{file_name}.csv"

    def print_init(self, n_slices=0):
        import os
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        if n_slices > 0:
            self.using_slicing = True
        if self.file_name is None:
            self.create_file_name(n_slices)
        file_path = f"{self.path}/{self.file_name}"
        if os.path.exists(file_path):
            self.file = open(file_path, 'a')
        else:
            self.file = open(file_path, 'w+')
            text_line = "time"
            if self.using_slicing:
                text_line = f"slice{self.sep}{text_line}"
            text_line = f"contraction_method{self.sep}n_qubits{self.sep}{text_line}"
            self.file.write(text_line)
            self.file.write("\n")

    def print_time_result(self, time, it=-1):
        text_line = f"{time}"
        if self.using_slicing:
            text_line = f"{it}{self.sep}{text_line}"
        n_qubits = -1 if self.circuit is None else self.circuit.num_qubits
        text_line = f"{self.cont_method}{self.sep}{n_qubits}{self.sep}{text_line}"
        self.file.write(text_line)
        self.file.write("\n")

    def end_printing(self):
        self.file.close()
