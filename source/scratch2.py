
for n in range(2, 131):
    print(n)
    text = ""
    path = f"qftentangled_indep_qiskit_{n}.qasm"
    with open(path) as file:
        text_list = file.read().split("\n")[:-(n+1)]
        for t in text_list:
            text = f"{text}{t}\n"

    with open(path, "w") as file:
        file.write(text)

