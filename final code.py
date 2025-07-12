import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from IPython.display import display

s, t = sp.symbols('s t')

node_no = int(input("Enter the number of nodes: "))
admittance_matrix = np.zeros((node_no, node_no), dtype=object)
current_matrix = np.zeros((node_no, node_no), dtype=object)
res_no=int(input("How many resistor in the circuit? "))
for i in range(0,res_no):
    print(f"\nDetails for Resistor {i + 1}:")
    first_node=int(input("Enter the first node: "))
    second_node=int(input("Enter the second node: "))
    value=float(input("Enter the value of resistor in (ohm): "))
    admittance_matrix[first_node-1][second_node-1]+=1/value
    admittance_matrix[second_node-1][first_node-1]+=1/value
            
ind_no=int(input("How many inductor in the circuit? "))

for i in range(0,ind_no):
    print(f"\nDetails for Inductor {i + 1}:")
    first_node=int(input("Enter the first node: "))
    second_node=int(input("Enter the second node: "))
    value=float(input("Enter the value of inductor in (H): "))
    admittance_matrix[first_node-1][second_node-1]+=1/(value*s)
    admittance_matrix[second_node-1][first_node-1]+=1/(value*s)
    in_current = int(input("Is there any initial current in the inductor? (1 for yes, 0 for no): "))
    if in_current:
      pos_dirc = int(input("Node from which current is coming: "))
      neg_dirc = int(input("Node at which current is going: "))
      curr_value = float(input("Value of initial current in (A): "))
      current_matrix[pos_dirc - 1][neg_dirc - 1] += curr_value/s
      current_matrix[neg_dirc - 1][pos_dirc - 1] -= curr_value/s
                  
cap_no=int(input("How many capacitor in the circuit? "))
for i in range(0,cap_no):
    print(f"\nDetails for Capacitor {i + 1}:")
    first_node=int(input("Enter the first node: "))
    second_node=int(input("Enter the second node: "))
    value=float(input("Enter the value of capacitor in (F): "))
    admittance_matrix[first_node-1][second_node-1]+=value*s
    admittance_matrix[second_node-1][first_node-1]+=value*s
    in_voltage = int(input("Is there any initial voltage on capacitor? (1 for yes, 0 for no): "))
    if in_voltage:
      pos_voldir = int(input("Positive side of capacitor at node: "))
      neg_voldir = int(input("Negative side of capacitor at node: "))
      voltval = float(input("Value of initial voltage: "))
      Lvoltval = value*voltval
      current_matrix[pos_voldir - 1][neg_voldir - 1] -= Lvoltval
      current_matrix[neg_voldir - 1][pos_voldir - 1] += Lvoltval
                  
indcurrent_no=int(input("How many current source in the circuit? "))
for i in range(0,indcurrent_no):
   current_val = float(input("Enter current source value: "))
   pos_dir = int(input("Node from which current is coming: "))
   neg_dir = int(input("Node at which current is going: "))
   current_matrix[pos_dir - 1][neg_dir - 1] += current_val/s
   current_matrix[neg_dir - 1][pos_dir - 1] -= current_val/s  
              
symbol = sp.symbols(f'v1:{node_no + 1}', real=True)
voltage_matrix = sp.Matrix(symbol)

num_sources = int(input("Enter the number of voltage sources: "))

sources = {}
supernodes = {}

for i in range(num_sources):
    print(f"\nDetails for voltage source {i + 1}:")
    pos = int(input(f"Positive terminal at node: "))
    neg = int(input(f"Negative terminal at node: "))
    volt = float(input(f"Value of voltage source (in volts): "))
    sources[i] = {'pos': pos, 'neg': neg, 'volt': volt}
    supernodes[i] = {'pos': pos, 'neg': neg}

grnd = int(input("Ground is at which Node?: "))
voltage_matrix[grnd - 1] = 0

for source in sources.values():
    pos, neg, volt = source['pos'], source['neg'], source['volt']
    Lvolt = sp.laplace_transform(volt, t, s, noconds=False)
   if pos != grnd:
       if neg != grnd:
        voltage_matrix[pos - 1] = voltage_matrix[neg - 1] + Lvolt[0]
       else:
         voltage_matrix[pos - 1] = Lvolt[0]
   elif neg != grnd:
        voltage_matrix[neg - 1] = -Lvolt       

SupernodalAnalysis_matrix = sp.Matrix(num_sources, 1, lambda i, _: 0)

i=-1
for supernode in supernodes.values():
    pos, neg = supernode['pos'], supernode['neg']
    if pos!=grnd and neg!=grnd:
        i+=1
        for col in range(node_no):
            if admittance_matrix[pos-1][col] or admittance_matrix[neg-1][col]:
                SupernodalAnalysis_matrix[i]+=(voltage_matrix[pos-1]-voltage_matrix[col])*admittance_matrix[pos-1][col] + (voltage_matrix[neg-1]-voltage_matrix[col])*admittance_matrix[neg-1][col]

display(SupernodalAnalysis_matrix)

nodalAnalysis_matrix = sp.Matrix(node_no, 1, lambda i, _: 0)

excluded_nodes = set()
for source in sources.values():
    excluded_nodes.add(source['pos'])
    excluded_nodes.add(source['neg'])

for node in range(node_no):
    if (node + 1) not in excluded_nodes:
        for col in range(node_no):
            if admittance_matrix[node][col]:
                nodalAnalysis_matrix[node] += (voltage_matrix[node] - voltage_matrix[col]) * admittance_matrix[node][col]

for i in range(node_no):
    if i+1 not in excluded_nodes:
        nodalAnalysis_matrix[i] += sum(current_matrix[i])

combined_matrix = sp.Matrix(num_sources + len(nodalAnalysis_matrix), 1, lambda i, _: 0)
for i in range(num_sources+len(nodalAnalysis_matrix)):
    combined_matrix[i]=nodalAnalysis_matrix[i] if i< len(nodalAnalysis_matrix) else SupernodalAnalysis_matrix[i-len(nodalAnalysis_matrix)]
    
nodalVoltages_solution = sp.solve(combined_matrix, symbol)


voltage_matrix_with_solution = voltage_matrix.subs(nodalVoltages_solution)

print("\nFrequency-Domain Solution for Node Voltages:")
for i, voltage in enumerate(voltage_matrix_with_solution, start=1):
    display(f"V{i} = {voltage}")

vol_matrix_soln_inT = sp.inverse_laplace_transform(voltage_matrix_with_solution, s, t)
vol_matrix_soln_inT = vol_matrix_soln_inT.applyfunc(lambda x: x.rewrite(sp.exp)).applyfunc(sp.simplify)

print("\nTime-Domain Solution for Node Voltages:")
for i, voltage in enumerate(vol_matrix_soln_inT, start=1):
    display(f"V{i} = {voltage}")
    
    
nodalCurrent_matrix_with_soln = np.zeros((node_no, node_no), dtype=object)

for row in range(0, node_no):
    is_voltage_source_pos = False
    voltage_source_neg = None
    for source in sources.values():
        if source['pos'] == row + 1:
            is_voltage_source_pos = True
            voltage_source_neg = source['neg'] - 1
            break

    if is_voltage_source_pos:
        current_sum = 0
        for col in range(0, node_no):
            if admittance_matrix[row][col]:
                current_sum += (voltage_matrix_with_solution[row] - voltage_matrix_with_solution[col]) * admittance_matrix[row][col]

        nodalCurrent_matrix_with_soln[row][voltage_source_neg] = current_sum
        nodalCurrent_matrix_with_soln[voltage_source_neg][row] = -current_sum
    else:
        for col in range(0, node_no):
            if admittance_matrix[row][col]:
                nodalCurrent_matrix_with_soln[row][col] = (voltage_matrix_with_solution[row] - voltage_matrix_with_solution[col]) * admittance_matrix[row][col]
                nodalCurrent_matrix_with_soln[col][row] = (voltage_matrix_with_solution[col] - voltage_matrix_with_solution[row]) * admittance_matrix[col][row]

display(nodalCurrent_matrix_with_soln)

def CAL_CURRENT(firnodes, secnodes, comp=None):
    if comp is None:
        current_inS = nodalCurrent_matrix_with_soln[firnodes-1][secnodes-1]
    else:
        current_inS = (voltage_matrix_with_solution[firnodes-1] - voltage_matrix_with_solution[secnodes-1]) / comp

    current_inT = sp.inverse_laplace_transform(current_inS, s, t)
    fun = lambda x: x.rewrite(sp.exp)
    current_inT = fun(current_inT)
    current_inT = sp.simplify(current_inT)
    display(current_inT)

    plot = int(input("Do you want to plot it? yes=1, no=0: "))
    if plot:
        f_lambdified = sp.lambdify(t, current_inT, modules=["numpy"])
        t_vals = np.linspace(0, 2*np.pi, 100)
        y_vals = f_lambdified(t_vals)

        plt.plot(t_vals, y_vals, label="Current through branch/element")
        plt.xlabel("Time (t)")
        plt.ylabel("Current")
        plt.title("Branch/Element Current")
        plt.legend()
        plt.grid(True)
        plt.show()

    plotv = int(input("do you want voltage plot? yes=1, no=0: "))
    if plotv:


       vplt=vol_matrix_soln_inT[a-1]-vol_matrix_soln_inT[b-1]

       f_lambdified = sp.lambdify(t,vplt, modules=["numpy"])
       to_vals = np.linspace(0, 2*np.pi, 100)
       yo_vals = f_lambdified(to_vals)
       f_lambdified = sp.lambdify(t, vplt, modules=["numpy"])
       plt.plot(to_vals, yo_vals, label="voltage across branch/element")
       plt.xlabel("Time (t)")
       plt.ylabel("voltage")
       plt.title("Branch/Element voltage")

       plt.grid(True)
       plt.show()


while(int(input("Do you want to see specific Branch Current or voltage? yes=1, no=0: "))):
   a = int(input("Enter first node: "))
   b = int(input("Enter second node: "))
   elem_type = int(input("Across which element:Press() resistor=0, inductor=1, capacitor=2, voltage source=3: "))

   if elem_type == 3:
       print("Current through voltage source or nodal current difference:")
       CAL_CURRENT(a, b)
   else:
       if elem_type == 0:
           comp = float(input("Enter value of resistance: "))
       elif elem_type == 1:
           comp = float(input("Enter the value of inductance: ")) * s
       elif elem_type == 2:
           comp = 1 / (float(input("Enter the value of capacitance: ")) * s)

       CAL_CURRENT(a, b, comp)            
