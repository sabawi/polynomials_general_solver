import numpy as np
import matplotlib.pyplot as plt
import re
from fractions import Fraction

def parse_polynomial(input_str):
    # Step 1: Sanitize the input string
    input_str = input_str.replace(" ", "")  # Remove spaces
    input_str = re.sub(r'^f\(x\)\s*=\s*', '', input_str)  # Remove 'f(x) = '
    input_str = re.sub(r'\s*=\s*0$', '', input_str)  # Remove '= 0' at the end
    input_str = re.sub(r'([+-])=', r'\1', input_str)  # Handle "= 0" format
    input_str = re.sub(r'[^\d\+\-\*/\^\.\(\)x]', '', input_str)  # Remove invalid characters
    
    # Step 2: Parse terms
    terms = re.findall(r'([+-]?[^+-]+)', input_str)
    exponents = {}
    
    for term in terms:
        if not term:
            continue
        
        sign = 1
        if term[0] in '+-':
            if term[0] == '-':
                sign = -1
            term = term[1:]
        
        coeff = 1.0  # Default coefficient
        exp = 0      # Default exponent
        
        if 'x' in term:
            if '^' in term:
                split_term = term.split('x^')
                coeff_part = split_term[0] if split_term[0] else '1'
                exp_part = split_term[1] if len(split_term) > 1 else '1'
                exp = int(exp_part) if exp_part else 1
            else:
                split_term = term.split('x')
                coeff_part = split_term[0] if split_term[0] else '1'
                exp = 1
            
            # Handle fractional coefficients like "1/500"
            try:
                coeff = float(Fraction(coeff_part)) if '/' in coeff_part else float(coeff_part)
            except ValueError:
                coeff = 1.0  # Default if parsing fails
        else:
            try:
                coeff = float(Fraction(term)) if '/' in term else float(term)
            except ValueError:
                coeff = 0.0  # Ignore invalid terms
        
        exponents[exp] = exponents.get(exp, 0.0) + sign * coeff
    
    # Filter out zero coefficients
    exponents = {e: c for e, c in exponents.items() if c != 0.0}
    
    if not exponents:
        return [0.0]  # Zero polynomial
    
    max_exp = max(exponents.keys())
    coeffs = [exponents.get(e, 0.0) for e in range(max_exp, -1, -1)]
    
    return coeffs

def find_roots(coeffs):
    roots = np.roots(coeffs)
    tolerance = 1e-5
    real_roots = []
    complex_roots = []
    
    for root in roots:
        if abs(root.imag) < tolerance:
            real_roots.append(np.round(root.real, 2))
        else:
            complex_roots.append(np.round(root, 2))
    
    return real_roots, complex_roots

def find_extrema(coeffs):
    if len(coeffs) <= 1:
        return [], [], []
    
    deriv_coeffs = [coeffs[i] * (len(coeffs)-1 - i) for i in range(len(coeffs)-1)]
    critical_points = np.roots(deriv_coeffs).tolist()
    
    real_critical = []
    for cp in critical_points:
        if np.isreal(cp):
            real_critical.append(np.round(cp.real, 2))
    
    local_min = []
    local_max = []
    second_deriv_coeffs = [deriv_coeffs[i] * (len(deriv_coeffs)-1 - i) for i in range(len(deriv_coeffs)-1)] if deriv_coeffs else []
    
    for x in real_critical:
        y = np.polyval(coeffs, x)
        second_deriv = np.polyval(second_deriv_coeffs, x) if second_deriv_coeffs else 0
        
        if second_deriv > 0:
            local_min.append((x, y))
        elif second_deriv < 0:
            local_max.append((x, y))
    
    return local_min, local_max, real_critical


import matplotlib.gridspec as gridspec
def plot_polynomial(coeffs, real_roots, complex_roots, local_min, local_max):
    x_range = []
    if real_roots:
        x_min = min(real_roots) - 2
        x_max = max(real_roots) + 2
        x_range.extend([x_min, x_max])
    if local_min:
        x_range.append(min(x[0] for x in local_min) - 2)
        x_range.append(max(x[0] for x in local_min) + 2)
    if local_max:
        x_range.append(min(x[0] for x in local_max) - 2)
        x_range.append(max(x[0] for x in local_max) + 2)
    
    x_min_plot = min(x_range) if x_range else -10
    x_max_plot = max(x_range) if x_range else 10
    x_values = np.linspace(x_min_plot, x_max_plot, 400)
    y_values = np.polyval(coeffs, x_values)
    
    plt.figure(figsize=(16, 14))  # Increased figure size for more space 
    plt.plot(x_values, y_values, label='f(x)')
    
    # Plot real roots
    for root in real_roots:
        plt.plot(root, 0, 'ro')
        plt.annotate(f'({root:.2f}, 0)', (root, 0), xytext=(0, 10), textcoords='offset points', ha='center')
    
    # Plot extrema
    for x, y in local_min:
        plt.plot(x, y, 'bo')
        plt.annotate(f'Min: ({x:.2f}, {y:.2f})', (x, y), xytext=(0, -20), textcoords='offset points', ha='center')
    
    for x, y in local_max:
        plt.plot(x, y, 'go')
        plt.annotate(f'Max: ({x:.2f}, {y:.2f})', (x, y), xytext=(0, 20), textcoords='offset points', ha='center')
    
    # Plot global extremes in view
    global_max_idx = np.argmax(y_values)
    global_min_idx = np.argmin(y_values)
    plt.plot(x_values[global_max_idx], y_values[global_max_idx], 'y*', markersize=10)
    plt.plot(x_values[global_min_idx], y_values[global_min_idx], 'm*', markersize=10)
    
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.title('Polynomial Function Analysis', fontsize=14, pad=20)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('f(x)', fontsize=12)
    plt.grid(True)
    plt.legend(fontsize=10)
    
    # Add a spacer between the x-axis and the table
    plt.text(0.5, -0.2, " ---- ", transform=plt.gca().transAxes, fontsize=12, ha='center', va='center')
    
    # Create table data
    table_data = []
    table_data.append(['Real Roots'] + [f'{r:.2f}' for r in real_roots] + [''] * (len(real_roots) == 0))
    table_data.append(['Complex Roots'] + 
                      [f'{c.real:.2f}{"+"+str(c.imag)+"j" if c.imag >=0 else str(c.imag)+"j"}' for c in complex_roots] + 
                      [''] * (len(complex_roots) == 0))
    table_data.append(['Local Minima'] + [f'({x:.2f}, {y:.2f})' for x, y in local_min] + [''] * (len(local_min) == 0))
    table_data.append(['Local Maxima'] + [f'({x:.2f}, {y:.2f})' for x, y in local_max] + [''] * (len(local_max) == 0))
    table_data.append(['Global Max in View', f'({x_values[global_max_idx]:.2f}, {y_values[global_max_idx]:.2f})'])
    table_data.append(['Global Min in View', f'({x_values[global_min_idx]:.2f}, {y_values[global_min_idx]:.2f})'])
    
    # Ensure all rows have the same number of columns
    max_cols = max(len(row) for row in table_data)
    for row in table_data:
        row.extend([''] * (max_cols - len(row)))
    
    # Adjust the bottom margin to fit the table
    plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.4)  # Adjust bottom margin for table
    plt.tight_layout(rect=[0.05, 0.4, 0.95, 0.95])   # Adjust tight layout

    table = plt.table(cellText=table_data,
                    loc='bottom',
                    cellLoc='center',
                    colWidths=[0.15] * max_cols,
                    bbox=[0.0, -0.6, 1, 0.4])  # Moves table even lower

    table.auto_set_font_size(False)
    table.set_fontsize(9)  # Slightly larger font
    table.scale(1.2, 1.5)  # Slightly scale up the table

    plt.show()
        
# User input
input_str = input("Enter polynomial (e.g., 'f(x) = -x^4 + x^3 -6x^2 +11x +6'): ")
coeffs = parse_polynomial(input_str)

if not coeffs:
    print("Invalid polynomial format")
else:
    real_roots, complex_roots = find_roots(coeffs)
    local_min, local_max, _ = find_extrema(coeffs)
    plot_polynomial(coeffs, real_roots, complex_roots, local_min, local_max)