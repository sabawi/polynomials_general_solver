import cupy as cp
import numpy as np

def find_polynomial_roots_gpu():
    print("CuPy Version:", cp.__version__)
    print("CUDA Version:", cp.cuda.runtime.runtimeGetVersion())  # [[9]]
    
    n = int(input("Enter the degree of the polynomial: "))
    coefficients = [float(input(f"Enter coefficient for x^{n - i} term: ")) if i < n 
                    else float(input("Enter the constant term: ")) 
                    for i in range(n + 1)]
    
    coeffs_gpu = cp.array(coefficients, dtype=cp.float64)
    if coeffs_gpu[0] == 0:
        raise ValueError("Leading coefficient cannot be zero")
    
    normalized_coeffs = coeffs_gpu / coeffs_gpu[0]
    degree = len(coefficients) - 1
    
    companion = cp.zeros((degree, degree), dtype=cp.complex128)
    companion[0, :] = -normalized_coeffs[1:].astype(cp.complex128)
    companion[1:, :-1] = cp.eye(degree - 1)
    
    try:
        eigenvalues = cp.linalg.eig(companion)[0]  # Try GPU-accelerated [[2]]
    except AttributeError:
        print("Falling back to NumPy for eigenvalue computation...")
        eigenvalues = np.linalg.eigvals(cp.asnumpy(companion))  # Explicit NumPy fallback [[10]]
    
    roots = cp.asnumpy(eigenvalues) if isinstance(eigenvalues, cp.ndarray) else eigenvalues
    
    print("\nRoots of the polynomial:")
    for root in roots:
        if abs(root.imag) < 1e-10:
            print(f"{root.real:.6f}")
        else:
            print(f"{root.real:.6f} + {root.imag:.6f}i")



# Write a function to read json data descriptor for the polynomial and return the roots
def find_polynomial_roots_gpu_json(json_data):
    coeffs = json_data["coefficients"]
    coeffs_gpu = cp.array(coeffs, dtype=cp.float64)
    if coeffs_gpu[0] == 0:
        raise ValueError("Leading coefficient cannot be zero")
    
    normalized_coeffs = coeffs_gpu / coeffs_gpu[0]
    degree = len(coeffs) - 1
    
    companion = cp.zeros((degree, degree), dtype=cp.complex128)
    companion[0, :] = -normalized_coeffs[1:].astype(cp.complex128)
    companion[1:, :-1] = cp.eye(degree - 1)
    
    try:
        eigenvalues = cp.linalg.eig(companion)[0]  # Try GPU-accelerated [[2]]
    except AttributeError:
        print("Falling back to NumPy for eigenvalue computation...")
        eigenvalues = np.linalg.eigvals(cp.asnumpy(companion))  # Explicit NumPy fallback [[10]]
    
    roots = cp.asnumpy(eigenvalues) if isinstance(eigenvalues, cp.ndarray) else eigenvalues
    
    return roots

if __name__ == "__main__":
    print("Running the polynomial solver on the GPU")
    print("First, manually enter the polynomial coefficients")
    find_polynomial_roots_gpu() # Call the function
    print("\nNow, let's use a JSON data descriptor")
    json_data = {
        "coefficients": [1, -6, 11, -6]
    }
    roots = find_polynomial_roots_gpu_json(json_data)
    print("\nRoots of the polynomial:")
    for root in roots:
        if abs(root.imag) < 1e-10:
            print(f"{root.real:.6f}")
        else:
            print(f"{root.real:.6f} + {root.imag:.6f}i")
            