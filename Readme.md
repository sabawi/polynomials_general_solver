# Polynomial Solver

This project provides a Python-based tool for analyzing and visualizing polynomial functions. It can parse polynomial expressions, find their roots (both real and complex), identify local and global extrema, and generate a plot of the polynomial along with key features.

## Features

-   **Polynomial Parsing:** Accepts polynomial expressions in a user-friendly format (e.g., `f(x) = -x^4 + x^3 - 6x^2 + 11x + 6`).
-   **Root Finding:** Computes both real and complex roots of the polynomial.
-   **Extrema Detection:** Identifies local minima and maxima, as well as global extrema within the plotted range.
-   **Visualization:** Generates a plot of the polynomial, highlighting real roots, local minima, local maxima, and global extrema.
-   **Informative Table:** Displays a table summarizing the real roots, complex roots, local minima, local maxima, and global extrema.
- **Fractional coefficients**: The program can handle fractional coefficients like "1/500"

## How to Use

1.  **Run the Script:** Execute the `poly2.py` Python script.
2.  **Input Polynomial:** When prompted, enter the polynomial expression in the specified format. For example:
    ```
    Enter polynomial (e.g., 'f(x) = -x^4 + x^3 -6x^2 +11x +6'): f(x) = -x^4 + x^3 - 6x^2 + 11x + 6
    ```
3.  **View Results:** The script will generate a plot of the polynomial and display a table with the calculated roots and extrema.

## Dependencies

-   **NumPy:** For numerical computations (e.g., finding roots, evaluating polynomials).
-   **Matplotlib:** For plotting the polynomial and its features.
-   **re:** For regular expression operations (parsing the polynomial string).
- **fractions**: For handling fractional coefficients

You can install these dependencies using pip:

```bash
pip install numpy matplotlib
```

## Example

**Input:**

```
f(x) = -x^4 + x^3 - 6x^2 + 11x + 6
