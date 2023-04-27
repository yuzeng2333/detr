import random
import itertools

def generate_polynomial_equation(terms, degree):
    return [random.randint(-10, 10) for _ in range(terms * (degree + 1))]

def generate_equations_and_inequalities(N, degree):
    # Generate random coefficients for polynomial equations and inequalities
    coefficients = [generate_polynomial_equation(3, degree) for _ in range(N)]

    # Generate random right-hand side values for polynomial equations and inequalities
    rhs_values = [random.randint(-10, 10) for _ in range(N)]

    # Generate a list of tuples representing the polynomial equations and inequalities
    equations_and_inequalities = list(zip(coefficients, rhs_values))

    return equations_and_inequalities

def evaluate_polynomial(coefficients, values, degree):
    result = 0
    for i in range(0, len(coefficients), degree + 1):
        term = 1
        for j in range(degree + 1):
            term *= (coefficients[i + j] * values[j // (degree + 1)]) ** j
        result += term
    return result

def find_solutions(equations_and_inequalities, degree):
    # Find all possible variable value combinations (assuming integer values)
    variable_values = list(itertools.product(range(-10, 11), repeat=3))

    # Check which combinations satisfy the polynomial equations and inequalities
    solutions = []
    for values in variable_values:
        is_solution = True
        for coefficients, rhs in equations_and_inequalities:
            if evaluate_polynomial(coefficients, values, degree) < rhs:
                is_solution = False
                break
        if is_solution:
            solutions.append(values)

    return solutions

# Initialize an empty list for solutions
solutions = []

# Set the maximum polynomial degree
max_degree = 5
random.seed(0)
# Repeat the process until at least 512 solutions are found
while len(solutions) < 512:
    N = random.randint(1, 4)
    degree = random.randint(1, max_degree)
    equations_and_inequalities = generate_equations_and_inequalities(N, degree)
    solutions = find_solutions(equations_and_inequalities, degree)

print(f"Generated random polynomial equations/inequalities (degree up to {max_degree}):")
for i, (coefficients, rhs) in enumerate(equations_and_inequalities, start=1):
    print(f"  {i}: {' + '.join([f'{c}x^{j}' if j > 0 else f'{c}' for j, c in enumerate(coefficients)])} >= {rhs}")

print(f"At least 512 solutions found. Total solutions found: {len(solutions)}")

with open("solutions.txt", "a") as f:
    print("Solutions:", file=f)
    for i, solution in enumerate(solutions, start=1):
        print(f"  {i}: {solution}", file=f)

