import random
import sympy

# this file is used to generate a dataset for invariance training

MAX_DEGREE = 5
MAX_VAR_NUM = 3
MAX_TERM_NUM = 5
MAX_EQ_NUM = 5

random.seed(0)
degree = random.randint(1, MAX_DEGREE)
var_num = random.randint(1, MAX_VAR_NUM)
term_num = random.randint(1, MAX_TERM_NUM)
eq_num = random.randint(1, MAX_EQ_NUM)

# dict of variable name and its degree
var_dict = {
    0: "x",
    1: "y",
    2: "z"
}

class SingleItem:
    # degress list have a fixed length of var_num
    def __init__(self, coeff, degree_list) -> None:
        self.coeff = coeff
        self.degree_list = degree_list
    

class SingleExpr:
    # is_eq is a boolean value
    # const is constant number on the right hand side
    def __init__(self, is_eq, item_list, const) -> None:
        self.is_eq = is_eq
        self.item_list = item_list
        self.const = const
        self.str = ""

    def gen_str(self):
        for item in self.item_list:
            self.str += str(item.coeff)
            idx = 0
            for degree in item.degree_list:
                # print the expression with variable name
                if degree != 0:
                    self.str += var_dict[idx]
                    if degree != 1:
                        self.str += "**"
                        self.str += str(degree)
                    self.str += " "  
                idx += 1
            self.str += "+"
        # remove the last "+"
        self.str = self.str[:-1]

expr_list = []
for i in range(eq_num):
    # generate a single expression
    is_eq = random.randint(0, 1)
    item_list = []
    for j in range(term_num):
        # generate a single item
        coeff = random.randint(1, 10)
        degree_list = []
        for k in range(var_num):
            degree_list.append(random.randint(0, degree))
        item_list.append(SingleItem(coeff, degree_list))
    const = random.randint(1, 10)
    expr_list.append(SingleExpr(is_eq, item_list, const))

# print the expression
for expr in expr_list:
    expr.gen_str()
    print(expr.str)


# declare the variables
x, y, z = sympy.symbols("x y z")

# find 512 solutions to the equations
# the solutions are stored in a tensor of shape [512, var_num]
sol_list = []
# solve the equations with sympy
# try to get 512 solutions
while sol_list.__len__() < 512:
    equations = []
    # assign a random number to x
    x_val = random.randint(-500, 500)
    x_eq = "x = " + str(x_val)
    equations.append(x_eq)
    # add all the equations to the list
    for expr in expr_list:
        expr_str = expr.str
        if expr.is_eq == 1:
            expr_str += " = " + str(expr.const)
            equations.append(expr_str)
    # solve the equations
    sol = sympy.solve(equations, dict=True)
    # check if the solutions satisfy all the inequalities
    all_pass = True
    for expr in expr_list:
        if expr.is_eq == 0:
            expr_str = expr.str
            expr_str += " < " + str(expr.const)
            # evaluate the expression
            expr_val = eval(expr_str)
            if expr_val == False:
                all_pass = False
                break
    if all_pass == True:
        # append the solution to the list
        sol_list.append(sol)


