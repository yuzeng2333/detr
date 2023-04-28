import random
import sympy
from sympy.parsing.sympy_parser import parse_expr

# this file is used to generate a dataset for invariance training

MAX_DEGREE = 5
MAX_VAR_NUM = 3
MAX_TERM_NUM = 5
MAX_EQ_NUM = 5
SOL_NUM = 2

random.seed(0)

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
        if self.str != "":
            return
        for item in self.item_list:
            self.str += str(item.coeff) + "*"
            idx = 0
            # check if degree_list has non-zero value
            all_zero = True
            for degree in item.degree_list:
                if degree != 0:
                    all_zero = False
                    break
            if all_zero == True:
                self.str += "1"
            else:
                for degree in item.degree_list:
                    # print the expression with variable name
                    if degree != 0:
                        self.str += "(" + var_dict[idx]
                        if degree != 1:
                            self.str += "**"
                            self.str += str(degree)
                        self.str += ")*"  
                    idx += 1
            # remove the last "*" if it is "*"
            if self.str[-1] == "*":
                self.str = self.str[:-1]
            self.str += " + "
        # remove the last " + "
        if self.str[-3:] == " + ":
            self.str = self.str[:-3]

    def print_expr(self):
        if self.is_eq == 1:
            print(self.str + " = " + str(self.const))
        else:
            print(self.str + " < " + str(self.const))

expr_list = []
for i in range(eq_num):
    term_num = random.randint(1, MAX_TERM_NUM)
    # generate a single expression
    is_eq = random.randint(0, 1)
    item_list = []
    for j in range(term_num):
        # generate a single item
        degree = random.randint(1, MAX_DEGREE)
        coeff = random.randint(1, 10)
        degree_list = []
        for k in range(MAX_VAR_NUM):
            var_not_included = random.randint(0, 1)
            if var_not_included == 1:
                degree_list.append(0)
            else:
                single_degree = random.randint(0, degree)
                degree_list.append(single_degree)
                degree = degree - single_degree
        item_list.append(SingleItem(coeff, degree_list))
    const = random.randint(1, 10)
    expr_list.append(SingleExpr(is_eq, item_list, const))

# print the expression
for expr in expr_list:
    expr.gen_str()
    expr.print_expr()


# declare the variables
x, y, z = sympy.symbols("x y z")

# find 512 solutions to the equations
# the solutions are stored in a tensor of shape [512, var_num]
sol_list = []
# solve the equations with sympy
# try to get 512 solutions
while sol_list.__len__() < SOL_NUM:
    equations = []
    # assign a random number to x
    x_val = random.randint(-500, 500)
    x_eq = "x + " + str(x_val)
    x_eq_expr = parse_expr(x_eq)
    equations.append(sympy.Eq(x_eq_expr, 0))
    # add all the equations to the list
    for expr in expr_list:
        expr_str = expr.str
        if expr.is_eq == 1:
            expr_str += " + " + str(expr.const)
            expr_str_expr = parse_expr(expr_str)
            equations.append(sympy.Eq(expr_str_expr, 0))
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


