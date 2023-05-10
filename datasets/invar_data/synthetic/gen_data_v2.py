import random
import sympy
import os
from sympy.parsing.sympy_parser import parse_expr

# this file is used to generate a dataset for invariance training

RUN_REAL = False
PRINT_SEPARATELY = True

MAX_DEGREE = 2
MAX_VAR_NUM = 3
MAX_TERM_NUM = 5
MAX_EXPR_NUM = 2
if RUN_REAL:
    SOL_NUM = 512
else:
    SOL_NUM = 4
CONST_MAX = 512
X_MAX = 32 
if RUN_REAL:
    MIN_SOL_NUM = 100
else:
    MIN_SOL_NUM = 3
ENABLE_INEQ = False
EXPERIMENT_TO_RUN = 32

#random.seed(0)
expr_num = random.randint(1, MAX_EXPR_NUM)

assert SOL_NUM >= MIN_SOL_NUM

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


def check_imaginary(sol_list):
    for sol in sol_list:
        for key in sol.keys():
            if sol[key].is_real == False:
                return True
    return False


def get_expr_list():
    expr_list = []
    for i in range(expr_num):
        # generate a single expression
        if ENABLE_INEQ == True:
            is_eq = random.randint(0, 1)
        else:
            is_eq = 1
        term_num = random.randint(1, MAX_TERM_NUM)
        item_list = []
        for j in range(term_num):
            # generate a single item
            # determine the degree of the item
            degree = random.randint(1, MAX_DEGREE)
            coeff = random.randint(1, 10)
            degree_list = []
            # determine the variables in this item and their degrees
            for k in range(MAX_VAR_NUM):
                var_not_included = random.randint(0, 1)
                if var_not_included == 1:
                    degree_list.append(0)
                else:
                    single_degree = random.randint(0, degree)
                    degree_list.append(single_degree)
                    degree = degree - single_degree
            item_list.append(SingleItem(coeff, degree_list))
        # with 50% chance, the const is 0
        const = 0
        if random.randint(0, 1) == 1:
            const = random.randint(0, CONST_MAX)
        expr_list.append(SingleExpr(is_eq, item_list, const))

    # print the expression
    for expr in expr_list:
        expr.gen_str()
        expr.print_expr()
    return expr_list


# return the poly label for the degree
def get_poly_label(degree):
    if degree == 0:
        return "1"
    elif degree == 1:
        return "x"
    elif degree == 2:
        return "x2"
    else:
        # raise an exception
        raise Exception("degree is not supported")


def print_result_to_single_file(expr_list, sol_list, data_point_idx):
    if sol_list.__len__() >= MIN_SOL_NUM:
        # store the equations
        with open("equations.txt", "a+") as f:
            # check if the file is empty or does not exist
            if os.stat("equations.txt").st_size == 0:
                data_point_idx = 0
            else:
                # Move the file pointer to the end of the file
                f.seek(0, 2)  # 2 means seek to the end of the file

                # Find the start of the number
                pos = f.tell()
                while pos > 0:
                    pos -= 1
                    f.seek(pos, 0)  # 0 means seek relative to the start of the file
                    if f.read(1) == ' ':
                        break

                # Read the number and convert it to an integer
                data_point_idx = int(f.readline().rstrip())
            for expr in expr_list:
                f.write(expr.str + "\n")
            f.write("\n")
            data_point_idx += 1
            f.write("idx: " + str(data_point_idx) + "\n")
        # store the solutions
        with open("solutions.txt", "a") as f:
            # if the file is empty, write the variables from the sol in the first line
            for key in sol_list[0][0].keys():
                num_spaces = MAX_DIGIT_WIDTH - len(str(key))
                f.write(" " * num_spaces + str(key))
            f.write("\n")
            for sol in sol_list:
                for key in sol[0].keys():
                    num_spaces = MAX_DIGIT_WIDTH - len(str(sol[0][key]))
                    f.write(" " * num_spaces + str(sol[0][key]))
                f.write("\n")
            f.write("\n")
            f.write("idx: " + str(data_point_idx) + "\n")
        # store the poly lables to the file
        with open("poly_labels.txt", "a") as f:
            num_expr = expr_list.__len__()
            # print "eq" of the quantity of num_expr
            eq_str = "\"eq\", " * num_expr
            eq_str = eq_str[:-2]
            f.write("{\n")
            f.write("  \"eq\": [" + eq_str +"],\n")
            f.write("  \"op\": [\n")
            for expr in expr_list:
                # declare a set {}
                poly = set()
                # always add x since we use w on the RHS
                poly.add("x")
                # iterate through each item in the expr
                for item in expr.item_list:
                    # get its degree_list
                    degree_list = item.degree_list
                    # sum up the degree_list
                    degree_sum = sum(degree_list)
                    poly_label = get_poly_label(degree_sum)
                    # add the poly_label to the set
                    poly.add(poly_label)
                # write the poly to the file
                f.write("    [")
                for idx, poly_label in enumerate(poly):
                    to_print = ""
                    if idx == poly.__len__() - 1:
                        to_print = "\"" + str(poly_label) + "\""
                    else:
                        to_print = "\"" + str(poly_label) + "\", "
                    f.write(to_print)
                f.write("],\n")
            f.write("  ]\n")
            f.write("},\n")
            f.write("\n")
            f.write("idx: " + str(data_point_idx) + "\n")
        return data_point_idx
    else:
        return -1


def print_result_to_separate_file(expr_list, sol_list, data_point_idx):
    # check if the three directories exist
    if not os.path.exists("./equations"):
        os.makedirs("./equations")
    if not os.path.exists("./data"):
        os.makedirs("./data")
    if not os.path.exists("./label"):
        os.makedirs("./label")
    # count how many files there are in the directory of equations
    path = "./equations/"
    num_files = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
    if num_files >= data_point_idx:
        data_point_idx = num_files

    file_name = str(data_point_idx) + ".csv"
    if sol_list.__len__() >= MIN_SOL_NUM:
        # store the equations
        with open("./equations/"+file_name, "w") as f:
            for expr in expr_list:
                f.write(expr.str + "\n")
            data_point_idx += 1
        # store the solutions
        with open("./data/"+file_name, "w") as f:
            # if the file is empty, write the variables from the sol in the first line
            for key in sol_list[0][0].keys():
                num_spaces = MAX_DIGIT_WIDTH - len(str(key))
                f.write(" " * num_spaces + str(key))
            f.write("\n")
            for sol in sol_list:
                for key in sol[0].keys():
                    num_spaces = MAX_DIGIT_WIDTH - len(str(sol[0][key]))
                    f.write(" " * num_spaces + str(sol[0][key]))
                f.write("\n")
        # store the poly lables to the file
        file_name = str(data_point_idx) + ".json"
        with open("./label/"+file_name, "w") as f:
            num_expr = expr_list.__len__()
            # print "eq" of the quantity of num_expr
            eq_str = "\"eq\", " * num_expr
            eq_str = eq_str[:-2]
            f.write("{\n")
            f.write("  \"eq\": [" + eq_str +"],\n")
            f.write("  \"op\": [\n")
            for expr_idx, expr in enumerate(expr_list, start=0):
                # declare a set {}
                poly = set()
                # always add x since we use w on the RHS
                poly.add("x")
                # iterate through each item in the expr
                for item in expr.item_list:
                    # get its degree_list
                    degree_list = item.degree_list
                    # sum up the degree_list
                    degree_sum = sum(degree_list)
                    poly_label = get_poly_label(degree_sum)
                    # add the poly_label to the set
                    poly.add(poly_label)
                # write the poly to the file
                f.write("    [")
                for poly_idx, poly_label in enumerate(poly, start=0):
                    to_print = ""
                    if poly_idx == poly.__len__() - 1:
                        to_print = "\"" + str(poly_label) + "\""
                    else:
                        to_print = "\"" + str(poly_label) + "\", "
                    f.write(to_print)
                if expr_idx == expr_list.__len__() - 1:
                    f.write("]\n")
                else:
                    f.write("],\n")
            f.write("  ]\n")
            f.write("}\n")
        return data_point_idx
    else:
        return -1

# the program begins here
# declare the variables
x, y, z = sympy.symbols("x y z")

w_list = []
for i in range(expr_num):
    w_list.append(sympy.symbols('w{}'.format(i)))


# before run, check if the file exists: equations.txt solutions.txt poly_labels.txt
# if only some of them exist, delete them and regenerate them
all_exist = True
if os.path.exists("poly_labels.txt") == False:
    all_exist = False
if os.path.exists("solutions.txt") == True and all_exist == False:
    os.remove("solutions.txt")
    all_exist = False
if os.path.exists("equations.txt") == True and all_exist == False:
    os.remove("equations.txt")
    all_exist = False

 # instead of solve the equation for solutions,
 # we do in this way:
    # 1. w is always on the rhs of the equaltion
    # 2. we assign random numbers to x, y, z

data_point_num = 0
data_point_idx = 0
MAX_DIGIT_WIDTH = 8
# 16 is the number of data points (a set of equations and inequalities)
#  we want to generate
while data_point_num < EXPERIMENT_TO_RUN: 
    expr_list = get_expr_list()
    # find up to SOL_NUM solutions to the equations
    sol_list = []
    # solve the equations with sympy
    # try to get 512 solutions
    run_num = 0
    while sol_list.__len__() < SOL_NUM:
        run_num += 1
        print("run number: " + str(run_num))
        if run_num > 1000:
            break
        equations = []
        max_xyz = 0
        # assign a random number to x
        x_val = int(random.gauss(0, X_MAX/3))
        max_xyz = max(max_xyz, x_val)
        x_eq = "x + " + str(x_val)
        x_eq_expr = parse_expr(x_eq)
        equations.append(sympy.Eq(x_eq_expr, 0))

        # assign a random number to y
        y_val = int(random.gauss(0, X_MAX/3))
        max_xyz = max(max_xyz, y_val)
        y_eq = "y + " + str(y_val)
        y_eq_expr = parse_expr(y_eq)
        equations.append(sympy.Eq(y_eq_expr, 0))

        # assign a random number to z
        z_val = int(random.gauss(0, X_MAX/3))
        max_xyz = max(max_xyz, z_val)
        z_eq = "z + " + str(z_val)
        z_eq_expr = parse_expr(z_eq)
        equations.append(sympy.Eq(z_eq_expr, 0))

        # add all the equations to the list
        assert expr_list.__len__() == expr_num
        for idx, expr in enumerate(expr_list):
            expr_str = expr.str
            if expr.is_eq == 1:
                expr_str += " - " + str(expr.const)
                expr_str_expr = parse_expr(expr_str)
                equations.append(sympy.Eq(expr_str_expr, w_list[idx]))
        # print all the equations
        for eq in equations:
            print(eq)
        # solve the equations
        sol = sympy.solve(equations, dict=True)
        # skip the sol if it is empty
        if sol.__len__() == 0:
            print("empty solution")
            break
        # if the solution has imaginary number, skip it
        if check_imaginary(sol) == True:
            print ("imaginary number")
            continue
        # if the value of any w in w_list is one order of magnitude larger than x, y, z,
        # skip the solution
        #all_w = True
        #for key in sol[0].keys():
        #    if key[0] == "w":
        #        if sol[0][key] > max_xyz*100:
        #            all_w = False
        #            break
        #if all_w == False:
        #    print("w is too large")
        #    continue
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
        # print the sol
        if all_pass == True:
            # append the solution to the list
            sol_num = sol_list.__len__()
            print("solution found: " + str(sol_num))
            print(sol)
            sol_list.append(sol)

        if PRINT_SEPARATELY == True:
            # print the result to a separate file
            data_point_idx = print_result_to_separate_file(expr_list, sol_list, data_point_idx)
        else:
            data_point_idx = print_result_to_single_file(expr_list, sol_list, data_point_idx)
        data_point_num += 1
        print("data point number: " + str(data_point_num))