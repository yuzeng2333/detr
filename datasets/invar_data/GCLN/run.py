import os

# Set the argument to pass to the C programs
arg = "100"

# Set the folder name and extension of C programs
folder = "code"
data_folder = "data"
extension = ".c"

# Set the compiler and compiler flags
compiler = "gcc"
compiler_flags = "-o"

# Iterate over all files in the folder with the specified extension
for filename in os.listdir(folder):
    if filename.endswith(extension):
        # Compile the C program with the specified compiler and flags
        program_name = filename[:-len(extension)]
        compile_command = f"{compiler} {folder}/{filename} {compiler_flags} {folder}/{program_name}"
        os.system(compile_command)

        # Run the compiled program with the specified argument and save the output to a text file
        output_filename = f"{program_name}.txt"
        run_command = f"./{folder}/{program_name} {arg} > ./{data_folder}/{output_filename}"
        os.system(run_command)
