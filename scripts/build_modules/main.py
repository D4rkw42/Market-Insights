import os, time, subprocess

# flushing
os.system("cls")

# switch with your favorite version of MSVC
TARGET_COMPILER = "Visual Studio 17 2022"

# select your Python executable
PYTHON_EXECUTABLE = "C:\\Program Files\\Python314\\Python.exe"

# generates stubs for IDE's
def gen_stubs():
    print("[scripts.build_modules]: Generating stubs for modules...")

    # pybind11-stubgen
    status_code = subprocess.call(["pybind11-stubgen", "neural_network", "-o", "bin"])
    
    # os.system("cls")

    # verifying exit code
    if status_code == 0:
        print("[scripts.build_modules]: Success while generating subs!")
    else:
        print('[scripts.build_modules]: Failure while generating subs!')

# overall build command
def build_command():
    # generates all module source from CMakeLists.txt
    commands = [
        f'cmake . -G "{TARGET_COMPILER}" -B build -DPython_EXECUTABLE="${PYTHON_EXECUTABLE}" -DCMAKE_BUILD_TYPE=Release',
        "cmake --build build --config Release"
    ]

    # verifies if build directory existance and then creates it if not exists

    if not os.path.exists("build"):
        os.mkdir("build")

    # running every command

    for cmd in commands:
        os.system("cls")

        print("[scripts.build_modules]: ", cmd) # view
        status_code = os.system(cmd)

        time.sleep(0.2)

        # error handling
        if status_code != 0:
            return print(f"[scripts.build_modules]: Error while running command: {cmd}")
    
    os.system("cls")

    print("[scripts.build_modules]: Success while building modules!")

    gen_stubs() # generating stubs

# command to build every module
def build():
    print("[scripts.build_modules]: Building configuration...")

    # verifies if compiler is Visual Studio
    if not "Visual Studio" in TARGET_COMPILER:
        return print("[scripts.build_modules]: Invalid compiler, must be MSVC!")

     # verifies if build_type is valid
    build_command()

# command to clear build configuration
def clear():
    print("[scripts.build_modules]: Clearing /build...")

    status_code = os.system("rmdir build /s /q")
    os.system("cls")

    if status_code == 0:
        print("[scripts.build_modules]: Cleared.")
    else:
        print("[scripts.build_modules]: /build already cleared!")

if __name__ == "__main__":
    clear()
    build()
