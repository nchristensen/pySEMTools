import json
import sys

print("Interpreting results ...")

with open('.report.json', 'r') as file:
    data = json.load(file)

exit_code = data['exitcode']

if exit_code == 0:
    print("Tests ran succesfully")
    sys.exit(0)
else:
    for i, test in enumerate(data['tests']):
        if test['outcome'] != 'passed':
            print("===================")
            print(test['nodeid']+" :: Failed")
            stdout = test["call"].get("stdout", 0)
            if stdout != 0:
                print("stdout:")
                print(stdout)
                
    print("Tests failed. Exit code 1. Exiting ...")
    sys.exit(1)