


# %%
import subprocess

command = """
python 'TFM_MartaRey/InceptionTime/main_folds.py' inception 40_1e5_N TFM_MartaRey/InceptionTime/try2
""" 

print("--------------------------------------------")
print(f'Command: {command}')
try:
    output = subprocess.check_output(command, shell=True)
    # Convert the byte output to a string
    out = output.decode('utf-8').strip()

    print("Output:", out)

except subprocess.CalledProcessError as e:
    print("Error executing command:", e)