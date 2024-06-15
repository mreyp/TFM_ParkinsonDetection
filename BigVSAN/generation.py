

# %%
import subprocess

command = """
python TFM_MartaRey/BigVSAN/inference.py \
--checkpoint_file 'TFM_MartaRey/BigVSAN/bigvsan_10mstep/g_10000000' \
--input_wavs_dir 'TFM_MartaRey/datos/control_files_short_24khz' \
--output_dir 'TFM_MartaRey/datos/generados/pretrained_120_1e5_BigVSAN_generated_control' \
--iterations 120 \
--adjustment_factor 1e-5
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


# %%
import subprocess

command = """
python TFM_MartaRey/BigVSAN/inference.py \
--checkpoint_file 'TFM_MartaRey/BigVSAN/bigvsan_10mstep/g_10000000' \
--input_wavs_dir 'TFM_MartaRey/datos/pathological_files_short_24khz' \
--output_dir 'TFM_MartaRey/datos/generados/pretrained_120_1e5_BigVSAN_generated_pathological' \
--iterations 120 \
--adjustment_factor 1e-5
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

