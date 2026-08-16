import subprocess
import os

k=0.5

OUTPUT_PATH = f"molecular_model/results_sdp/{k}"
os.makedirs(OUTPUT_PATH, exist_ok=True)

with open(OUTPUT_PATH+"/datos_WS_10.csv", "w") as f:
    f.write("orden,tiempo,dist\n")

for i in range(11):
        print(f"\n\n\n\nEmpezando orden {i}\n\n\n\n")
        
        out = subprocess.run(["python", "molecular_model/min_cvxpy_3.py", str(i)], text=True)

        if out.returncode != 0:
            print(f"¡El subproceso falló en i={i}!")
            print("Motivo del error:")
            print(out.stderr) 
            continue