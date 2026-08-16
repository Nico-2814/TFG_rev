import subprocess
import csv
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'font.size': 16})

k = 0.5
puntos = [50, 100, 200, 300, 500, 1000, 2000, 3000, 5000]
'''
with open(f"molecular_model/trace_analysis_{k}.csv", "w") as f:
    print("Puntos,Z,tiempo", file=f)
    for i in puntos:
        out = subprocess.run(["python", "molecular_model/HCE_calc_called.py", str(i)], capture_output=True, text=True)

        if out.returncode != 0:
            print(f"¡El subproceso falló en i={i}!")
            print("Motivo del error:")
            print(out.stderr) 
            continue

        values = out.stdout.strip().split(",")
        print(f"Progreso: i={i}, Valores={values}") 

        # Guardamos en el archivo
        print(i, values[0], values[1], sep=",", file=f)
'''

x_puntos = []
y_hce = []
c_columna = [] # Esta lista controlará los colores

with open(f"molecular_model/trace_analysis_{k}.csv", "r") as f:
    lector_csv = csv.reader(f)
    next(lector_csv) 
    
    for fila in lector_csv:
        x_puntos.append(float(fila[0]))
        y_hce.append(float(fila[1]))
        c_columna.append(float(fila[2]))

fig, ax = plt.subplots(figsize=(9, 6))

ax.plot(x_puntos, y_hce, color='gray', linestyle='--', alpha=0.5, zorder=1)

puntos_scatter = ax.scatter(
    x_puntos, 
    y_hce, 
    c=c_columna,          # La 3ª columna dicta el color
    cmap='viridis',        # Paleta de color (puedes probar 'plasma', 'inferno', 'coolwarm')
    s=80,                  # Tamaño de los puntos
    edgecolors='black',    # Borde negro para que destaquen mejor
    zorder=2
)

cbar = fig.colorbar(puntos_scatter, ax=ax)
cbar.set_label("Tiempo de ejecución (s)") # Cambia esta etiqueta si la 3ª columna representa otra cosa


plt.title(r"$Z$ frente a $N_{points}$")
plt.xlabel(r"$N_{points}$")
plt.ylabel(r"$Z$")
plt.grid(True)

#plt.tight_layout()
plt.savefig(f"molecular_model/Z_puntos_{k}.png")