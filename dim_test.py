import subprocess
import csv
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os

rcParams.update({'font.size': 18})

k = 1.0
Ncmax = 15

os.makedirs("molecular_model/dim_test", exist_ok=True)
os.makedirs(f"molecular_model/dim_test/k_{k}", exist_ok=True)
'''
# 1. GENERACIÓN DEL ARCHIVO CSV
with open(f"molecular_model/dim_test/k_{k}/dim_analysis_{k}.csv", "w") as f:
    print("Nc,Q^2,P^2,QP,r10,r20,r30,r01,r02,r03,r11,r12,r13,r21,r22,r23,r31,r32,r33,tiempo", file=f)
    f.flush()
    
    out = subprocess.run(["python", "molecular_model/HCE_calc_called_dim_test.py"], capture_output=True, text=True)

    if out.returncode != 0:
        print(f"¡El subproceso falló!")
        print("Motivo del error:")
        print(out.stderr) 
        


    # Guardamos en el archivo (Corregido para que meta salto de línea y sin coma extra al final)
'''    

# 2. LECTURA DEL ARCHIVO CSV
Nc, Q2, P2, QP = [], [], [], []
r10, r20, r30 = [], [], []
r01, r02, r03 = [], [], []
r11, r12, r13 = [], [], []
r21, r22, r23 = [], [], []
r31, r32, r33 = [], [], []
c_columna = [] 

with open(f"molecular_model/dim_test/k_{k}/dim_analysis_{k}.csv", "r") as f:
    lector_csv = csv.reader(f)
    next(lector_csv) 
    
    for fila in lector_csv:
        Nc.append(float(fila[0]))
        Q2.append(float(fila[1]))
        P2.append(float(fila[2]))
        QP.append(float(fila[3]))
        r10.append(float(fila[4]))
        r20.append(float(fila[5]))
        r30.append(float(fila[6]))
        r01.append(float(fila[7]))
        r02.append(float(fila[8]))
        r03.append(float(fila[9]))
        r11.append(float(fila[10]))
        r12.append(float(fila[11]))
        r13.append(float(fila[12]))
        r21.append(float(fila[13]))
        r22.append(float(fila[14]))
        r23.append(float(fila[15]))
        r31.append(float(fila[16]))
        r32.append(float(fila[17]))
        r33.append(float(fila[18]))
        c_columna.append(float(fila[19]))

# 3. GRÁFICO 1x3: Q^2, P^2 y QP
# 3. GRÁFICO 1x3: Q^2, P^2 y QP
from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.pyplot as plt

# 3. GRÁFICO 1x3: Q^2, P^2 y QP con mapa de color por estabilidad

fig1, axes1 = plt.subplots(1, 3, figsize=(33, 5), dpi=600)
vars_main = [Q2, P2, QP]
labels_main = [r"$\langle Q^2 \otimes \hat{\mathbb{I}}_Q\rangle$", r"$\langle P^2 \otimes \hat{\mathbb{I}}_Q\rangle$", r"$\langle QP \otimes \hat{\mathbb{I}}_Q\rangle$"]

# A. Calcular las variaciones medias para Nc >= 10
variaciones_main = []
for var in vars_main:
    # Filtramos los valores correspondientes a Nc >= 10
    var_tail = [v for n, v in zip(Nc, var) if n >= 10]
    
    if len(var_tail) > 1:
        var_media = np.mean(np.abs(np.diff(var_tail)))
    else:
        var_media = 0.0
        
    # Sumamos 1e-15 para evitar errores con el LogNorm si la variación es nula
    variaciones_main.append(var_media + 1e-15)

# B. Definir la escala de colores global para esta figura
norma_color_main = LogNorm(vmin=min(variaciones_main), vmax=max(variaciones_main))

# C. Dibujar los gráficos
for ax, var, label, var_actual in zip(axes1, vars_main, labels_main, variaciones_main):
    # Creamos el array de colores con el mismo valor para todos los puntos del subgráfico
    color_array = [var_actual] * len(Nc)
    
    sc1 = ax.scatter(Nc, var, c=color_array, cmap='viridis', norm=norma_color_main, 
                     s=80, edgecolors='black', zorder=2)
    ax.set_title(f"{label} frente a $N_c$")
    ax.set_xlabel(r"$N_c$")
    ax.set_ylabel(label)
    ax.set_xticks([3, 6, 9, 12, 15])
    ax.grid(True)

# Añadimos la barra de color general a la derecha
cbar1 = fig1.colorbar(sc1, ax=axes1.tolist(), aspect=30, pad=0.02)
cbar1.set_label(r"Variación media para $N_c \geq 10$")

fig1.align_titles()
# En este caso particular, tight_layout sin parámetros no suele chocar con la cbar 
# porque le pasamos la lista de ejes, pero si te diera problemas puedes borrar el tight_layout
#fig1.tight_layout()

plt.savefig(f"molecular_model/dim_test/k_{k}/Q2_P2_QP_{k}.png", bbox_inches='tight')
plt.close(fig1)

import numpy as np
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt

# Organizamos las variables
rhos = [
    [r10, r20, r30],
    [r01, r02, r03],
    [r11, r12, r13],
    [r21, r22, r23],
    [r31, r32, r33]
]
indices = [
    ["10", "20", "30"],
    ["01", "02", "03"],
    ["11", "12", "13"],
    ["21", "22", "23"],
    ["31", "32", "33"]
]

# A. CALCULAMOS LAS VARIACIONES MEDIAS (para Nc >= 10)
variaciones = np.zeros((5, 3))

for i in range(5):
    for j in range(3):
        # Filtramos solo los valores de rho donde Nc >= 10
        rho_tail = [r for n, r in zip(Nc, rhos[i][j]) if n >= 10]
        
        if len(rho_tail) > 1:
            # np.diff calcula la resta entre pasos consecutivos.
            # np.abs saca el valor absoluto y np.mean hace la media de esos saltos.
            var_media = np.mean(np.abs(np.diff(rho_tail)))
        else:
            var_media = 0.0
            
        # Sumamos 1e-15 para evitar errores si la variación es exactamente cero (LogNorm fallaría)
        variaciones[i, j] = var_media + 1e-15

# B. DEFINIMOS LA ESCALA DE COLOR GLOBAL
var_min = np.min(variaciones)
var_max = np.max(variaciones)
norma_color = LogNorm(vmin=var_min, vmax=var_max)

# C. CREAMOS LOS GRÁFICOS
for i in range(5):
    fig2, axes2 = plt.subplots(1, 3, figsize=(33, 5), dpi=600)
    
    for j in range(3):
        ax = axes2[j]
        
        # El valor escalar de variación para ESTE subgráfico en concreto
        var_actual = variaciones[i, j]
        
        # Creamos una lista con el mismo valor repetido tantas veces como puntos haya
        color_array = [var_actual] * len(Nc)
        
        # Dibujamos los puntos. 
        # cmap='plasma' va de azul (baja variación/estable) a amarillo (alta variación/inestable)
        sc2 = ax.scatter(Nc, rhos[i][j], c=color_array, cmap='viridis', norm=norma_color, 
                         s=80, edgecolors='black', zorder=2)
        
        rho_label = r"$\rho_{" + indices[i][j] + r"}$"
        
        ax.set_title(f"{rho_label} frente a $N_c$")
        ax.set_xlabel(r"$N_c$")
        ax.set_ylabel(rho_label)
        ax.set_xticks([3, 6, 9, 12, 15])
        ax.grid(True)

    # Añadimos UNA barra de color general a la derecha de la figura
    # ax=axes2.ravel().tolist() le dice que ajuste el tamaño de los 3 gráficos para hacerle hueco
    cbar = fig2.colorbar(sc2, ax=axes2.ravel().tolist(), aspect=30, pad=0.02)
    cbar.set_label(r"Variación media para $N_c \geq 10$")

    fig2.align_titles()
    #fig2.tight_layout()
    # Con barras de color personalizadas, tight_layout a veces recorta mal, 
    # pero al haber pasado los 'axes2' a colorbar, Matplotlib suele gestionarlo bien.
    
    plt.savefig(f"molecular_model/dim_test/k_{k}/rhos_fila_{indices[i][0]}_{k}.png", bbox_inches='tight')
    plt.close(fig2)

print("Gráficos con gradiente de estabilidad generados con éxito.")