import qutip as qt
import numpy as np
import matplotlib.pyplot as plt

input_path = 'molecular_model/'
# Ruta obligatoria en Kaggle para guardar cosas que luego quieras descargar
output_path = 'molecular_model/results/results_0_5_energ_sep/'
k=0.5
# ==============================================================================
# FUNCIONES DE ANÁLISIS
# ==============================================================================

def visualizar_espectro_liouvilliano(evals, order):
    plt.figure(figsize=(10, 8))
    plt.scatter(evals.real, evals.imag, c=evals.real, cmap='viridis_r', s=10, alpha=0.7)
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
    plt.colorbar(label='Parte Real (Tasa de Decaimiento)')
    plt.title(f"Espectro del Liouvilliano (Autovalores) - Orden {order}")
    plt.xlabel("Re(λ)")
    plt.ylabel("Im(λ)")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(output_path+f"results/evals/liouvillian_evals_g{order}.png")
    plt.close()

# Aseguramos de que ultimo_orden está definido
ultimo_orden=4

# ==============================================================================
# BUCLE DE DIBUJADO Y ANÁLISIS DE TEMPERATURAS
# ==============================================================================



with open(output_path+"results/temps/temps.csv", "w") as f_temps:
    for order in range(3, ultimo_orden + 1):
        # --- 3. Búsqueda de la temperatura óptima (beta) ---
        csv_filename = output_path+f"results/temps/distances_full_g{order}.csv"
        
        # Cargamos el HCE de este orden
        HCE = qt.qload(output_path+f"results/steadies/steady_g{order}")
        HCE.dims = [[2,2,5,5],[2,2,5,5]]

        def norm(beta, HCE_matrix):
            # Asegúrate de que los archivos HCE_{beta}_5_diag existen en esa ruta
            rho = qt.qload(input_path+f"HCEs_diag/k_{k}/HCE_{beta:.4f}") 
            rho.dims = [[2,2,5,5],[2,2,5,5]]
            dif = (rho - HCE_matrix)
            # traza de diff al cuadrado
            return (dif * dif).tr().real / 100

        betas = np.logspace(-0.2, 0.2, 50)
        
        with open(csv_filename, "w", encoding='utf-8') as f:
            f.write("beta,dist\n")
            for beta in betas:
                distancia = norm(beta, HCE)
                f.write(f"{beta},{distancia}\n")

        # Leemos el CSV recién creado para interpolar
        with open(csv_filename, "r", encoding='utf-8') as f:
            data = np.loadtxt(f, delimiter=',', skiprows=1)
            bet = data[:,0]
            d = data[:,1]
            
            pol = np.poly1d(np.polyfit(bet, d, 4))
            roots = pol.deriv().roots
            real_roots = roots[np.isreal(roots)].real
            
            # Buscamos raíces dentro del rango de betas
            ext_candidatos = real_roots[(real_roots > 10**(-0.1)) & (real_roots < 10**0.1)]
            
            if len(ext_candidatos) > 0:
                ext = ext_candidatos[0]
                print(f" Mínimo encontrado en beta = {ext:.5f}")
            else:
                ext = bet[np.argmin(d)]
                print(f" No se encontró mínimo analítico, usando el empírico: beta = {ext:.5f}")
            
            fig, ax = plt.subplots(figsize=(10,6))
            ax.plot(bet, pol(bet), label='Ajuste polinómico (grado 4)')
            ax.axvline(x=ext, c="r", ls="--", label=f'Mínimo: {ext:.4f}')
            ax.plot(bet, d, 'o', label='Datos reales')
            ax.set_title(rf"Distancia al estacionario (Orden {order}) en función de $\beta$")
            ax.set_xlabel(r"$\beta$")
            ax.set_ylabel("Distancia de traza")
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()

            plt.savefig(output_path+f"results/temps/steady_g{order}_temps_z_a.png")
            plt.close(fig)

            f_temps.write(f"{order},{ext}\n")

with open(output_path+"results/temps/temps.csv", "r") as f:
    data=np.loadtxt(f, delimiter=',', skiprows=0)
    deg=data[:,0]
    bet=data[:,1]

    fig, ax = plt.subplots(figsize=(10,6))

    ax.axhline(y=1, c="r", ls="--")

    ax.plot(deg, bet, 'o')
    ax.set_title(r"Temperatura del estado estable en función del grado de $d$ y $e$")
    ax.set_xlabel(r"Grado de $d$ y $e$")
    ax.set_ylabel(r"$\beta$")
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.savefig(output_path+"results/temps/temps.png")

import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
for order in range(3, ultimo_orden+1):


    # ==============================================================================
    # 0. CONFIGURACIÓN GENERAL
    # ==============================================================================
    archivo_datos = output_path+f'results/Ks/K_params_g{order}.txt' 
    GRADO_MAX = order 
    dim = GRADO_MAX + 1 

    # ==============================================================================
    # 1. GENERACIÓN DINÁMICA DE LA BASE
    # ==============================================================================
    potencias_pi = []
    potencias_base_spin = []

    # Generamos los bloques desde el grado máximo hasta el 0 automáticamente
    for g in range(GRADO_MAX, -1, -1):
        # Esto crea (g,0), (g-1,1)... hasta (0,g)
        bloque = [(q, g - q) for q in range(g, -1, -1)] 
        
        # Base Pi: intercala dos veces el mismo bloque de orden
        potencias_pi.extend(bloque * 2)
        # Base Spin: acumula un solo bloque
        potencias_base_spin.extend(bloque)

    # Spin repite toda la base completa 16 veces
    potencias_spin = potencias_base_spin * 16

    # ==============================================================================
    # 2. LECTOR DEL ARCHIVO DE RESULTADOS
    # ==============================================================================
    def leer_parametros(filename):
        pesos_totales = np.zeros((dim, dim)) 
        
        with open(filename, 'r', encoding='utf-8') as f:
            lineas = f.readlines()
            
        modo = None
        idx_coeficiente = 0
        
        for linea in lineas:
            linea = linea.strip()
            if "=== OPERADORES DE DERIVADA (Pi) ===" in linea:
                modo = 'Pi'
                continue
            elif "=== OPERADORES DE ESP" in linea:
                modo = 'Spin'
                continue
            elif linea.startswith("---"):
                idx_coeficiente = 0
                continue
            elif linea == "":
                continue
                
            try:
                val_str = linea.replace(' ', '')
                c_val = complex(val_str)
                magnitud = abs(c_val)
                
                if modo == 'Pi':
                    if idx_coeficiente < len(potencias_pi):
                        q, p = potencias_pi[idx_coeficiente]
                        pesos_totales[q, p] += magnitud
                        
                elif modo == 'Spin':
                    if idx_coeficiente < len(potencias_spin):
                        q, p = potencias_spin[idx_coeficiente]
                        pesos_totales[q, p] += magnitud
                    
                idx_coeficiente += 1
                
            except ValueError:
                pass

        return pesos_totales

    # ==============================================================================
    # 3. GENERACIÓN DE GRÁFICOS DINÁMICOS
    # ==============================================================================
    matriz_pesos = leer_parametros(archivo_datos)

    fig = plt.figure(figsize=(18, 5.5))
    rango_ejes = list(range(dim))

    # --- GRÁFICO 1: MAPA DE CALOR 2D ---
    ax1 = fig.add_subplot(131)
    sns.heatmap(matriz_pesos, annot=True, cmap="YlGnBu", fmt=".4f", ax=ax1,
                xticklabels=rango_ejes, yticklabels=rango_ejes,
                cbar_kws={'label': 'Suma de Magnitudes |c|'})
    ax1.invert_yaxis()
    ax1.set_xlabel("Grado de Momento (P)")
    ax1.set_ylabel("Grado de Posición (Q)")
    ax1.set_title(f"Relevancia por Monomio (Grado {GRADO_MAX})")

    # --- GRÁFICO 2: HISTOGRAMA 3D ---
    ax2 = fig.add_subplot(132, projection='3d')
    _x = np.arange(dim)
    _y = np.arange(dim)
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()
    top = matriz_pesos.ravel()
    bottom = np.zeros_like(top)
    width = depth = 0.8

    cmap = plt.get_cmap('viridis')
    max_top = np.max(top) if np.max(top) > 0 else 1.0
    colores = cmap(top / max_top)

    ax2.bar3d(y-width/2, x-depth/2, bottom, width, depth, top, shade=True, color=colores, alpha=0.6)
    ax2.set_xlabel('Grado de P')
    ax2.set_ylabel('Grado de Q')
    ax2.set_zlabel('Magnitud Total')
    ax2.set_title('Histograma 3D del Espacio de Fases')
    ax2.invert_xaxis()
    ax2.set_xticks(range(dim))
    ax2.set_yticks(range(dim))

    # --- GRÁFICO 3: RELEVANCIA POR ORDEN TOTAL ---
    # --- GRÁFICO 3: RELEVANCIA POR ORDEN TOTAL (NORMALIZADO) ---
    ax3 = fig.add_subplot(133)
    pesos_por_orden = []

    # El rango debe ir de 0 hasta GRADO_MAX inclusive
    for orden in range(GRADO_MAX + 1):
        peso_suma = 0
        num_terminos = 0
        
        for q in range(dim):
            for p in range(dim):
                if q + p == orden:
                    peso_suma += matriz_pesos[q, p]
                    num_terminos += 1
        
        if num_terminos > 0:
            peso_medio = peso_suma / num_terminos
        else:
            peso_medio = 0
            
        pesos_por_orden.append(peso_medio)

    sns.barplot(x=rango_ejes, y=pesos_por_orden, ax=ax3, hue=rango_ejes, palette="magma", legend=False)
    ax3.set_xlabel("Orden Total (Q + P)")
    ax3.set_ylabel("Magnitud Acumulada")
    ax3.set_title("Convergencia Perturbativa (Peso por Orden)")

    plt.tight_layout()
    plt.savefig(output_path+f"results/Ks/graphs_{order}.png")
    plt.close()

    # ==============================================================================
    # 1. GENERACIÓN DINÁMICA DE LA BASE
    # ==============================================================================
    potencias_pi = []
    potencias_base_spin = []

    for g in range(GRADO_MAX, -1, -1):
        bloque = [(q, g - q) for q in range(g, -1, -1)] 
        potencias_pi.extend(bloque * 2)
        potencias_base_spin.extend(bloque)

    # Tamaño exacto de un solo operador de Espín (bloque base x 16 matrices de Pauli)
    potencias_spin = potencias_base_spin * 16

    # ==============================================================================
    # 2. LECTOR CON AUTO-RESETEO MODULAR Y SEPARACIÓN DE MODOS
    # ==============================================================================
    def leer_parametros(filename):
        matriz_pi = np.zeros((dim, dim))
        matriz_spin = np.zeros((dim, dim))
        
        with open(filename, 'r', encoding='utf-8') as f:
            lineas = f.readlines()
            
        modo = None
        idx_coeficiente = 0
        
        for linea in lineas:
            linea = linea.strip()
            if "=== OPERADORES DE DERIVADA (Pi) ===" in linea:
                modo = 'Pi'
                idx_coeficiente = 0  # Reset al cambiar de sección
                continue
            elif "=== OPERADORES DE ESP" in linea:
                modo = 'Spin'
                idx_coeficiente = 0  # Reset al cambiar de sección
                continue
            elif linea.startswith("---") or linea == "":
                continue
                
            try:
                val_str = linea.replace(' ', '')
                c_val = complex(val_str)
                magnitud = abs(c_val)
                
                if modo == 'Pi':
                    # El truco del `%` hace que el contador vuelva a 0 al terminar el operador
                    idx_mapeo = idx_coeficiente % len(potencias_pi)
                    q, p = potencias_pi[idx_mapeo]
                    matriz_pi[q, p] += magnitud
                    idx_coeficiente += 1
                        
                elif modo == 'Spin':
                    idx_mapeo = idx_coeficiente % len(potencias_spin)
                    q, p = potencias_spin[idx_mapeo]
                    matriz_spin[q, p] += magnitud
                    idx_coeficiente += 1
                    
            except ValueError:
                pass

        return matriz_pi, matriz_spin

    # ==============================================================================
    # 3. FUNCIÓN REUTILIZABLE PARA GENERAR LA FOTO DE CADA MODO
    # ==============================================================================
    def generar_figura_modo(matriz, titulo_seccion, mode):
        fig = plt.figure(figsize=(18, 5.5))
        rango_ejes = list(range(dim))
        
        # --- GRÁFICO 1: MAPA DE CALOR 2D ---
        ax1 = fig.add_subplot(131)
        sns.heatmap(matriz, annot=True, cmap="YlGnBu", fmt=".4f", ax=ax1,
                    xticklabels=rango_ejes, yticklabels=rango_ejes,
                    cbar_kws={'label': 'Suma de Magnitudes |c|'})
        ax1.invert_yaxis()
        ax1.set_xlabel("Grado de Momento (P)")
        ax1.set_ylabel("Grado de Posición (Q)")
        ax1.set_title(f"{titulo_seccion}\nRelevancia por Monomio")
        
        # --- GRÁFICO 2: HISTOGRAMA 3D ---
        ax2 = fig.add_subplot(132, projection='3d')
        _x = np.arange(dim)
        _y = np.arange(dim)
        _xx, _yy = np.meshgrid(_x, _y)
        x, y = _xx.ravel(), _yy.ravel()
        top = matriz.ravel()
        bottom = np.zeros_like(top)
        width = depth = 0.8
        
        cmap = plt.get_cmap('viridis')
        max_top = np.max(top) if np.max(top) > 0 else 1.0
        colores = cmap(top / max_top)
        
        ax2.bar3d(y-width/2, x-depth/2, bottom, width, depth, top, shade=True, color=colores, alpha=0.6)
        ax2.set_xlabel('Grado de P')
        ax2.set_ylabel('Grado de Q')
        ax2.set_zlabel('Magnitud Total')
        ax2.set_title(f"{titulo_seccion}\nVista Espacial 3D")
        ax2.invert_xaxis()
        ax2.set_xticks(range(dim))
        ax2.set_yticks(range(dim))
            
        
        # --- GRÁFICO 3: RELEVANCIA POR ORDEN TOTAL ---
        ax3 = fig.add_subplot(133)
        pesos_por_orden = []
        
        for orden in range(GRADO_MAX + 1):
            peso_suma = 0
            
            for q in range(dim):
                for p in range(dim):
                    if q + p == orden:
                        # ¡CORRECCIÓN AQUÍ! Usamos 'matriz' en lugar de 'matriz_pesos'
                        peso_suma += matriz[q, p]
            pesos_por_orden.append(peso_suma)
            
        rango_ordenes = list(range(len(pesos_por_orden)))
        sns.barplot(x=rango_ordenes, y=pesos_por_orden, ax=ax3, hue=rango_ordenes, palette="magma", legend=False)
        ax3.set_xlabel("Orden Total (Q + P)")
        ax3.set_ylabel("Magnitud Acumulada")
        ax3.set_title(f"{titulo_seccion}\nConvergencia Perturbativa")
        
        plt.tight_layout()
        plt.savefig(output_path+f"results/Ks/graphs_{order}_{mode}.png")
        plt.close(fig)

    # ==============================================================================
    # 4. EJECUCIÓN DEL PROGRAMA
    # ==============================================================================
    # Leemos los datos (separa automáticamente en dos matrices)
    matriz_pi, matriz_spin = leer_parametros(archivo_datos)

    # Foto 1: Gráficos para los operadores de Derivada
    generar_figura_modo(matriz_pi, "MODO DERIVADA (Pi)", "pi")

    # Foto 2: Gráficos para los operadores de Espín
    generar_figura_modo(matriz_spin, "MODO ESPÍN (Spin)", "spin")