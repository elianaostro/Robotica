import numpy as np
import matplotlib.pyplot as plt

N_CELDAS = 20

def predict(bel, command):
    """
    Prediccion del filtro de Bayes discreto.
    command: 'avanzar' o 'retroceder'
    """
    new_bel = np.zeros(N_CELDAS)

    for j in range(N_CELDAS):
        # Para cada celda j, calcular P(x_t = i | u, x_{t-1} = j)
        # y acumular en new_bel[i]
        if command == 'avanzar':
            # Caso normal: 25% quedarse, 50% +1, 25% +2
            if j == N_CELDAS - 1:
                # Ultima celda: 100% quedarse
                new_bel[j] += bel[j]
            elif j == N_CELDAS - 2:
                # Penultima: 25% quedarse, 75% +1
                new_bel[j]     += 0.25 * bel[j]
                new_bel[j + 1] += 0.75 * bel[j]
            else:
                new_bel[j]     += 0.25 * bel[j]
                new_bel[j + 1] += 0.50 * bel[j]
                new_bel[j + 2] += 0.25 * bel[j]

        elif command == 'retroceder':
            if j == 0:
                # Primera celda: 100% quedarse
                new_bel[j] += bel[j]
            elif j == 1:
                # Segunda celda: 25% quedarse, 75% -1
                new_bel[j]     += 0.25 * bel[j]
                new_bel[j - 1] += 0.75 * bel[j]
            else:
                new_bel[j]     += 0.25 * bel[j]
                new_bel[j - 1] += 0.50 * bel[j]
                new_bel[j - 2] += 0.25 * bel[j]

    return new_bel


if __name__ == "__main__":
    # Belief inicial: robot en celda 10 (indice 10, celda 11 si 1-indexed)
    bel = np.hstack((np.zeros(10), 1, np.zeros(9)))

    # 9 comandos de avanzar
    for _ in range(9):
        bel = predict(bel, 'avanzar')

    # 3 comandos de retroceder
    for _ in range(3):
        bel = predict(bel, 'retroceder')

    # Graficar
    plt.figure(figsize=(10, 5))
    plt.bar(range(N_CELDAS), bel, color='steelblue', edgecolor='black')
    plt.xlabel('Celda')
    plt.ylabel('Probabilidad (belief)')
    plt.title('Belief del robot después de 9 avances y 3 retrocesos')
    plt.xticks(range(N_CELDAS))
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("/home/elianaostro/Documents/robotica/tp2/ejercicio4_filtro_discreto.jpg", dpi=150)
    plt.show()

    print("Belief final:")
    for i, b in enumerate(bel):
        if b > 0.001:
            print(f"  Celda {i}: {b:.4f}")
    print(f"Suma total: {np.sum(bel):.6f}")
