import numpy as np
import matplotlib.pyplot as plt
import timeit

# --- Ejercicio 1: Tres funciones de muestreo de N(mu, sigma^2) ---

def normal_suma_uniformes(mu, sigma2, n=1):
    """Genera muestras normales sumando 12 distribuciones uniformes."""
    sigma = np.sqrt(sigma2)
    samples = np.zeros(n)
    for i in range(n):
        s = np.sum(np.random.uniform(0, 1, 12)) - 6  # media 0, var 1
        samples[i] = mu + sigma * s
    return samples


def normal_rechazo(mu, sigma2, n=1):
    """Genera muestras normales usando muestreo con rechazo."""
    sigma = np.sqrt(sigma2)
    # Usamos una uniforme en [-5sigma, 5sigma] como propuesta
    # y la envolvente M * g(x) donde g es uniforme
    samples = []
    # f(x) proporcional a exp(-0.5*((x-mu)/sigma)^2)
    # max de f es 1 (en x=mu), g(x) = 1/(10*sigma)
    # M = 10*sigma (para que M*g(x) >= f(x) en todo el rango)
    while len(samples) < n:
        x = np.random.uniform(mu - 5*sigma, mu + 5*sigma)
        u = np.random.uniform(0, 1)
        fx = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        if u <= fx:
            samples.append(x)
    return np.array(samples)


def normal_box_muller(mu, sigma2, n=1):
    """Genera muestras normales usando Box-Muller."""
    sigma = np.sqrt(sigma2)
    samples = np.zeros(n)
    for i in range(n):
        u1 = np.random.uniform(0, 1)
        u2 = np.random.uniform(0, 1)
        x = np.cos(2 * np.pi * u1) * np.sqrt(-2 * np.log(u2))
        samples[i] = mu + sigma * x
    return samples


if __name__ == "__main__":
    mu, sigma2 = 0, 1
    N = 10000

    # Generar muestras y graficar histogramas
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    methods = [
        ("Suma 12 uniformes", lambda: normal_suma_uniformes(mu, sigma2, N)),
        ("Muestreo con rechazo", lambda: normal_rechazo(mu, sigma2, N)),
        ("Box-Muller", lambda: normal_box_muller(mu, sigma2, N)),
        ("numpy.random.normal", lambda: np.random.normal(mu, np.sqrt(sigma2), N)),
    ]

    for ax, (name, fn) in zip(axes.flat, methods):
        samples = fn()
        ax.hist(samples, bins=50, density=True, alpha=0.7, label=name)
        # Superponer la normal teorica
        x = np.linspace(-4, 4, 200)
        ax.plot(x, (1/np.sqrt(2*np.pi*sigma2)) * np.exp(-0.5*(x-mu)**2/sigma2), 'r-', lw=2)
        ax.set_title(name)
        ax.legend()

    plt.tight_layout()
    plt.savefig("/home/elianaostro/Documents/robotica/tp2/ejercicio1_histogramas.jpg", dpi=150)
    plt.show()

    # Comparar tiempos
    print("\n--- Comparacion de tiempos (10000 muestras, 10 repeticiones) ---")
    for name, fn in methods:
        t = timeit.timeit(fn, number=10)
        print(f"{name:25s}: {t:.4f} s (total 10 reps)")
