import numpy as np
import matplotlib.pyplot as plt
from ejercicio1_muestreo import normal_box_muller

def odometry_motion_model(xt, ut, alpha):
    """
    Modelo de movimiento basado en odometria.
    xt: [x, y, theta]
    ut: [delta_r1, delta_r2, delta_t]
    alpha: [alpha1, alpha2, alpha3, alpha4]
    Retorna xt+1 = [x_new, y_new, theta_new]
    """
    x, y, theta = xt
    delta_r1, delta_r2, delta_t = ut
    a1, a2, a3, a4 = alpha

    # Agregar ruido a los comandos de odometria
    delta_r1_hat = delta_r1 + normal_box_muller(0, a1 * delta_r1**2 + a2 * delta_t**2)[0]
    delta_t_hat  = delta_t  + normal_box_muller(0, a3 * delta_r1**2 + a3 * delta_r2**2 + a4 * delta_t**2)[0]
    delta_r2_hat = delta_r2 + normal_box_muller(0, a1 * delta_r2**2 + a2 * delta_t**2)[0]

    # Calcular nueva pose
    x_new = x + delta_t_hat * np.cos(theta + delta_r1_hat)
    y_new = y + delta_t_hat * np.sin(theta + delta_r1_hat)
    theta_new = theta + delta_r1_hat + delta_r2_hat

    return np.array([x_new, y_new, theta_new])


if __name__ == "__main__":
    # Parametros del enunciado
    xt = np.array([2.0, 4.0, 0.0])
    ut = np.array([np.pi/2, 0.0, 1.0])  # [delta_r1, delta_r2, delta_t]
    alpha = np.array([0.1, 0.1, 0.01, 0.01])

    # Generar 5000 muestras
    N = 5000
    samples = np.array([odometry_motion_model(xt, ut, alpha) for _ in range(N)])

    plt.figure(figsize=(8, 8))
    plt.scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.5)
    plt.plot(xt[0], xt[1], 'ro', markersize=8, label='Pose inicial')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Modelo de movimiento basado en odometría (5000 muestras)')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("/home/elianaostro/Documents/robotica/tp2/ejercicio2_odometria.jpg", dpi=150)
    plt.show()
