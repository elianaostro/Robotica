import numpy as np

class RobotFunctions:

    def __init__(self,):
        pass

    def odometry_motion_model(self, xt, ut, alpha):
        '''
        Odometry motion model recibe:
        xt: estado actual [x, y, theta]
        ut: movimiento [delta_r1, delta_t, delta_r2]
        alpha: parámetros de ruido [alpha1, alpha2, alpha3, alpha4]
        Devuelve el nuevo estado [x_new, y_new, theta_new] con ruido normal aplicado.
        '''
        
        x, y, theta = xt
        delta_r1, delta_t, delta_r2 = ut
        a1, a2, a3, a4 = alpha

        # Agregar ruido normal a los comandos de odometria
        delta_r1_hat = delta_r1 + np.random.normal(0, np.sqrt(abs(a1 * delta_r1**2 + a2 * delta_t**2) + 1e-10))
        delta_t_hat  = delta_t  + np.random.normal(0, np.sqrt(abs(a3 * delta_r1**2 + a3 * delta_r2**2 + a4 * delta_t**2) + 1e-10))
        delta_r2_hat = delta_r2 + np.random.normal(0, np.sqrt(abs(a1 * delta_r2**2 + a2 * delta_t**2) + 1e-10))

        # Calcular nueva pose
        x_new = x + delta_t_hat * np.cos(theta + delta_r1_hat)
        y_new = y + delta_t_hat * np.sin(theta + delta_r1_hat)
        theta_new = theta + delta_r1_hat + delta_r2_hat

        return np.array([x_new, y_new, theta_new])