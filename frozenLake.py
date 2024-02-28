import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import numpy as np
import matplotlib.pyplot as plt
import json

def ejecutar(iteraciones=1000, mostrar=False, entrenamiento=False):
    flag = False

    # Crear el ambiente
    env = gym.make('FrozenLake-v1', desc=generate_random_map(size=4), is_slippery=True, render_mode='human' if mostrar else None)

    # Inicializar la tabla para el valor de q segun el numero de estados y acciones solo si se va a entrenar el modelo
    if(entrenamiento):
        q = np.zeros([env.observation_space.n, env.action_space.n]) # 16 estados y 4 acciones
    else:
        # Cargar la tabla q desde un archivo json
        with open("frozen_lake4x4.json", "r") as f:
            q = np.array(json.load(f))
    
    # Parametros de aprendizaje
    tasaAprendizaje = 0.8 # alpha
    descuento = 0.95 # gamma, debe ser alto ya que solo el ultimo estado es el que tiene recompensa

    # Parametros de exploracion
    epsilon = 1
    epsilonDecay = 0.001
    random = np.random.default_rng()

    recompensaPorIteracion = np.zeros(iteraciones)

    for i in range(iteraciones):
        # Inicializar el ambiente
        estado =  env.reset()[0] # Estado inicial
        terminado = False
        estancado = False

        while(not terminado and not estancado):
            # Escogemos una accion mientras no se haya terminado la ejecucion ni se haya estancado

            # Epsilon-greedy
            # Si el valor aleatorio es mayor a epsilon, escogemos la mejor accion segun la tabla q (para explotar el ambiente)
            # Si no, escogemos una accion aleatoria (para explorar el ambiente)
            if random.random() > epsilon:
                accion = np.argmax(q[estado, :])
            else:
                accion = env.action_space.sample()

            
            nuevoEstado, recompensa, terminado, estancado, _ = env.step(accion)

            # formula de q-learning
            q[estado, accion] = q[estado, accion] + tasaAprendizaje * (
                recompensa + descuento * np.max(q[nuevoEstado, :]) - q[estado, accion]
            )

            estado = nuevoEstado
    
        # Disminuir el valor de epsilon para que el modelo sea menos exploratorio
        epsilon = max(epsilon - epsilonDecay, 0)

        if(epsilon == 0 and not flag):
            flag = True
            print("El modelo ha dejado de explorar")
            tasaAprendizaje = 0 # Desactivar el aprendizaje (solo explotar el ambiente)

        if recompensa == 1:
            recompensaPorIteracion[i] = 1
    
    # Cerrar el ambiente
    env.close()


    # Graficar la recompensa acumulada por cada conjunto de iteraciones (cada 100 iteraciones)
    sum_rewards = np.zeros(iteraciones)
    for t in range(iteraciones):
        sum_rewards[t] = np.sum(recompensaPorIteracion[max(0, t-100):(t+1)])
    plt.plot(sum_rewards)
    plt.xlabel('Iteraciones')
    plt.ylabel('Recompensa acumulada')
    plt.title('Recompensa acumulada por cada conjunto de iteraciones')
    plt.savefig('frozen_lake4x4.png')

    # Guardar la tabla q en un archivo json, donde se guardan los valores de q para cada estado y accion
    with open("frozen_lake4x4.json", "w") as f:
        q_serializable = q.tolist()
        json.dump(q_serializable, f)

if __name__ == "__main__":
    iteraciones = 10000 # numero de iteraciones para entrenar el modelo
    mostrar = False
    entrenamiento = False
    ejecutar(iteraciones, mostrar, entrenamiento)
