import numpy as np
import matplotlib.pylab as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import math

def funcionObjetivo(x,elec):
    if elec== 1:
        return sum(xi**2 for xi in x) #esfera
    elif elec ==2:
        return -(x[1]+47)*np.sin(np.sqrt(np.abs(x[1]+(x[0]/2)+47)))-x[0]*np.sin(np.sqrt(np.abs(x[0]-(x[1]+47)))) #eggholder
    else:
        return -(1+np.cos(12 * np.sqrt(x[0] **2 + x[1]**2)))/(0.5 * (x[0]**2 + x[1]**2)+2)          #dropwave


def inicializar_poblacion(num_particles, dimension,elec):
    if elec== 2:
        return np.random.uniform(-512,512, (num_particles, dimension)) #eggholder
    else:
        return np.random.uniform(-5, 5, (num_particles, dimension))        #dropwave y esfera
    


def PSO(num_particles, dimension, num_iterations, elec, c1, c2, rand):
    part = inicializar_poblacion(num_particles, dimension,elec)

    # Inicialización de PBest y GBest (PBest Mejor Posición Individual, GBest (Mejor Posición Global):):
    Pbest = np.copy(part)
    best_fitness = np.array([funcionObjetivo(p, elec) for p in Pbest])
    best_position = Pbest[np.argmin(best_fitness)]
    gBest = np.copy(best_position)

    historicoBestF = []

    trayectoria = [np.copy(part)]   #esta lista es para tener la trayectoria de las particulas 
    # Algoritmo PSO
    for it in range(num_iterations):
        # Evaluación de la función objetivo para cada partícula
        for p in range(num_particles):
            fitness = funcionObjetivo(part[p],elec)
            # Actualización de PBest si se encuentra en una mejor posicion 
            if fitness < best_fitness[p]:
                best_fitness[p] = fitness
                Pbest[p] = np.copy(part[p])
        best_index = np.argmin(best_fitness)
        gBest = np.copy(Pbest[best_index])

        # Actualización de las posiciones de las partículas
        for i in range(num_particles):
            # Actualización de la velocidad y posición de la partícula
            v = rand * (c1 * np.random.rand() * (Pbest[i] - part[i]) + c2 * np.random.rand() * (gBest - part[i]))
            part[i] = part[i] + v

        trayectoria.append(np.copy(part))
        historicoBestF.append(np.min(best_fitness))

    # Devuelve la mejor solución encontrada (posición de GBest)
    return gBest, trayectoria, historicoBestF

def printF(num, trayectoria,elec, historicoBestF):

    imagenes = []
    # Visualización de la función objetivo en 3D
    if elec== 2:               #eggholder
        x = np.linspace(-512, 512,100)
        y = np.linspace(-512, 512,100)
        x, y = np.meshgrid(x, y)
        z= funcionObjetivo([x, y],elec)
    else:                       #esfera y dropwave
        x = np.linspace(-5, 5, 100)
        y = np.linspace(-5, 5, 100)
        x, y = np.meshgrid(x, y)
        z = funcionObjetivo([x, y],elec)

    fig = plt.figure()

    ax = fig.add_subplot(121, projection='3d')
    #ax.plot_surface(x, y, z, cmap='coolwarm')
    ax.plot_wireframe(x, y, z, color='r', linewidth=0.1)
    # Etiquetas y título
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z = f(x, y)')
    ax.set_title('Función Objetivo')

    #cont=0
    for tr in trayectoria:
        imagen = ax.scatter3D([tr[i][0] for i in range(num)],
                         [tr[i][1] for i in range(num)],
                         [funcionObjetivo(tr[n],elec) for n in range(num)], c='k')
        imagenes.append([imagen])
        
    animacion = animation.ArtistAnimation(fig, imagenes)
    animacion.save('./psoDrop_wave.gif', writer='pillow') 


    ax2 = fig.add_subplot(122)
    ax2.plot(historicoBestF, linestyle='-', color='b')
    ax2.set_xlabel('Iteración')
    ax2.set_ylabel('Valor de la función objetivo')
    ax2.set_title('Evolución de la Función Objetivo a lo largo de las Iteraciones')

    plt.tight_layout()

    #ax1 = fig.add_subplot(1,2,2)
    #ax1.plot(trayectoria,)
    #ax.set_xlabel('Iteraciones')
    #ax.set_ylabel('Fitness')

    #plt.plot(tr, fitness_values, label='Convergencia del Fitness')
    plt.show()
   

if __name__ == "__main__":
    num_particles = 50  # 30, partículas en la población
    dimension = 2      # Dimensión del espacio de búsqueda
    num_iterations = 100# 100, Número de iteraciones del algoritmo PSO

    elec = int(input("Ingresa el numero de la funcion con la que quieres trabajar (1-esfera, 2-eggholder, 3-dropwave): "))

    best_solution, trayectoria, hbf = PSO(num_particles, dimension, num_iterations, elec,c1=2, c2=2, rand=0.3)
    print("Mejor solución encontrada:", best_solution)

    printF(num_particles, trayectoria,elec, hbf)
