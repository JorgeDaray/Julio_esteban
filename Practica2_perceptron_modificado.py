# Programa del perceptron mejorado
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

# Lista para almacenar los puntos que el usuario marca
user_points = []

# Pesos y bias del perceptrón aleatorio
random_w1 = np.random.uniform(-1, 1)
random_w2 = np.random.uniform(-1, 1)
random_bias = np.random.uniform(-1, 1)

learning_rate = 0.01  # Tasa de aprendizaje para ajustar el perceptrón aleatorio
training_iterations = 100  # Número de iteraciones de entrenamiento

# Función para calcular la salida del perceptrón
def perceptron(x, y, w1, w2, bias):
    return w1 * x + w2 * y + bias

# Función para actualizar la gráfica del perceptrón
def update_plot():
    w1 = float(entry_w1.get())
    w2 = float(entry_w2.get())
    bias = float(entry_bias.get())

    # Limpiamos la gráfica actual
    ax.clear()

    # Si hay puntos marcados por el usuario, los clasificamos
    if user_points:
        user_points_np = np.array([p[:2] for p in user_points])  # Coordenadas
        outputs = np.array([perceptron(x, y, w1, w2, bias) for x, y in user_points_np])
        random_outputs = np.array([perceptron(x, y, random_w1, random_w2, random_bias) for x, y in user_points_np])
        
        # Dibujamos los marcadores según el botón usado y la salida del perceptrón
        for (i, (x, y, marker)) in enumerate(user_points):
            color = 'blue' if outputs[i] >= 0 else 'red'
            random_color = 'cyan' if random_outputs[i] >= 0 else 'magenta'
            if marker == 'o':  # Punto (clic izquierdo)
                ax.plot(x, y, marker='o', color=color)  # Dibuja un punto azul o rojo
            elif marker == 'x':  # X (clic derecho)
                ax.plot(x, y, marker='x', color=color, markersize=10)  # Dibuja una X azul o roja

            # Dibujar el punto correspondiente del perceptrón aleatorio
            ax.plot(x, y, marker='o', color=random_color, alpha=0.5, markersize=6)  # Perceptrón aleatorio (transparente)

    # Graficamos la frontera de decisión (donde la salida es 0)
    x_vals = np.linspace(-3, 3, 100)
    y_vals = -(w1 * x_vals + bias) / w2
    random_y_vals = -(random_w1 * x_vals + random_bias) / random_w2

    ax.plot(x_vals, y_vals, color='green', label='Frontera del Perceptrón Principal')
    ax.plot(x_vals, random_y_vals, color='purple', linestyle='--', label='Frontera del Perceptrón Aleatorio')

    # Configuraciones de la gráfica
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_title(f'Perceptrón Principal: w1={w1}, w2={w2}, bias={bias}\nPerceptrón Aleatorio: w1={random_w1:.2f}, w2={random_w2:.2f}, bias={random_bias:.2f}')
    ax.legend()
    ax.grid(True)

    # Actualizar la visualización en Tkinter
    canvas.draw()

# Función para manejar los clics en la gráfica
def on_click(event):
    # Agregar el punto donde el usuario hace clic
    if event.xdata is not None and event.ydata is not None:
        # Clic izquierdo (botón 1) para un punto
        if event.button == 1:
            user_points.append((event.xdata, event.ydata, 'o'))  # Guardamos un punto
        # Clic derecho (botón 3) para una X
        elif event.button == 3:
            user_points.append((event.xdata, event.ydata, 'x'))  # Guardamos una X

        update_plot()

# Función para limpiar la gráfica (eliminar todos los puntos)
def clear_plot():
    global user_points
    user_points = []  # Vaciamos la lista de puntos
    update_plot()  # Actualizamos la gráfica

# Función para ajustar el perceptrón aleatorio con pausas entre iteraciones
def train_random_perceptron(iteration=0):
    global random_w1, random_w2, random_bias
    w1 = float(entry_w1.get())
    w2 = float(entry_w2.get())
    bias = float(entry_bias.get())

    if iteration < training_iterations:
        # Ajustamos el perceptrón aleatorio para que se ajuste al perceptrón principal
        for x, y, _ in user_points:
            output_principal = perceptron(x, y, w1, w2, bias)
            output_random = perceptron(x, y, random_w1, random_w2, random_bias)

            # Calculamos el error entre las salidas
            error = output_principal - output_random

            # Actualizamos los pesos y el bias del perceptrón aleatorio
            random_w1 += learning_rate * error * x
            random_w2 += learning_rate * error * y
            random_bias += learning_rate * error

        update_plot()

        # Pausar 100 ms antes de la siguiente iteración
        root.after(100, train_random_perceptron, iteration + 1)

# Configuración de la ventana principal
root = tk.Tk()
root.title("Perceptrón Interactivo")

# Crear las entradas para los pesos y bias
frame = tk.Frame(root)
frame.pack(pady=20)

tk.Label(frame, text="Peso w1:").grid(row=0, column=0)
entry_w1 = tk.Entry(frame)
entry_w1.grid(row=0, column=1)
entry_w1.insert(0, "1")

tk.Label(frame, text="Peso w2:").grid(row=1, column=0)
entry_w2 = tk.Entry(frame)
entry_w2.grid(row=1, column=1)
entry_w2.insert(0, "1")

tk.Label(frame, text="Bias:").grid(row=2, column=0)
entry_bias = tk.Entry(frame)
entry_bias.grid(row=2, column=1)
entry_bias.insert(0, "0")

# Botón para actualizar la gráfica
button_update = tk.Button(root, text="Actualizar Gráfica", command=update_plot)
button_update.pack()

# Botón para limpiar la gráfica
button_clear = tk.Button(root, text="Limpiar Gráfica", command=clear_plot)
button_clear.pack()

# Botón para entrenar el perceptrón aleatorio
button_train = tk.Button(root, text="Entrenar Perceptrón Aleatorio", command=lambda: train_random_perceptron(0))
button_train.pack()

# Crear la figura de Matplotlib
fig, ax = plt.subplots(figsize=(6, 6))

# Integrar Matplotlib en Tkinter
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

# Conectar el evento de clic a la función on_click
canvas.mpl_connect("button_press_event", on_click)

# Inicializar la gráfica
update_plot()

# Iniciar el loop principal de Tkinter
root.mainloop()
