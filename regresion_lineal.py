import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

# Lista para almacenar los puntos que el usuario marca
user_points = []

# Pesos y bias iniciales aleatorios para la regresión lineal
w1 = np.random.uniform(-1, 1)
w2 = np.random.uniform(-1, 1)
bias = np.random.uniform(-1, 1)

learning_rate = 0.01  # Tasa de aprendizaje

# Función para calcular la salida de la regresión lineal
def linear_regression(x, w1, w2, bias):
    return w1 * x + bias

# Función para actualizar la gráfica de la regresión
def update_plot():
    global w1, w2, bias

    # Limpiamos la gráfica actual
    ax.clear()

    # Si hay puntos marcados por el usuario, los predecimos
    if user_points:
        user_points_np = np.array(user_points)  # Coordenadas

        # Dibujamos los puntos que el usuario ha ingresado
        for (x, y) in user_points_np:
            ax.plot(x, y, 'o', color='blue')  # Dibuja el punto original

    # Graficamos la línea de regresión
    x_vals = np.linspace(-3, 3, 100)
    y_vals = w1 * x_vals + bias
    ax.plot(x_vals, y_vals, color='green', label='Línea de Regresión')

    # Configuraciones de la gráfica
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Regresión Lineal: w1={w1:.2f}, bias={bias:.2f}')
    ax.legend()
    ax.grid(True)

    # Actualizar la visualización en Tkinter
    canvas.draw()

# Función para manejar los clics en la gráfica
def on_click(event):
    if event.xdata is not None and event.ydata is not None:
        # Guardar el punto que el usuario hizo clic
        user_points.append((event.xdata, event.ydata))
        # Entrenar la regresión lineal con el nuevo punto
        train_regression()
        # Actualizar la gráfica
        update_plot()

# Función para entrenar la regresión lineal con descenso de gradiente
def train_regression():
    global w1, w2, bias

    if user_points:
        # Ajustar los parámetros de la regresión usando todos los puntos
        for x, y in user_points:
            output = linear_regression(x, w1, w2, bias)

            # Calculamos el error como la diferencia entre la salida real y la predicha
            error = y - output

            # Actualizamos los pesos y el bias
            w1 += learning_rate * error * x
            bias += learning_rate * error

# Función para limpiar la gráfica (eliminar todos los puntos)
def clear_plot():
    global user_points, w1, w2, bias
    user_points = []  # Vaciamos la lista de puntos
    w1 = np.random.uniform(-1, 1)  # Reiniciar pesos y bias
    bias = np.random.uniform(-1, 1)
    update_plot()  # Actualizamos la gráfica

# Configuración de la ventana principal
root = tk.Tk()
root.title("Regresión Lineal Interactiva")

# Crear las entradas para el aprendizaje
frame = tk.Frame(root)
frame.pack(pady=20)

# Botón para limpiar la gráfica
button_clear = tk.Button(root, text="Limpiar Gráfica", command=clear_plot)
button_clear.pack()

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

