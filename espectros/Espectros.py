import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, Button, Frame, filedialog
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class FourierSpectraApp:
    def __init__(self, master):
        self.master = master
        master.title("Análisis de Espectros con Transformada de Fourier")
        master.configure(bg="#2b3e42")  # Fondo verde oscuro

        # Botones
        self.load_button = Button(
            master, text="Cargar Imagen", command=self.load_image,
            bg="#5a7d7c", fg="white", font=("Arial", 10, "bold")
        )
        self.load_button.pack(pady=10)

        self.save_button = Button(
            master, text="Descargar Gráficas", command=self.save_graphs,
            bg="#5a7d7c", fg="white", font=("Arial", 10, "bold")
        )
        self.save_button.pack(pady=10)

        # Marco para el gráfico
        self.canvas_frame = Frame(master, bg="#2b3e42")
        self.canvas_frame.pack(fill="both", expand=True)

        # Crear subplots
        self.figure, self.axes = plt.subplots(3, 3, figsize=(12, 10))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Archivos de Imagen", "*.jpg;*.png;*.bmp")])
        if file_path:
            self.image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            self.process_image()

    def process_image(self):
        # Transformada de Fourier
        fft_image = np.fft.fft2(self.image)
        fft_shift = np.fft.fftshift(fft_image)  # Centraliza bajas frecuencias

        # Espectros 2D
        magnitude_spectrum = np.log(1 + np.abs(fft_shift))
        phase_spectrum = np.angle(fft_shift)
        real_spectrum = np.real(fft_shift)
        imaginary_spectrum = np.imag(fft_shift)

        # Reconstrucción del espectro invertido
        reconstructed_spectrum = np.log(1 + np.abs(np.fft.ifftshift(fft_shift)))

        # Espectro horizontal (línea central)
        center_row = magnitude_spectrum[magnitude_spectrum.shape[0] // 2, :]

        # Mostrar resultados
        self.display_results(
            self.image, magnitude_spectrum, phase_spectrum,
            real_spectrum, imaginary_spectrum, reconstructed_spectrum,
            center_row
        )

    def display_results(self, original, mag, phase, real, imag, reconstructed, freq_row):
        for ax in self.axes.flat:
            ax.clear()

        # Imagen original
        self.axes[0, 0].imshow(original, cmap='gray')
        self.axes[0, 0].set_title("Imagen Original")
        self.axes[0, 0].axis('off')

        # Magnitud del espectro
        self.axes[0, 1].imshow(mag, cmap='jet')
        self.axes[0, 1].set_title("Espectro de Magnitud")
        self.axes[0, 1].axis('off')

        # Fase del espectro
        self.axes[0, 2].imshow(phase, cmap='jet')
        self.axes[0, 2].set_title("Espectro de Fase")
        self.axes[0, 2].axis('off')

        # Parte real
        self.axes[1, 0].imshow(real, cmap='jet')
        self.axes[1, 0].set_title("Parte Real del Espectro")
        self.axes[1, 0].axis('off')

        # Parte imaginaria
        self.axes[1, 1].imshow(imag, cmap='jet')
        self.axes[1, 1].set_title("Parte Imaginaria del Espectro")
        self.axes[1, 1].axis('off')

        # Espectro reconstruido
        self.axes[1, 2].imshow(reconstructed, cmap='jet')
        self.axes[1, 2].set_title("Espectro Reconstruido")
        self.axes[1, 2].axis('off')

        # Gráfica en el dominio de la frecuencia (fila central)
        self.axes[2, 0].plot(freq_row)
        self.axes[2, 0].set_title("Espectro de Frecuencia")
        self.axes[2, 0].set_xlabel("Frecuencia")
        self.axes[2, 0].set_ylabel("Amplitud")

        # Espacios vacíos
        self.axes[2, 1].axis('off')
        self.axes[2, 2].axis('off')

        self.figure.tight_layout()
        self.canvas.draw()

    def save_graphs(self):
        save_path = filedialog.asksaveasfilename(
            defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        if save_path:
            self.figure.savefig(save_path)
            print(f"Gráficas guardadas en: {save_path}")

def main():
    root = Tk()
    app = FourierSpectraApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
    