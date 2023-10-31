import numpy as np


class CustomGaussianThresholding:

    def __init__(self, image, sigma=1, window_size=11, c=2):
        """
        Inicializamos la clase con la imagen y los parámetros dados.

        Parámetros:
        - image: matriz numpy 2D que representa una imagen en escala de grises.
        - window_size: entero impar que representa el tamaño de la ventana gaussiana.
        - C: Constante restada del umbral calculado.
        """
        self.image = image
        self.window_size = window_size
        self.c = c
        self.sigma = sigma
        self.mask = self.create_gaussian_mask()

    def create_gaussian_mask(self):
        """Create a gaussian mask of given window_size."""
        center = self.window_size // 2
        x, y = np.mgrid[0 - center:self.window_size - center, 0 - center:self.window_size - center]
        g = np.exp(-(x**2 + y**2) / (2*self.sigma**2))
        return g / g.sum()

    def apply_threshold(self):
        """Apply adaptive gaussian thresholding and return the result."""
        result = np.zeros_like(self.image)
        padded_image = np.pad(self.image, (self.window_size // 2, self.window_size // 2), mode='constant')

        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1]):
                window = padded_image[i:i+self.window_size, j:j+self.window_size]
                threshold = np.sum(window * self.mask) - self.c
                result[i, j] = 255 if self.image[i, j] > threshold else 0

        return result

#%%
