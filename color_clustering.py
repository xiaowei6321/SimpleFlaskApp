from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def pixel_clustering(image_path,n_clusters):
    # Load image and convert to RGB
    image = Image.open(image_path)
    image = image.convert("RGB")

    # Convert image to numpy array
    pixels = np.array(image)

    # Convert 2D pixel data (width x height) to 1D data (each pixel as an RGB value)
    pixels_reshaped = pixels.reshape(-1, 3)

    # Use KMeans clustering to quantize colors into 8 colors
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(pixels_reshaped)

    # Get the cluster label for each pixel
    labels = kmeans.predict(pixels_reshaped)

    # Get the cluster center (the 8 representative colors)
    centroids = kmeans.cluster_centers_

    # Quantize the pixels into 8 colors
    quantized_pixels = centroids[labels].reshape(pixels.shape)

    # Limit the quantized pixel values to 0-255
    quantized_pixels = np.clip(quantized_pixels, 0, 255).astype(np.uint8)

    # Convert the quantized pixel data to an image
    quantized_image = Image.fromarray(quantized_pixels)

    # Save the new image
    quantized_image.save(f"quantized_image_{n_clusters}colors.jpg")

    # Display the image
    quantized_image.show()

    # Optional: Display the center colors of K-means clustering
    plt.imshow([centroids.astype(int)])
    plt.axis('off')
    plt.title(f"{n_clusters} Cluster Colors")
    plt.show()


if __name__ == "__main__":
    
    pixel_clustering("1000.jpg",200)

