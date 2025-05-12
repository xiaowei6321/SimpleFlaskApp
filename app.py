from flask import Flask, render_template, request, redirect, url_for, send_file, send_from_directory
import os
from werkzeug.utils import secure_filename
import cv2
from sklearn.cluster import KMeans
import numpy as np
from PIL import Image
from fpdf import FPDF

app = Flask(__name__)

# Set the folder for uploading images and the allowed file types
UPLOAD_FOLDER = 'uploads'
CLUSTERED_FOLDER = 'clustered_images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CLUSTERED_FOLDER'] = CLUSTERED_FOLDER


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}


# Create uploads directory (if it doesn't exist)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# Create clustered_images directory (if it doesn't exist)
os.makedirs(CLUSTERED_FOLDER, exist_ok=True)

# Check if the file extension is valid
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET'])
def index():
    return render_template('upload_display_image.html', image_file=None)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file uploaded", 400
    file = request.files['file']
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return render_template('display_image.html', image_file=filename)
    
    return 'File type not supported', 400

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

@app.route('/clustered_images/<filename>')
def clustered_image(filename):
    return send_file(os.path.join(app.config['CLUSTERED_FOLDER'], filename))

@app.route('/color_clustering', methods=['POST'])
def color_clustering():
    image_file = request.form['image_file']
    n_clusters = int(request.form['n_clusters'])

    # Perform color clustering
    image = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], image_file))

    # Convert to RGB
    pixels = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Flatten the pixels
    pixels_flatten = pixels.reshape(-1, 3)

    # Perform color clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(pixels_flatten)

    # Get the cluster label for each pixel
    labels = kmeans.predict(pixels_flatten)

    # Get the cluster center
    centroids = kmeans.cluster_centers_

    # Quantize the pixels to the cluster center
    quantized_pixels = centroids[labels].reshape(pixels.shape)

    # Limit the quantized pixel values to 0-255
    quantized_pixels = np.clip(quantized_pixels, 0, 255).astype(np.uint8)

    # Save the clustered image
    quantized_image = Image.fromarray(quantized_pixels)
    clustered_image_path = f'clustered_{n_clusters}_{image_file}'
    quantized_image.save(os.path.join(app.config['CLUSTERED_FOLDER'], clustered_image_path))

    # Convert the cluster center to integer RGB colors
    cluster_colors = centroids.round(0).astype(int).tolist()
    

    return render_template('display_both_image.html', image_file=image_file, clustered_image_path=clustered_image_path, cluster_colors=cluster_colors, n_clusters=n_clusters)


@app.route('/export_pdf', methods=['POST'])
def export_pdf():
    # image_file = request.form['image_file']
    # clustered_image_path = request.form['clustered_image_path']
    clustered_image = request.args.get('filename')
    clustered_image_path = os.path.join(app.config['CLUSTERED_FOLDER'], clustered_image)
    img = Image.open(clustered_image_path)
    
    # Create PDF
    pdf = FPDF()
    pdf.add_page()

    pdf.image(clustered_image_path, x=10, y=10, w=190)

    # Save PDF
    pdf_output_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{clustered_image.split(".")[0]}.pdf')
    pdf.output(pdf_output_path)

    return send_from_directory(app.config['UPLOAD_FOLDER'], f'{clustered_image.split(".")[0]}.pdf')

if __name__ == '__main__':
    app.run(debug=True)


