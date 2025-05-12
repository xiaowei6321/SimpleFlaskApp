Here is a sample `README.md` file for your project:

---

# Color Clustering with Flask

This project is a simple Flask web application that allows users to upload an image and perform color clustering using the KMeans algorithm. It then displays both the original and clustered images. Additionally, the clustered image can be saved and exported as a PDF.

## Features

* Upload an image.
* Perform KMeans color clustering on the image.
* Display the original image and the clustered image.
* Show the cluster centers as color boxes.
* Export the clustered image to a PDF file.

## Technologies Used

* **Flask**: Web framework for Python to build the web application.
* **Pillow**: Python Imaging Library to process images.
* **Scikit-learn**: For performing KMeans clustering on the image pixels.
* **Matplotlib**: Used for visualizing the clustering results.
* **OpenCV**: For reading and manipulating images.
* **FPDF**: For generating PDF files.

## Installation

To get started, install the required dependencies.


### Create a virtual environment (optional but recommended):

```bash
python3 -m venv venv
# source venv/bin/activate  
# On Windows use 
venv\Scripts\activate
```

### Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

To run the application, use the following command:

```bash
flask run
```

By default, the app will run on `http://127.0.0.1:5000/`. Open this URL in your browser to access the app.

## How to Use

1. **Upload Image**:

   * Go to the homepage, upload an image, and click on the "Upload" button.
2. **Set Clusters**:

   * Use the `n_clusters` number field to set the number of color clusters you'd like to perform on the image. You can adjust this number and the clustering will be recalculated automatically when the number is changed.
3. **View Results**:

   * After submitting, you will see both the original and clustered images.
   * The clustered image will show the result of the KMeans algorithm applied to the image.
4. **Export to PDF**:

   * Click on the "Export to PDF" button to download the clustered image as a PDF.

## Directory Structure

```
.
├── app.py                # Flask app
├── static/               # Static files (like uploaded images and PDF files)
├── templates/            # HTML templates
│   ├── upload_display_image.html        # The form for uploading
│   ├── display_both_image.html  # Template to display both original and clustered images
│   └── ...
├── uploads/              # Directory for saving uploaded images
├── clustered_images/     # Directory for saving clustered images
├── requirements.txt      # Python dependencies
└── README.md             # Project description and instructions
```


---

Feel free to contact me at 1525025896@qq.com, and customize any other details as needed.


