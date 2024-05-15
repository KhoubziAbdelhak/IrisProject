# Iris Recognition System

This project is an Iris Recognition System implemented in Python using Flask, OpenCV, and SQLAlchemy.

## Description

The Iris Recognition System is a web application that allows users to upload images of eyes for iris recognition. The
system processes the images, extracts features, and matches them against a database of known irises.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/KhoubziAbdelhak/IrisProject.git
```

2. Navigate to the project directory:

```bash
cd IrisProject
```

3. install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the Flask application:

```bash
python app.py
```

2. Open a web browser and navigate to http://localhost:5000.

3. Upload an image or set of images compressed (.zip) to the system.

4. Test the system by uploading a new image and comparing it to the database.