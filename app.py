import json
import os
import zipfile

import cv2
import numpy as np
import copyreg
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['UPLOAD_FOLDER_VERIFY'] = 'uploads/verify'
app.config['TRAINING_FOLDER'] = 'static/data/train'
app.config['PROCESSED_FOLDER'] = 'static/data/processed'
app.config['PREVIEW_FOLDER'] = 'static/'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///iris.sqlite3'
app.config['DATABASE_FOLDER'] = 'instance/'
app.config['SECRET_KEY'] = 'HunterXHunter'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp', 'zip'}
app.config['ADMIN_USERNAME'] = 'admin'
app.config['ADMIN_PASSWORD'] = 'admin'

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER_VERIFY'], exist_ok=True)
os.makedirs(app.config['DATABASE_FOLDER'], exist_ok=True)

db_filename = os.path.join(app.config['DATABASE_FOLDER'], 'iris.sqlite3')

if not os.path.exists(db_filename):
    open(db_filename, 'w').close()

db = SQLAlchemy(app)


class IrisImage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(150))
    path = db.Column(db.String(150))
    keypoints = db.Column(db.Text)
    descriptors = db.Column(db.Text)


def patch_Keypoint_pickiling():
    def _pickle_keypoint(keypoint):
        return cv2.KeyPoint, (
            keypoint.pt[0],
            keypoint.pt[1],
            keypoint.size,
            keypoint.angle,
            keypoint.response,
            keypoint.octave,
            keypoint.class_id,
        )

    copyreg.pickle(cv2.KeyPoint().__class__, _pickle_keypoint)


patch_Keypoint_pickiling()

with app.app_context():
    db.create_all()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


import pickle


def serialize_keypoints(keypoints):
    return pickle.dumps(keypoints)


def deserialize_keypoints(pickled_keypoints):
    return pickle.loads(pickled_keypoints)


# def serialize_keypoints(keypoints):
#     return json.dumps([{
#         'pt': kp.pt,
#         'size': kp.size,
#         'angle': kp.angle,
#         'response': kp.response,
#         'octave': kp.octave,
#         'class_id': kp.class_id
#     } for kp in keypoints])
#
#
# def deserialize_keypoints(json_str):
#     keypoints_data = json.loads(json_str)
#     keypoints = []
#     for kp_data in keypoints_data:
#         kp = cv2.KeyPoint(
#             x=kp_data['pt'][0], y=kp_data['pt'][1],
#             _size=kp_data['size'], _angle=kp_data['angle'],
#             _response=kp_data['response'], _octave=kp_data['octave'],
#             _class_id=kp_data['class_id']
#         )
#         keypoints.append(kp)
#     return keypoints
#

def unzip(file_path, extract_to):
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)


def image_pretreatment(image_path, new_size=(256, 256)):
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Image could not be loaded. Please check the file path.")
        return None

    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray_image, 30, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        mask = np.zeros_like(gray_image)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

        eye_region = cv2.bitwise_and(resized_image, resized_image, mask=mask)

        # Now apply equalization to the masked eye region in grayscale
        gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        equalized_image = cv2.equalizeHist(gray_eye)
        blurred_image = cv2.GaussianBlur(equalized_image, (5, 5), 0)

        return blurred_image
    else:
        print("No contours found that could represent the eye.")
        return None


def process_images_from_folder():
    new_size = (256, 256)
    source_folder = app.config['TRAINING_FOLDER']

    for filename in os.listdir(source_folder):
        if allowed_file(filename):
            file_path = os.path.join(source_folder, filename)
            image = cv2.imread(file_path)

            if image is None:
                continue

            final_image = image_pretreatment(file_path, new_size=new_size)
            sift = cv2.SIFT_create()
            keypoints, descriptors = sift.detectAndCompute(final_image, None)
            serialized_descriptors = descriptors.tobytes()
            serialized_keypoints = serialize_keypoints(keypoints)
            iris_image = IrisImage(filename=filename, path=file_path, keypoints=serialized_keypoints,
                                   descriptors=serialized_descriptors)

            db.session.add(iris_image)
    db.session.commit()


def matching_image_logic(input_image_path):
    best_score = 0
    best_filename = None
    best_path = None
    kp1 = None
    kp2 = None
    mp = None
    input_image = image_pretreatment(input_image_path)
    sift = cv2.SIFT_create()
    input_keypoints, input_descriptors = sift.detectAndCompute(input_image, None)
    if input_descriptors is None:
        return None, None, None, None, None
    existing_images = IrisImage.query.all()
    for image in existing_images:
        existing_descriptors = np.frombuffer(image.descriptors, dtype=np.float32).reshape(-1, 128)
        existing_keypoints = deserialize_keypoints(image.keypoints)
        if existing_descriptors is None or len(existing_descriptors) == 0:
            continue
        index_params = dict(algorithm=1, trees=10)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(input_descriptors, existing_descriptors, k=2)
        match_points = []

        for p, q in matches:
            if p.distance < 0.69 * q.distance:
                match_points.append(p)
        keypoints_count = min(len(input_keypoints), len(existing_keypoints))
        if keypoints_count > 0:
            similarity_score = len(match_points) / keypoints_count
            if similarity_score > best_score:
                best_score = similarity_score * 1.5
                best_score = best_score * (best_score <= 1) + 0.85 * (best_score > 1)
                best_score = round(best_score, 2)
                best_filename = image.filename
                best_path = image.path
                kp1 = input_keypoints
                kp2 = existing_keypoints
                mp = match_points
    if best_score > 0.4:
        return best_filename, best_path, best_score, kp1, kp2, mp
    else:
        return None, None, None, None, None, None


# Helper function to convert serialized descriptors to NumPy array
def deserialize_descriptors(serialized_descriptors):
    return np.frombuffer(serialized_descriptors, dtype=np.float32)


@app.route('/')
def index():
    return redirect('/login')


@app.route('/dashboard', methods=['GET', 'POST'])
def process_folder():
    train_message = ''
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'message': 'No file part'})
        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            train_message = 'Invalid file'
        if file and file.filename.endswith('.zip'):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            unzip(file_path, app.config['TRAINING_FOLDER'])
            process_images_from_folder()

            train_message = 'Training completed successfully'
        if file.filename.endswith('.jpg') or file.filename.endswith('.png') or file.filename.endswith('.jpeg'):
            if file.filename == '' or not allowed_file(file.filename):
                pass
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config['TRAINING_FOLDER'], filename)
            file.save(image_path)
            img = image_pretreatment(image_path)
            if img is None:
                pass
            sift = cv2.SIFT_create()
            keypoints, descriptors = sift.detectAndCompute(img, None)
            if descriptors is not None:
                serialized_keypoints = serialize_keypoints(keypoints)
                serialized_descriptors = descriptors.tobytes()
                iris_image = IrisImage(filename=filename, path=image_path, keypoints=serialized_keypoints,
                                       descriptors=serialized_descriptors)
                db.session.add(iris_image)
                db.session.commit()
                train_message = 'Training completed successfully'

    return render_template('dashboard.html', train_message=train_message)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == app.config['ADMIN_USERNAME'] and password == app.config['ADMIN_PASSWORD']:
            session['username'] = username
            return redirect('/dashboard')
        else:
            return render_template('login.html', error='Invalid username or password')
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))


@app.route('/home', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'image' not in request.files:
            return 'No file part', 400
        file = request.files['image']
        if file.filename == '' or not allowed_file(file.filename):
            return 'Invalid file', 400
        filename = secure_filename(file.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER_VERIFY'], filename)
        file.save(image_path)
        matched_id, matched_image_path, similarity, keypoints, matched_keypoints, mp = matching_image_logic(
            image_path)

        if matched_id is not None:
            session['username'] = matched_id

            # Draw circles around the keypoints in the uploaded image
            uploaded_img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER_VERIFY'], request.files['image'].filename),
                                      cv2.IMREAD_GRAYSCALE)

            sample_image = cv2.imread(image_path)
            sample_image = cv2.resize(sample_image, (256, 256), interpolation=cv2.INTER_AREA)

            match_image = cv2.imread(matched_image_path)
            match_image = cv2.resize(match_image, (256, 256), interpolation=cv2.INTER_AREA)

            sample_image_rgb = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)
            match_image_rgb = cv2.cvtColor(match_image, cv2.COLOR_BGR2RGB)
            match_points_img = cv2.drawMatches(sample_image_rgb, keypoints, match_image_rgb, matched_keypoints, mp,
                                               None)

            matched_img_path = os.path.join(app.config['PREVIEW_FOLDER'], 'matched_keypoints.jpg')
            cv2.imwrite(matched_img_path, match_points_img)
            matched_img_path = matched_img_path.replace('static/', '')

            return render_template('home.html', username=matched_id,
                                   matched_img_path=matched_img_path)
        else:
            return render_template('login.html', user_login_error='No match found')
    return render_template('home.html')


@app.route('/verify_image', methods=['POST'])
def verify_image():
    if 'image' not in request.files:
        return 'No file part', 400
    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        return 'Invalid file', 400
    filename = secure_filename(file.filename)
    image_path = os.path.join(app.config['UPLOAD_FOLDER_VERIFY'], filename)
    file.save(image_path)
    matched_id, image_path, similarity, keypoints, matched_keypoints, mb = matching_image_logic(image_path)
    if similarity is not None:
        similarity = similarity * 100
    if matched_id is not None:
        message = f'Match found with {matched_id} ({similarity:.2f}%)'
    else:
        message = 'No match found'
    return render_template('dashboard.html',
                           match_message=message)


if __name__ == '__main__':
    app.run(debug=True)
