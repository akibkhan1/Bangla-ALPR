from flask import Flask, render_template, request, redirect, Response, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
# from Yolov3 import VideoCamera
# from CascadeYolov3 import VideoCamera
from CascadeSSD import VideoCamera
# from SSD import VideoCamera
from ocr import *
import glob
import os
from werkzeug.utils import secure_filename
import cv2
import shutil

app = Flask(__name__)

app.config['VIDEO_UPLOADS'] = 'static\\uploads\\'
app.config['ALLOWED_VIDEO_EXTENSIONS'] = ['mp4', 'avi', 'mov']

def allowed_video(filename):
    if not '.' in filename:
        return False
    
    extension = filename.rsplit('.', 1)[1]

    if extension.lower() in app.config['ALLOWED_VIDEO_EXTENSIONS']:
        return True
    else:
        return False

#database integration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///plates.db'
db = SQLAlchemy(app)

class license_plates(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    plate_number = db.Column(db.String(100), nullable=False)
    time_stamp = db.Column(db.DateTime, nullable=False, default=datetime.now)

    def repr(self):
        return 'License Plates' + str(self.id)

@app.route('/')

def index():
    path = 'static/images'
    shutil.rmtree(path)
    os.mkdir(path)
    return render_template('index.html')

@app.route('/upload_video', methods=['GET', 'POST'])

def upload_video():
    if request.method == 'POST':
        if request.files:
            video = request.files['video']

            if video.filename == "":
                print('Video must contain a filename')
                return redirect(request.url)

            if not allowed_video(video.filename):
                print('File format not supported')
                return redirect(request.url)
            else:
                filename = secure_filename(video.filename)
                filepath = os.path.join(app.config['VIDEO_UPLOADS']+filename)
                video.save(filepath)
                print(video.filename +' saved successfully to '+ str(app.config['VIDEO_UPLOADS']))
                return render_template('videofeed.html', filepath=filepath)

    return render_template('upload.html')

@app.route('/table')

def view_history():
    all_posts = license_plates.query.order_by(license_plates.time_stamp).all()
    return render_template('table.html', all_posts=all_posts)

# @app.route('/livefeed')

# def feed():
#     return render_template('videofeed.html')

@app.route('/table/delete/<int:id>')

def delete(id):
    post = license_plates.query.get_or_404(id)
    db.session.delete(post)
    db.session.commit()
    return redirect('/table')

@app.route('/text_feed')

def text_feed():
    text = []
    files = glob.glob('static\processed_images\*')
    if not files:
        text.append("No license plate detected yet")
    else:
        for f in files:
            print(f)
            plate = detectText(f)
            print(plate)
            # if len(plate) >= 21 and plate not in text:
            #     first_part = plate[0:-6]
            #     second_part = plate[-5:]
            #     text.append(first_part + '-' + second_part)
            #     print(first_part + '-' + second_part)
            text.append(plate)
    # if text:
    #     last_plate = text[-1]
    # else:
    #     last_plate = 'Unreadable'
    
    # if not last_plate == 'Unreadable':
    #     new_post = license_plates(plate_number=last_plate)
    #     db.session.add(new_post)
    #     db.session.commit()

    cropped = os.listdir('static/processed_images/')
    if cropped:
        last_image = cropped[-1]
        last = 'static/processed_images/'+str(last_image)
        resized = cv2.imread(last)
        resized = cv2.resize(resized, (640, 480))
        cv2.imwrite(last, resized)
    else:
        last = 'static/dummy.png'

    return jsonify(text=text, image=last)

@app.route('/video_feed')

def video_feed():
    filepath = request.args.get('filepath')
    return Response(gen(VideoCamera(filepath)), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen(camera):
    while True:
        data = camera.get_frame()
        if not data is None:
            frame = data[0]
            yield(b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            break

@app.route('/plate_gallery')

def plate_gallery():
    detected_plate_paths = glob.glob('static/images/*.jpg')
    image_with_text = {}
    image_with_text.clear()
    for plate in detected_plate_paths:
        image_with_text[plate] = get_text(plate)
    return render_template('plate_gallery.html', image_with_text=image_with_text)

def get_text(image_path):
    if not image_path:
        return "Image not found"
    else:
        return detectText(image_path)

if __name__ == "__main__":
    app.run(debug=True)