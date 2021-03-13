from flask import Flask, render_template, request, redirect, Response, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from webcam import VideoCamera
from ocr import *
import glob
import os
from werkzeug.utils import secure_filename
import cv2

app = Flask(__name__)

app.config['VIDEO_UPLOADS'] = 'F:\\Contests, Codes and Assignments\\Pioneer Alpha\\Main Project\\yolov5\\static\\uploads\\'
app.config['ALLOWED_VIDEO_EXTENSIONS'] = ['mp4', 'avi', 'mov']

def allowed_video(filename):
    if not '.' in filename:
        return False
    
    extension = filename.rsplit('.', 1)[1]

    if extension.lower() in app.config['ALLOWED_VIDEO_EXTENSIONS']:
        return True
    else:
        return False

# text = "Loading"

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
    # if request.method == 'GET':
    all_posts = license_plates.query.order_by(license_plates.time_stamp).all()
    return render_template('table.html', all_posts=all_posts)
    # return render_template('table.html')

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
    files = glob.glob('F:\Contests, Codes and Assignments\Pioneer Alpha\Main Project\yolov5\static\images\*')
    if not files:
        text.append("No license plate detected yet")
    else:
        for f in files:
            plate = detectText(f)
            if len(plate) >= 21 and plate not in text:
                first_part = plate[0:-6]
                second_part = plate[-5:]
                text.append(first_part + '-' + second_part)
    
    last_plate = text[-1]
    print(last_plate)
    
    new_post = license_plates(plate_number=last_plate)
    db.session.add(new_post)
    db.session.commit()

    cropped = os.listdir('static/images/')
    last_image = cropped[-1]
    last = 'static/images/'+str(last_image)
    resized = cv2.imread(last)
    resized = cv2.resize(resized, (640, 480))
    cv2.imwrite(last, resized)

    return jsonify(text=text, image=last)

@app.route('/video_feed')

def video_feed():
    filepath = request.args.get('filepath')
    # print(filepath)
    return Response(gen(VideoCamera(filepath)), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen(camera):
    while True:
        data = camera.get_frame()
        if not data is None:
            frame = data[0]
            yield(b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

if __name__ == "__main__":
    app.run(debug=True)