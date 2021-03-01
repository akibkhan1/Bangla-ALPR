from flask import Flask, render_template, request, redirect, Response
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from webcam import VideoCamera

app = Flask(__name__)

#database integration
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///posts.db'
# db = SQLAlchemy(app)

@app.route('/')

def index():
    return render_template('index.html')

@app.route('/livefeed')

def feed():
    return render_template('videofeed.html')

@app.route('/video_feed')

def video_feed():
    return Response(gen(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen(camera):
    while True:
        data = camera.get_frame()
        frame = data[0]
        yield(b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

if __name__ == "__main__":
    app.run(debug=True)