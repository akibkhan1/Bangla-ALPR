# License-Plate-Detection-Web-App
This app takes as input a video feed and runs a machine learning model to recognize the characters in the license plate from vehicles.
Open the LP-Detection-WebApp directory

Create virtual environment using the command: 
```
python -m venv work_env
```
Activate the virtual environment using the command: 
```
work_env/Scripts/activate
```
Make sure the path is selected properly. I am using VS Code. So I can check the
`settings.json` file inside `/.vscode` directory. If the path is correct, your 
virtual environment should be running.

Use the following command to download the necessary dependencies: 
```
pip install -r requirements.txt
```
This will not install pytorch and torchvision. Because of dependency issues, we installed them separately. To install them use the following command: 
```
pip install torch==1.7.0+cpu torchvision==0.8.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
```
Now run the web app using the command:
```
python app.py
```
