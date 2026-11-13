# FaceLock - face-first login

Flow: username -> face check (match + blink) -> password box -> dashboard.

## Setup
1. Python 3.9+ and a webcam.
2. pip install -r requirements.txt
   (face_recognition needs dlib: on Windows install CMake and Visual Studio
   Build Tools first; on Ubuntu: sudo apt install cmake build-essential)
3. python app.py
4. Open http://127.0.0.1:5000/register, sign up, then /login.

The webcam only works on localhost or HTTPS.
Set a strong SECRET_KEY environment variable before any real deployment.
