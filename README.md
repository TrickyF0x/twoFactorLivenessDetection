# twoFactorLivenessDetection
Created as master degree final project, using CNN and blink detection to prevent face spoofing

App could prevent authentication with face spoofing by photo or video (maybe printed 3d mask, not yet tested)

How to use:

Upload project and install all requirments from txt (should be edited: pipwin)

Find and add shape_predictor_68_face_landmarks.dat to face_detector folder

Put fake and real videos in videos folder

Collect dataset by "python modules/gather_examples.py --input ./videos/fake.mp4 --output ./dataset/fake --detector ./face_detector --skip 1" in terminal and "python modules/gather_examples.py --input ./videos/real.mp4 --output ./dataset/real --detector ./face_detector --skip 2"

Train the model and get encoder by "python modules/train.py --dataset ./dataset --model ./models/liveness.model --le ./models/le.pickle"

Run ArgusApp.py, create new user, test authentication

More accuracy may be provided by using good webcam and bigger dataset consisting of several different peoples


