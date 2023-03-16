# emotionreg
face emoji reg

Thís ís a project from my subject DAP301m at university. Thís project is about human emotion recognition using dataset from AffectNetHQ.

File haarcascade_frontalface_default.xml using haarcascade face recognition to get the bouding boxes and face ROIs.
File Code_DAP.ipynb mentions about data generation, get datasets from folder and push into data loaders. Then file tuning model EfficientNetB0 to get the best parameters for my data.
File demo.py included to deploy webcam 30FPS 720p from my laptop to get face emotion recognition.
File shuffle_array.pth used to split the datasets into train, valid and test data.
