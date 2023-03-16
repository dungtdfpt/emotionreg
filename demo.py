import cv2

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np

matplotlib.style.use('ggplot')
# Required constants.
# ROOT_DIR = '/media/dinh/New Volume/DAP_prj'
# VALID_SPLIT = 0.1
IMAGE_SIZE = 224 # Image size of resize when applying transforms.
# BATCH_SIZE = 4
NUM_WORKERS = 8 # Number of parallel processes for data preparation.

data_classes=['anger','contempt','disgust','fear','happy','neutral','sad','surprise']

device='cuda'

import torchvision.models as models
import torch.nn as nn

def build_model(pretrained=None, fine_tune=False, num_classes=8):
    model = models.efficientnet_b0(weights=pretrained)
    # Change the final classification head.
    # model.classifier = nn.Sequential(nn.Linear(1408, 512), 
    #                                        nn.ReLU(),  
    #                                        nn.Dropout(0.25),
    #                                        nn.Linear(512, 128), 
    #                                        nn.ReLU(),   
    #                                        nn.Dropout(0.25),
    #                                        nn.Linear(128,8),
    #                                        nn.ReLU(),
    #                                        nn.Dropout(0.25),                            
    #                                        nn.Softmax(dim=1))
    model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)
    return model

def get_transform(IMAGE_SIZE):
    valid_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )
    ])
    return valid_transform

args = {'epochs': 20, "pretrained": None, 'learning_rate': 0.0001, 'finetune': False}

model = build_model(
    pretrained=args['pretrained'], 
    fine_tune= args["finetune"], 
    num_classes=len(data_classes)
).to(device)

model.load_state_dict(torch.load('model_pretrained_best_valid_acc.pth')['model_state_dict'])


path='haarcascade_frontalface_default.xml'

front_scale=1.5

font=cv2.FONT_HERSHEY_PLAIN

rectangle_bgr=(255,255,255)


cap=cv2.VideoCapture(0)

transforms = get_transform(IMAGE_SIZE=IMAGE_SIZE)

while True:
    ret,frame=cap.read()
    faceCascade=cv2.CascadeClassifier(path)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    gray = cv2.equalizeHist(gray)
    # frame = cv2.equalizeHist(frame)

    faces=faceCascade.detectMultiScale(gray,1.1,10)
    if len(faces) >0:
        for x,y,w,h in faces:
            roi_gray=gray[y:y+h,x:x+w]
            roi_color=frame[y:y+h,x:x+w]
            #cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            facess=faceCascade.detectMultiScale(roi_gray)
            if len(facess)==0:
                # print('Face not detected')
                face_roi = roi_color
            else:
                for (ex,ey,ew,eh) in facess:
                    face_roi=roi_color[ey:ey+eh,ex:ex+ew]

        # graytemp=cv2.cvtColor(face_roi,cv2.COLOR_BGR2GRAY)
        # final_image=cv2.resize(roi_color,(IMAGE_SIZE, IMAGE_SIZE))
        # dataa=torch.from_numpy(final_image)
        # dataa=dataa.permute(2,0,1)
        
        # dataa=dataa.type(torch.FloatTensor)
        face_roi = Image.fromarray(face_roi)
        dataa = transforms(face_roi)
        dataa=torch.unsqueeze(dataa,0)
        dataa=dataa.to(device)
        
        pred=model(dataa)
        pred = torch.softmax(pred, dim = -1)

        pred=(pred/torch.sum(pred) * 100 + 30).astype(int)
        pred_idx=torch.argmax(pred)   
        
        if pred_idx.item()==0:
            status=f'{data_classes[0]}:{torch.max(pred).item():.2f}'
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.putText(frame,status,(x,y+h+30),font,2,(0,0,255),2,cv2.LINE_4)
            
        elif pred_idx.item()==1:
            status=f'{data_classes[1]}:{torch.max(pred).item():.2f}'
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.putText(frame,status,(x,y+h+30),font,2,(0,0,255),2,cv2.LINE_4)

        elif pred_idx.item()==2:
            status=f'{data_classes[2]}:{torch.max(pred).item():.2f}'
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.putText(frame,status,(x,y+h+30),font,2,(0,0,255),2,cv2.LINE_4)

        elif pred_idx.item()==3:
            status=f'{data_classes[3]}:{torch.max(pred).item():.2f}'
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.putText(frame,status,(x,y+h+30),font,2,(0,0,255),2,cv2.LINE_4)

        elif pred_idx.item()==4:
            status=f'{data_classes[4]}:{torch.max(pred).item():.2f}'
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.putText(frame,status,(x,y+h+30),font,2,(0,0,255),2,cv2.LINE_4)

        elif pred_idx.item()==5:
            status=f'{data_classes[5]}:{torch.max(pred).item():.2f}'
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.putText(frame,status,(x,y+h+30),font,2,(0,0,255),2,cv2.LINE_4)

        elif pred_idx.item()==6:
            status=f'{data_classes[6]}:{torch.max(pred).item():.2f}'
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.putText(frame,status,(x,y+h+30),font,2,(0,0,255),2,cv2.LINE_4)

        elif pred_idx.item()==7:
            status=f'{data_classes[7]}:{torch.max(pred).item():.2f}'
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.putText(frame,status,(x,y+h+30),font,2,(0,0,255),2,cv2.LINE_4)
    

    cv2.imshow('Face emotion Recognition',frame)    

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()


