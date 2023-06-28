from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

helmetModel = YOLO('helmet.pt')  # load a pretrained model (recommended for training)
vehicleModel = YOLO('last.pt')  # load a pretrained model (recommended for training)
def checkInPoinst(frame1,frame2):
  x_frame1_point1 = frame1[0][0]
  y_frame1_point1 = frame1[0][1]
  x_frame1_point2 = frame1[1][0]
  y_frame1_point2 = frame1[1][1]

  x_frame2_point1 = frame1[0][0]
  y_frame2_point1 = frame1[0][1]
  x_frame2_point2 = frame1[1][0]
  y_frame2_point2 = frame1[1][1]

  if x_frame1_point1 in range(x_frame2_point1,x_frame2_point2):
    return True
  
  if y_frame1_point1 in range(y_frame2_point1,y_frame2_point2):
    return True
  
  if x_frame2_point2 in range(y_frame2_point1,y_frame2_point2):
    return True
  
  if y_frame2_point2 in range(y_frame2_point1,y_frame2_point2):
    return True
  
  return False
  

cap = cv2.VideoCapture('no hel.mp4')
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame

  ret, frame = cap.read()
  frame = cv2.resize(frame, (500, 500))
  helmetResult = helmetModel(frame)
  vehicleResult = vehicleModel(frame)
  
  listVehicle = []
  if ret == True:
    
    for bbox in vehicleResult[0].numpy().boxes:
      bboxNumpy = np.ceil(bbox.xyxy).astype(int)
      vehicle = {
        "bbox":[(bboxNumpy[0][0],bboxNumpy[0][1]),(bboxNumpy[0][2],bboxNumpy[0][3])],
        "classPredict": int(bbox.cls[0])
      }
      listVehicle.append(vehicle)

    # Display the resulting frame
    for bbox in helmetResult[0].numpy().boxes:
      bboxNumpy = np.ceil(bbox.xyxy).astype(int)

      if int(bbox.cls[0]) == 1:
        cv2.rectangle(frame, pt1=(bboxNumpy[0][0],bboxNumpy[0][1]), pt2=(bboxNumpy[0][2],bboxNumpy[0][3]), color=(255,0,0), thickness=2)
        for vehicle in listVehicle:
          if checkInPoinst([(bboxNumpy[0][0],bboxNumpy[0][1]),(bboxNumpy[0][2],bboxNumpy[0][3])],vehicle['bbox']):
            point0 = vehicle['bbox'][0]
            point1 = vehicle['bbox'][1]
            
            cv2.rectangle(frame, pt1=point0, pt2=point1, color=(255,0,0), thickness=2)


            frame1 = frame[point0[1]:point1[1],point0[0]:point1[0]]
            # frame1 = frame[14:393,0:16]
            filename = 'image_' + str(random.randint(1, 1000)) + '.jpg'
            cv2.imwrite(filename, cv2.resize(frame1, (224, 224)))
            cv2.imshow('fam1',frame1)

    cv2.imshow('Frame',frame)
 
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
 
  # Break the loop
  else: 
    break
 
# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()