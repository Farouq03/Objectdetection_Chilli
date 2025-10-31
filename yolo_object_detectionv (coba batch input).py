import cv2
import numpy as np
import os 
import csv

# Load Yolo
net = cv2.dnn.readNet("./yolov3trained2000.weights", "./yolov3.cfg")
classes = []

with open("./coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names [i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

listdata2 = []

# Loading image
files = os.listdir("./Img_testsample/")
print(files)

for file in files:
    listdata = []
    if file.endswith(".jpg"): 
        img = cv2.imread(os.path.join("./Img_testsample/", file))
        dim = (720,720)
        img = cv2.resize(img, dim, fx=0.4, fy=0.4)
        height, width, channels = img.shape

        # Detecting objects
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        outs = net.forward(output_layers)

        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.25:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.35)
        print(indexes)
        font = cv2.FONT_HERSHEY_PLAIN
        font2 = cv2.FONT_HERSHEY_SIMPLEX

        #Counting Object
        Number = str("Jumlah Cabai Terdeteksi = ")
        count=0
        numdetect=len(indexes)
        chillinum = str(numdetect)
        color = [0, 0, 255]
            
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = [0, 0, 255]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y + 18), font, 2, color, 3) 
                cv2.putText(img, Number+chillinum, (0,700), font2, 1, color, 2)
                
        listdata.append(file)
        listdata.append(chillinum)
        listdata2.append(listdata)

        path = './result/'
        cv2.imwrite(os.path.join(path, file), img)

               
fields = ["Nama Foto", "Jumlah Buah (Deteksi Object)"]   
with open('HasildeteksiSHR.csv','w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f, delimiter = ",")
            # write the data
            writer.writerow(fields)
            for i in listdata2:
                writer.writerow(i)
                
cv2.destroyAllWindows()
