import cv2
import numpy as np

def detection(img):
    # Load class names
    with open('models/coco.names', 'r') as f:
        class_names = [line.strip() for line in f.readlines()]

    # Load the network
    net = cv2.dnn.readNet('models/yolov3.weights', 'models/yolov3.cfg')
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    height, width = img.shape[:2]

    # Convert image to blob
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:  # confidence threshold
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = class_names[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
            print(label)
    else:
        print("No objects detected with current thresholds.")
        exit()

    cv2.imshow("result", img)
    cv2.waitKey(0)
        
    return [img, x,y,label,w,h]

img = cv2.imread('images/test.png')
detection(img)