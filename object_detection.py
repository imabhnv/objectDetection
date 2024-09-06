import cv2

thres = 0.45
nms_threshold = 0.2

cap = cv2.VideoCapture(1)

classNames = []
classFile = "__your_path__"
with open(classFile, 'rt') as f:
    classNames = f.read().strip().split('\n')

configPath = "__your_path__"
weightsPath = "__your_path__"

net=cv2.dnn.DetectionModel(weightsPath,configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:

    success, img = cap.read()
    if not success:
        break

    classIds, confs, bbox = net.detect(img, confThreshold=thres)

    bbox = list(bbox)
    confs = list(map(float, confs))

    indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold) 

    if len(indices) > 0:
        for i in indices.flatten():
            box = bbox[i]
            x, y, w, h = box
            cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
            label = classNames[classIds[i] - 1].upper() 
            cv2.putText(img, label, (box[0] + 10, box[1] + 30),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Output", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
