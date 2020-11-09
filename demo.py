import argparse
import glob
import os

import cv2

from yolo import YOLO

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--images', default="images", help='Path to images or image file')
ap.add_argument('-n', '--network', default="normal", help='Network Type: normal / tiny / prn')
ap.add_argument('-d', '--device', default=0, help='Device to use')
ap.add_argument('-s', '--size', default=416, help='Size for yolo')
ap.add_argument('-c', '--confidence', default=0.5, help='Confidence for yolo')
args = ap.parse_args()

classes = ["good", "bad", "none"]

if args.network == "normal":
    print("loading yolov4...")
    yolo = YOLO("models/mask-yolov4.cfg", "models/mask-yolov4.weights", classes)
elif args.network == "prn":
    print("loading yolov3-tiny-prn...")
    yolo = YOLO("models/mask-yolov3-tiny-prn.cfg", "models/mask-yolov3-tiny-prn.weights", classes)
else:
    print("loading yolov4-tiny...")
    yolo = YOLO("models/mask-yolov4-tiny.cfg", "models/mask-yolov4-tiny.weights", classes)

yolo.size = int(args.size)
yolo.confidence = float(args.confidence)

colors = [(0, 255, 0), (0, 165, 255), (0, 0, 255)]

print("extracting tags for each image...")
if args.images.endswith(".txt"):
    with open(args.images, "r") as myfile:
        lines = myfile.readlines()
        files = map(lambda x: os.path.join(os.path.dirname(args.images), x.strip()), lines)
else:
    files = sorted(glob.glob("%s/*.jpg" % args.images))

conf_sum = 0
detection_count = 0

for file in files:
    print(file)
    mat = cv2.imread(file)

    width, height, inference_time, results = yolo.inference(mat)

    print("%s in %s seconds: %s classes found!" %
          (os.path.basename(file), round(inference_time, 2), len(results)))

    output = []

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 848, 640)

    for detection in results:
        id, name, confidence, x, y, w, h = detection
        cx = x + (w / 2)
        cy = y + (h / 2)

        conf_sum += confidence
        detection_count += 1

        # draw a bounding box rectangle and label on the image
        color = colors[id]
        cv2.rectangle(mat, (x, y), (x + w, y + h), color, 2)
        text = "%s (%s)" % (name, round(confidence, 2))
        cv2.putText(mat, text, (x, y - 5), cv2.FONT_HERSHEY_DUPLEX,
                    0.5, color, 1)

        print("%s with %s confidence" % (name, round(confidence, 2)))

        # cv2.imwrite("export.jpg", mat)

    # show the output image
    cv2.imshow('image', mat)
    cv2.waitKey(0)

print("AVG Confidence: %s Count: %s" % (round(conf_sum / detection_count, 2), detection_count))
cv2.destroyAllWindows()
