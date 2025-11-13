import cv2, argparse
parser = argparse.ArgumentParser()
parser.add_argument("--source", "-s", default="0")
args = parser.parse_args()
cap = cv2.VideoCapture(args.source)
print("Opened:", cap.isOpened())
ret, frame = cap.read()
print("Got frame?", ret)
