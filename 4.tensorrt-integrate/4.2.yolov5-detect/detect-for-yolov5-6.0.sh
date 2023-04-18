cd yolov5-6.0

python detect.py --weights=../yolov5s.pt --source=../data/car.jpg --iou-thres=0.5 --conf-thres=0.25 --project=../data/
