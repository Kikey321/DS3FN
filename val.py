import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/root/autodl-fs/yolov8-main/train/DIOR/weights/best.pt')
    model.val(data='ultralytics/cfg/datasets/DIOR.yaml',
              split='val',
              imgsz=400,
              batch=4,
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='DIOR640',
              )