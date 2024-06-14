import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('train/ASF-DySample-EMSC/weights/best.pt') # select your model.pt path
    model.predict(source='images/UAVDT/UAVDT2.jpg',
                  imgsz=1536,
                  project='runs/detect',
                  name='VisDrone-UAVDT2',
                  save=True,
                  show_conf=False,
                  show_labels=False,
                  visualize=True # visualize model features maps
                )