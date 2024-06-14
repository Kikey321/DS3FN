import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8/yolov8-ASF-DySample-EMSC.yaml')
    model.load('yolov8n.pt') # loading pretrain weights
    model.train(data='ultralytics/cfg/datasets/UAVDT.yaml',
                cache=False,
                imgsz=1536, #1536 1024 800
                epochs=100,
                batch=4, #4 8
                close_mosaic=10,
                workers=2, #4 8
                device='0',
                optimizer='Adam', # using Adam/SGD
                # resume='/root/autodl-fs/yolov8-main/train/yolov8plus/weights/last.pt', # last.pt path
                amp=False, # close amp
                # fraction=0.2,
                project='train',
                name='UAVDT1536',
                )