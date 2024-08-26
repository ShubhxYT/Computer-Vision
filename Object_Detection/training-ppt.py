# from ultralytics import YOLO

# # Initialize the model
# model = YOLO('yolov8n.yaml')  # Choose your model architecture

# # Define training parameters
# training_params = {
#     'data': 'ConstructionSiteSafety_Yolov8/data.yaml',  # Path to your data.yaml file
#     'epochs': 100,  # Number of training epochs
#     'imgsz': 640,  # Image size for training
#     'batch': 16,  # Batch size
#     'lr0': 0.001,  # Initial learning rate
#     'lrf': 0.1,  # Final learning rate (fraction of initial)
#     'momentum': 0.937,  # SGD momentum
#     'weight_decay': 0.0005,  # Weight decay
#     'warmup_epochs': 3.0,  # Number of warmup epochs
#     'warmup_momentum': 0.8,  # Warmup initial momentum
#     'warmup_bias_lr': 0.1,  # Warmup initial bias learning rate
#     'box': 0.05,  # Box loss gain
#     'cls': 0.5,  # Classification loss gain
#     'fl_gamma': 0.0,  # Focal loss gamma (efficientDet default is gamma=1.5)
#     'hsv_h': 0.015,  # HSV-Hue augmentation (fraction)
#     'hsv_s': 0.7,  # HSV-Saturation augmentation (fraction)
#     'hsv_v': 0.4,  # HSV-Value augmentation (fraction)
#     'degrees': 0.0,  # Image rotation (+/- deg)
#     'translate': 0.1,  # Image translation (+/- fraction)
#     'scale': 0.5,  # Image scale (+/- gain)
#     'shear': 0.0,  # Image shear (+/- deg)
#     'perspective': 0.0,  # Image perspective (+/- fraction), range 0-0.001
#     'flipud': 0.0,  # Flip up-down (probability)
#     'fliplr': 0.5,  # Flip left-right (probability)
#     'mosaic': 1.0,  # Mosaic augmentation (probability)
#     'mixup': 0.0,  # MixUp augmentation (probability)
#     'copy_paste': 0.0,  # Copy-Paste augmentation (probability)
#     'conf': 0.25,  # Confidence threshold for validating and testing
#     'iou': 0.6,  # IOU threshold for validating and testing
# }

# # Train the model
# model.train(**training_params)
from ultralytics import YOLO
def training_ppt():

    # Initialize the model
    model = YOLO('yolov8n.yaml')  # You can choose different model architectures like yolov8n.yaml, yolov8m.yaml, etc.

    # Train the model
    model.train(data='ConstructionSiteSafety_Yolov8/data.yaml', epochs=100, imgsz=640)
    
if __name__ == '__main__':
    training_ppt()
