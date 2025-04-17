def main():
    from ultralytics import YOLO
    from pathlib import Path


    # Helper to set nc and names
    # t = '.../Plant_Disease_Dataset/Plant_Disease_Dataset/valid'
    # t = Path(t)
    # print(len(list(t.glob('*'))))
    # print([d.name for d in t.glob('*') if d.is_dir()])
    # exit(0)

    from ultralytics import YOLO

    # Load a model
    model = YOLO("yolo11m.pt")  # build a new model from pre-trained model

    # Train the model
    results = model.train(data="config.yaml", epochs=5)

    '''

    1. Python 3.10 env
    2. pip install ultralytics
    3. pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    4. pip install torchvision --extra-index-url https://download.pytorch.org/whl/cu124
    5. yolo train data=config.yaml model=yolo11n.pt epochs=6 imgsz=128 batch=32 workers=12 device=0 

    On testing hash of best.pt abd last.pt the model created at 6th epoch is considered as best (precision : 0.94081) . The hash of last and best models are checked by hash_check.py
    
    '''
    


if __name__ == '__main__':
    main()