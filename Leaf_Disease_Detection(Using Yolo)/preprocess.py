import os, json

def create_yolo_labels(data_dir, labels_dir):
    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)

    classes = os.listdir(data_dir)
    class_to_index = {cls: idx for idx, cls in enumerate(classes)}
    
    with open(os.path.join(labels_dir, 'class_to_index.json'), 'w') as json_file:
        json.dump(class_to_index, json_file)


    for cls in classes:
        class_dir = os.path.join(data_dir, cls)
        label_dir = os.path.join(labels_dir, cls)
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)

        for img_file in os.listdir(class_dir):
            if img_file.endswith(('.JPG','.jpg', '.jpeg', '.png')):
                img_path = os.path.join(class_dir, img_file)
                label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + '.txt')

                with open(label_path, 'w') as label_file:
                    # YOLO format: class_id center_x center_y width height
                    # Here we assume the bounding box covers the entire image
                    # You may need to adjust this based on your specific requirements
                    label_file.write(f"{class_to_index[cls]} 0.5 0.5 1.0 1.0\n")
                

if __name__ == "__main__":
    data_dir = './data/images/valid'
    labels_dir = './data/labels/valid'
    create_yolo_labels(data_dir, labels_dir)