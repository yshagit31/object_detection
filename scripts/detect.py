import cv2
import numpy as np
import os

def load_yolo():
    net = cv2.dnn.readNet("data/yolov3.weights", "data/yolov3.cfg")
    with open("data/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    layer_names = net.getLayerNames()
    out_layer_indices = net.getUnconnectedOutLayers()

    if len(out_layer_indices.shape) == 1:
        out_layer_indices = out_layer_indices.flatten()
    elif len(out_layer_indices.shape) == 2:
        out_layer_indices = out_layer_indices[:, 0].flatten()
    else:
        raise ValueError("Unexpected shape for out_layer_indices")

    output_layers = [layer_names[i - 1] for i in out_layer_indices]
    return net, classes, output_layers


def detect_objects(img, net, outputLayers):
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(outputLayers)
    return outs, width, height



def get_box_dimensions(outs, height, width):
    boxes = []
    confidences = []
    class_ids = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    return boxes, confidences, class_ids






def draw_labels(boxes, confidences, class_ids, classes, img):
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0, 255, 0)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            # Calculate the position for the text to be centered vertically on the left side of the rectangle but inside
            text_size = cv2.getTextSize(label, font, 1, 2)[0]
            text_x = x + 5  # 5 pixels padding to the right of the left side of the rectangle
            text_y = y + h // 2 + text_size[1] // 2  # Centered vertically
            cv2.putText(img, label, (text_x, text_y), font, 1, color, 2)


def start():
    net, classes, output_layers = load_yolo()
    for img_file in os.listdir("images/input"):
        if img_file.endswith(".jpg") or img_file.endswith(".png"):
            img_path = os.path.join("images/input", img_file)
            img = cv2.imread(img_path)
            outs, width, height = detect_objects(img, net, output_layers)
            boxes, confidences, class_ids = get_box_dimensions(outs, height, width)
            draw_labels(boxes, confidences, class_ids, classes, img)
            output_path = os.path.join("images/output", img_file)
            cv2.imwrite(output_path, img)
            print(f"Processed {img_file}")




if __name__ == "__main__":
    start()
