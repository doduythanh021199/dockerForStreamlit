import PIL
import cv2
import numpy as np
import tensorflow as tf


def load_class():
    class_id_file = open('class_ids.txt')
    class_ids = {}
    lines = class_id_file.readlines()
    for line in lines:
        id = line.split(' ')[0]
        class_name = " ".join(line.split(' ')[1:])[:-1]
        class_ids[id] = class_name
    return class_ids


def load_model():
    model_dir="saved_model/saved_model"
    model = tf.compat.v2.saved_model.load(str(model_dir))
    return model

def run_inference_for_single_image(model, image):
    class_ids = load_class()
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_copy = image.copy()
    h, w = image.shape[:2]
    input_tensor = tf.convert_to_tensor(image)

    input_tensor = input_tensor[tf.newaxis, ...]

    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    for i in range(num_detections):
        if output_dict["detection_scores"][i] > 0.5:
            [ymin, xmin, ymax, xmax] = output_dict['detection_boxes'][i]
            ymin = int(ymin * h)
            ymax = int(ymax * h)
            xmin = int(xmin * w)
            xmax = int(xmax * w)

            ymin_crop = ymin
            ymax_crop = ymax + int((ymax - ymin) * 0.1)
            xmin_crop = xmin - int((xmax - xmin) * 0.1)
            xmax_crop = xmax + int((xmax - xmin) * 0.1)

            classes = class_ids.get(str(int(output_dict["detection_classes"][i])))
            object_img = image[ymin_crop:ymax_crop, xmin_crop:xmax_crop]
            cv2.imwrite("object/" + str(ymin) + "_" + str(ymax) + "_" + str(xmin) + "_" + str(xmax) + "-" + str(
                classes) + ".jpg", object_img)
            cv2.rectangle(img_copy, (xmin_crop, ymin_crop), (xmax_crop, ymax_crop), (255, 0, 0), 1)
    cv2.imwrite("result.jpg",img_copy)


detection_model = load_model()
image=PIL.Image.open("passport.jpg")

run_inference_for_single_image(detection_model,image)
