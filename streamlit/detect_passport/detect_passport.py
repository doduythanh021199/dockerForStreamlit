import cv2
import numpy as np
import tensorflow as tf


def load_model():
    model_dir = "saved_model/saved_model"
    model = tf.compat.v2.saved_model.load(str(model_dir))
    return model


def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    input_tensor = tf.convert_to_tensor(image)

    input_tensor = input_tensor[tf.newaxis, ...]

    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections
    image = np.asarray(image)
    points = [[0, 0], [0, 0], [0, 0], [0, 0]]

    for i in range(num_detections):
        if output_dict["detection_scores"][i] > 0.9:
            [ymin, xmin, ymax, xmax] = output_dict['detection_boxes'][i]
            ymin = int(ymin * h)
            ymax = int(ymax * h)
            xmin = int(xmin * w)
            xmax = int(xmax * w)

            x = int((xmax + xmin) / 2)
            y = int((ymax + ymin) / 2)
            index = int(output_dict["detection_classes"][i]) - 1
            points[index] = [x, y]

    return image, points


def four_point_transform(image, points):
    (tl, tr, br, bl) = points

    widthA = np.sqrt(((br[0] - bl[0]) * (br[0] - bl[0])) + ((br[1] - bl[1]) * (br[1] - bl[1])))
    widthB = np.sqrt(((tr[0] - tl[0]) * (tr[0] - tl[0])) + ((tr[1] - tl[1]) * (tr[1] - tl[1])))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) * (tr[0] - br[0])) + ((tr[1] - br[1]) * (tr[1] - br[1])))
    heightB = np.sqrt(((tl[0] - bl[0]) * (tl[0] - bl[0])) + ((tl[1] - bl[1]) * (tl[1] - bl[1])))
    maxHeight = max(int(heightA), int(heightB))

    dst = [[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]]
    points = np.float32(points)
    dst = np.float32(dst)
    M = cv2.getPerspectiveTransform(points, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


detection_model = load_model()

image = PIL.Image.open('passport.jpg')
image, points = run_inference_for_single_image(detection_model, image)
image = four_point_transform(image, points)
cv2.imwrite("result.jpg", image)
cv2.imwrite("../object_detection/passport.jpg", image)
