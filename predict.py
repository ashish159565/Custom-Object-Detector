import tensorflow as tf
import cv2

# Load the pre-trained model
model = tf.saved_model.load('C:/Users/ashis/Desktop/ASHISH/PROJECTS/Tensorflow/workspace/training_custom_coco/exported-models/my_model/saved_model')

# Prepare the image data
image = cv2.imread('C:/Users/ashis/Desktop/ASHISH/PROJECTS/Tensorflow/workspace/training_custom_coco/images/test/bed_2.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (640, 480))
image = image / 255.0
image = tf.expand_dims(image, axis=0)

# Run the inference
detections = model.predict(image)

class_names = ['background','bed','chair','sink','toilet']

# Print the results
for i in range(detections['num_detections']):
    class_id = detections['detection_classes'][i]
    class_name = class_names[class_id]
    score = detections['detection_scores'][i]
    bbox = detections['detection_boxes'][i]
    print(f'{class_name}: {score}, bbox: {bbox}')
