import os
from inspect import signature
from PIL import Image
# from ISR.models import RDN, RRDN
from pprint import pprint
from termcolor import colored, cprint
from typing import Iterable, Tuple, Union

from imageai.Classification import ImageClassification
from imageai.Detection import ObjectDetection

# def image_super_resolution(img: Union[object, str], model: str = 'RDN', weights: str = 'psnr-large'):
    # """
    # Produce super-resolution version of image.

    # Parameters:
    #     img (Union[object, str]): Image object, or path to image file.
    #     model (str): Name of ISR model.
    #     weights (str): Name if weights used by ISR model.

    # Returns:
    #     object: Image after ISR.
    # """
    # model_used = None
    # if model == 'RDN':
    #     model_used = RDN(weights=weights)
    # elif model == 'RRDN':
    #     model_used = RRDN(weights=weights)
    # else:
    #     raise ValueError(colored(text=f'Invalid combinatino of model ({model}) and weights ({weights}).', color='yellow')) 

    # img_used = Image.open(fp=img, mode='r') if type(img) == str else img
    # isr_img = model_used.predict(img_used)
    # return isr_img
    
def classify_image(img_path: str, model: str = 'MobileNetV2', n_predictions: int = 1):
    """
    Classifies image with specified DL model and number of predicted labels.

    Parameters:
        img_path (str) - Path to image file.
        model (str) - Name of model for classification. Defaults to 'resnet50_imagenet_tf.2.0.h5'.
        n_predictions (int) - Number of predictions to make, ranked from most probable to least probable. Defaults to 1.
    
    Returns:
        Tuple[Iterable, Iterable] - Predicted labels and probabilities ranked descendingly.
    """

    clf = ImageClassification()
    
    if model == 'MobileNetV2':
        clf.setModelTypeAsMobileNetV2()
        clf.setModelPath(os.path.join(os.getcwd(), './models/mobilenet_v2.h5'))
    elif model == 'ResNet50':
        clf.setModelTypeAsResNet50()
        clf.setModelPath(os.path.join(os.getcwd(), './models/resnet50_imagenet_tf.2.0.h5'))
    elif model == 'InceptionV3':
        clf.setModelTypeAsInceptionV3()
        clf.setModelPath(os.path.join(os.getcwd(), './models/inception_v3_weights_tf_dim_ordering_tf_kernels.h5'))
    elif model == 'DenseNet121':
        clf.setModelTypeAsDenseNet121()
        clf.setModelPath(os.path.join(os.getcwd(), './models/DenseNet-BC-121-32.h5'))
    else:
        raise ValueError(colored(text=f'Invalid model name ({model}).', color='yellow'))

    clf.loadModel()
    predictions, probabilities = clf.classifyImage(image_input=img_path, result_count=n_predictions, input_type='file')
    return predictions, probabilities

def detect_objects_from_image(img_path: str, model: str = 'RetinaNet'):
    """
    Classifies image with specified DL model and number of predicted labels.

    Parameters:
        img_path (str) - Path to image file.
        model (str) - Name of model for classification. Defaults to 'resnet50_imagenet_tf.2.0.h5'.
    
    Returns:
        Tuple[Iterable, Iterable] - Predicted labels and probabilities ranked descendingly.
    """
    
    detector = ObjectDetection()

    if model == 'RetinaNet':
        detector.setModelTypeAsRetinaNet() 
        detector.setModelPath(os.path.join(os.getcwd(), './models/resnet50_coco_best_v2.1.0.h5'))
    elif model == 'AsYOLOv3':
        detector.setModelTypeAsYOLOv3()
        detector.setModelPath(os.path.join(os.getcwd(), './models/yolo.h5'))
    elif model == 'TinyYOLOv3':
        detector.setModelTypeAsTinyYOLOv3()
        detector.setModelPath(os.path.join(os.getcwd(), './models/yolo-tiny.h5'))
    else:
        raise ValueError(colored(text=f'Invalid model name ({model}).', color='yellow'))

    detector.loadModel()
    detections = detector.detectObjectsFromImage(
        input_image=img_path,
        output_image_path=os.path.join('detections/', img_path.split(sep='/')[-1]),
        minimum_percentage_probability=30,
        output_type='file',
        extract_detected_objects=True
    )
    pprint(type(detections))
    return detections

if __name__ == '__main__':
    # pprint(classify_image(img_path='./images/english_village.png', model='MobileNetV2'))
    # pprint(classify_image(img_path='./images/english_village.png', model='DenseNet121'))
    pprint(detect_objects_from_image(img_path='images/english_village.png'))

