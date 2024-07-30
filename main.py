import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np

def load_image(image_path):
    image = Image.open(image_path)
    image = image.convert('RGB')
    return image

def resize_image(image, size=(1920, 1080)):
    return image.resize(size, Image.LANCZOS)

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)

def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

def apply_style_transfer(content_image_path, style_image_path, output_path):
    content_image = load_img(content_image_path)
    style_image = load_img(style_image_path)

    hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]

    result_image = tensor_to_image(stylized_image)
    result_image.save(output_path)

# Paths to input and output images
content_image_path = 'da.jpg'  # Your input screenshot
style_image_path = 'u6aeeo3aza651.jpg'  # Your cartoon reference image
output_image_path = 'stylized_output.jpg'  # Output path for the cartoon-styled thumbnail

apply_style_transfer(content_image_path, style_image_path, output_image_path)
