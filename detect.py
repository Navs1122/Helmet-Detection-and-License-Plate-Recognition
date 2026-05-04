"""
Helmet detection inference script using Faster RCNN Inception model.

Usage:
    python detect.py --image_dir data/test/ --output_dir outputs/
"""

import numpy as np
import sys
import os
import argparse
import tensorflow as tf
import cv2
import shutil

sys.path.append("..")
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

PATH_TO_LABELS = 'models/faster_RCNN_Inception_v2/helmet_label_map.pbtxt'
PATH_TO_CKPT = 'models/savedModel/frozen_inference_graph.pb'
NUM_CLASSES = 1

# Load frozen model
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True
)
category_index = label_map_util.create_category_index(categories)


def run_inference_for_single_image(image, graph):
    """Run inference on a single image and return output dict."""
    with graph.as_default():
        with tf.Session() as sess:
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}

            for key in ['num_detections', 'detection_boxes',
                        'detection_scores', 'detection_classes', 'detection_masks']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

            if 'detection_masks' in tensor_dict:
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1]
                )
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8
                )
                tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)

            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
            output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})

            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]

    return output_dict


def detect_single_image(image_path, write_path, filename):
    """Run detection on one image and save the result."""
    print(f"Processing: {image_path}")
    img = cv2.imread(image_path)
    output_dict = run_inference_for_single_image(img, detection_graph)

    img, bndbox_helmet = vis_util.visualize_boxes_and_labels_on_image_array(
        img,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=4,
        min_score_thresh=0.9
    )

    cv2.imwrite(os.path.join(write_path, filename), img)
    return bndbox_helmet


def detect_folder(image_dir, output_dir):
    """Run detection on all images in a folder."""
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    image_list = os.listdir(image_dir)
    results = {}

    for filename in image_list:
        image_path = os.path.join(image_dir, filename)
        bndbox = detect_single_image(image_path, output_dir, filename)
        results[filename] = bndbox

    np.save(os.path.join(output_dir, 'detection_results.npy'), results)
    print(f"\nResults saved to {output_dir}")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Helmet Detection Inference')
    parser.add_argument('--image_dir', default='data/test/', help='Directory of test images')
    parser.add_argument('--output_dir', default='outputs/', help='Directory to save results')
    args = parser.parse_args()

    detect_folder(args.image_dir, args.output_dir)
