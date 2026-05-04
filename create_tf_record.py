"""
Script to generate TFRecords for helmet detection training.

Usage:
    python create_tf_record.py
"""

import io
import logging
import os
import glob
import random
from lxml import etree
import numpy as np
import PIL.Image
import tensorflow as tf
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util


def dict_to_tf_example(data, label_map_dict, image_dir):
    """Convert XML annotation dict to TFExample proto."""
    height = int(data['size']['height'])
    width = int(data['size']['width'])

    s = data['filename']
    if s[(len(s) - 3):len(s)] == 'png':
        s = s[0:(len(s) - 3)] + 'jpg'

    filename = os.path.join(image_dir, s)

    with tf.gfile.GFile(filename, 'rb') as fid:
        encoded_image_data = fid.read()

    encoded_image_data_io = io.BytesIO(encoded_image_data)
    image = PIL.Image.open(encoded_image_data_io)

    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')

    image_format = b'jpeg'

    xmins, xmaxs, ymins, ymaxs = [], [], [], []
    classes_text, classes = [], []

    if 'object' in data:
        for obj in data['object']:
            if obj:
                xmin = float(obj['bndbox']['xmin'])
                xmax = float(obj['bndbox']['xmax'])
                ymin = float(obj['bndbox']['ymin'])
                ymax = float(obj['bndbox']['ymax'])

                xmins.append(xmin / width)
                ymins.append(ymin / height)
                xmaxs.append(xmax / width)
                ymaxs.append(ymax / height)

                class_name = obj['name']
                classes_text.append(class_name.encode('utf8'))
                classes.append(label_map_dict[class_name])

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))

    return tf_example


def create_tf_record(output_filename, label_map_dict, annotations_dir, image_dir, examples):
    """Create TFRecord file from a list of examples."""
    writer = tf.python_io.TFRecordWriter(output_filename)

    for idx, example in enumerate(examples):
        if idx % 100 == 0:
            logging.info('On image %d of %d', idx, len(examples))

        xml_path = os.path.join(annotations_dir, example + '.xml')

        if not os.path.exists(xml_path):
            logging.warning('Could not find %s, ignoring example.', xml_path)
            continue

        with tf.gfile.GFile(xml_path, 'r') as fid:
            xml_str = fid.read()

        xml = etree.fromstring(xml_str)
        data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

        try:
            tf_example = dict_to_tf_example(data, label_map_dict, image_dir)
            writer.write(tf_example.SerializeToString())
        except ValueError:
            logging.warning('Invalid example: %s, ignoring.', xml_path)

    writer.close()


def main():
    data_dir = 'data/train/'
    label_map_dict = label_map_util.get_label_map_dict(
        'models/faster_RCNN_Inception_v2/helmet_label_map.pbtxt'
    )

    logging.info('Reading from dataset.')
    image_dir = os.path.join(data_dir, 'JPEGImages')
    annotations_dir = os.path.join(data_dir, 'Annotations')

    examples_list = os.listdir(image_dir)
    examples_list = list(map(lambda s: s[0:(len(s) - 4)], examples_list))

    random.seed(42)
    random.shuffle(examples_list)

    num_examples = len(examples_list)
    num_train = int(0.8 * num_examples)
    train_examples = examples_list[:num_train]
    val_examples = examples_list[num_train:]

    logging.info('%d training and %d validation examples.',
                 len(train_examples), len(val_examples))

    train_output_path = os.path.join('data', 'cap_train.record')
    val_output_path = os.path.join('data', 'cap_val.record')

    create_tf_record(train_output_path, label_map_dict,
                     annotations_dir, image_dir, train_examples)
    create_tf_record(val_output_path, label_map_dict,
                     annotations_dir, image_dir, val_examples)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
