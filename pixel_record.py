# Create TF record for ray style MPIs

import argparse
import os
import numpy as np
import tensorflow as tf
from matplotlib.pyplot import imread
from read_write_model import read_model
import cv2, scipy
from scipy.spatial.transform import Rotation
from timeit import default_timer as timer

def parse_intrinsic(cam, scales):
    i, model, width, height, params = cam
    focals = []
    principles = []
    if model == 'PINHOLE': #undistorted photo
        fx, fy, px, py = params
    elif model == 'SIMPLE_RADIAL': #default colmap distort image
        fx, px, py, _ = params
        fy = fx
    else:
        raise NotImplementedError('Camera model {} isn\'t implement yet')
    scale_x, scale_y = scales
    return [fx*scale_x,fy*scale_y], [px*scale_x,py*scale_y]

def create_images_record(dataset_path, scale = 1.0):
    cameras, images, _ = read_model(os.path.join(dataset_path,'dense/sparse'),'.bin')
    pairs = [(cameras[images[i][3]], images[i]) for i in images]
    records = []
    for cam, img in pairs:
        image_path = os.path.join(dataset_path,'dense/images',img[4])
        if not os.path.exists(image_path):
            raise RuntimeError("Image {} not found in dataset".format(img[4]))
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape 
        real_scale = [1.0, 1.0]
        if scale != 1.0:
            new_height = int(height * scale)
            new_width = int(width * scale)
            real_scale = [new_width / width,new_height / height]
            image = cv2.resize(src, (new_width,new_height))
        focals, principles  = parse_intrinsic(cam, real_scale)
        rotation = Rotation.from_quat(img[1])
        image = image.astype(np.float32)
        image /= 255.0
        records.append({
            'focals': focals,
            'principles': principles, 
            'rotation': rotation.as_dcm(),
            'translation': img[2],
            'pixels': image,
            'width': width,
            'height': height,
            'name': img[4]
        })
    return records   

def floats_feature(values):
    return tf.train.Feature(
        float_list=tf.train.FloatList(value=np.array(values).flatten())
    )
def int64_feature(values):
    return tf.train.Feature(
        int64_list=tf.train.Int64List(value=[values])
    )
def byte_feature(values):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[values])
    )

def write_pixel_tfrecord(path,images_records):
    with tf.python_io.TFRecordWriter(path) as tfrecord_writer, \
        tf.Graph().as_default(), tf.Session() as sess:
        for record in images_records:
            for h in range(record['height']):
                for w in range(record['width']):
                    feature_record = {
                        'pixels': floats_feature(record['pixels'][h,w]),
                        'focals': floats_feature(record['focals']),
                        'principles': floats_feature(record['principles']),
                        'rotation': floats_feature(record['rotation']),
                        'translation': floats_feature(record['translation']),
                        'x': int64_feature(w),
                        'y': int64_feature(h),
                        'height': int64_feature(record['height']),
                        'width': int64_feature(record['width']),
                        'file_name': byte_feature(record['name'].encode('utf-8'))
                    }
                    example = tf.train.Example(features=tf.train.Features(feature=feature_record))
                    tfrecord_writer.write(example.SerializeToString())


def main(args):
    if args.gpus != "":
      os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
      os.environ["CUDA_VISIBLE_DEVICES"]=args.gpus
    images_records = create_images_record(args.input_dir, args.scale)
    pix_path = os.path.join(args.output_dir, args.name + '.pixel')
    write_pixel_tfrecord(pix_path,images_records)

def entry_point():
    parser = argparse.ArgumentParser(description='pixel_record.py - TFrecord for mpis-ray')
    parser.add_argument('-i', '--input-dir', default='./datasets/orchids/', type=str,
        help='path to the input directory, where input images are stored.')
    parser.add_argument('-o', '--output-dir', default='', type=str,
        help='path to the output directory where train and test will be stored. (default: same directroy as input path)')
    parser.add_argument('--name', default='orchids_ray', type=str,
        help='name of the dataset, it will create .train, .test and cfg')
    parser.add_argument('-gpus', default='0', type=str,
        help='CUDA_VISIBLE_DEVICE (default: 0)')
    parser.add_argument('-s','--scale', default=1.0, type=float,
        help='image scale size (default: 1.0) ')
    parser.add_argument('--ref-file', default='IMG_4467.JPG', type=str,
        help='reference camera')
    args = parser.parse_args()
    if args.output_dir == '':
        args.output_dir = args.input_dir
    start_time = timer()
    main(args)
    print("pixel_record finished in {} seconds".format(timer() - start_time))

if __name__ == '__main__':
    entry_point()
