import pandas, cv2
import tensorflow as tf
import numpy as np
import math

def convert_reference(reference):
    ref_str = str(reference)
    if reference < 10:
        index = '0000'
    elif reference < 100:
        index = '000'
    else:
        index = '00'
    return index + ref_str


def read_images(index):
    depth_filename = "/Users/georgestoica/Desktop/Research/Mini-Project/vkitti_1.3.1_depthgt/0001/clone/{}.png".format(index)
    image_filename = "/Users/georgestoica/Desktop/Research/Mini-Project/vkitti_1.3.1_rgb/0001/clone/{}.png".format(index)
    depth = np.reshape(cv2.imread(depth_filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH), (375, 1242, 1))
    image = cv2.imread(image_filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    return image, depth

def downsize_image(image, max_height, max_width, num_channels):
    downsized_image = tf.image.resize_image_with_crop_or_pad(image, max_height, max_width)
    downsized_image.set_shape([max_height, max_width, num_channels])
    downsized_image_encoded = tf.image.encode_png(downsized_image)
    return downsized_image_encoded



def main(unusedargv):
    NUM_IMAGES = 250
    max_height, max_width = 188, 621
    init_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init_op)
    for reference in range(NUM_IMAGES):
        image_index = convert_reference(reference)
        image, depth = read_images(image_index)
        # print("image shape: " + str(image.shape))
        image_downsized = downsize_image(image, max_height, max_width, num_channels = 3)
        depth_downsized = downsize_image(depth, max_height, max_width, num_channels = 1)

        image_fpath = tf.constant("/Users/georgestoica/Desktop/Research/Mini-Project/ds_image/{}.png".format(image_index))
        depth_fpath = tf.constant("/Users/georgestoica/Desktop/Research/Mini-Project/ds_depth/{}.png".format(image_index))

        image_write = tf.write_file(image_fpath, image_downsized)
        depth_write = tf.write_file(depth_fpath, depth_downsized)
        dummy_ = sess.run(image_write)
        dummy_ = sess.run(depth_write)








if __name__ == "__main__":
    tf.app.run()


