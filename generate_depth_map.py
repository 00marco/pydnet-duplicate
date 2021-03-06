#
# MIT License
#
# Copyright (c) 2018 Matteo Poggi m.poggi@unibo.it
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import tensorflow as tf
import sys
import os
import argparse
import time
import datetime
from utils import *
from pydnet import *
import png

# forces tensorflow to run on CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

parser = argparse.ArgumentParser(description='Argument parser')

""" Arguments related to network architecture"""
parser.add_argument('--width', dest='width', type=int, default=512, help='width of input images')
parser.add_argument('--height', dest='height', type=int, default=256, help='height of input images')
parser.add_argument('--resolution', dest='resolution', type=int, default=3, help='resolution [1:H, 2:Q, 3:E]')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', type=str, default='checkpoint/IROS18/pydnet', help='checkpoint directory')
parser.add_argument('--filename', dest='filename', type=str, default='london_walk(1).mp4', help='file name of the video file to be used in inference')

args = parser.parse_args()

def main(_):

  with tf.Graph().as_default():
    height = args.height
    width = args.width
    placeholders = {'im0':tf.placeholder(tf.float32,[None, None, None, 3], name='im0')}
    filename = args.filename

    with tf.variable_scope("model") as scope:
      model = pydnet(placeholders)

    init = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

    loader = tf.train.Saver()
    saver = tf.train.Saver()
    cam = cv2.VideoCapture('../test-videos/' + filename)

    with tf.Session() as sess:
        sess.run(init)
        loader.restore(sess, args.checkpoint_dir)
        count=0
        while True:
          for i in range(4):
            cam.grab()
          ret_val, img = cam.read() 
          img = cv2.resize(img, (width, height)).astype(np.float32) / 255.
          img = np.expand_dims(img, 0)
          start = time.time()
          disp = sess.run(model.results[args.resolution-1], feed_dict={placeholders['im0']: img})
          end = time.time()

          disp_color = applyColorMap(disp[0,:,:,0]*20, 'plasma')
          toShow = (np.concatenate((img[0], disp_color), 0)*255.).astype(np.uint8)
          toShow = cv2.resize(toShow, (width//2, height))
          #os.mkdir(filename[:-4])
          #np.savetxt(filename[:-4] + '/' + str(count)+'.out' ,disp)
          count += 1

          cv2.imshow('pydnet', toShow)
          k = cv2.waitKey(1)         
          if k == 1048603 or k == 27: 
            break  # esc to quit
          if k == 1048688:
            cv2.waitKey(0) # 'p' to pause

          #print("Time: " + str(end - start))
          dc = disp_color * 255
          L1 = dc[: ,0: 64]
          L2 = dc[: ,64: 128]
          L3 = dc[: ,128: 192]
          L4 = dc[: ,192: 256]
          R1 = dc[: ,256: 320]
          R2 = dc[: ,320: 384]
          R3 = dc[: ,384: 448]
          R4 = dc[: ,448: 512]
          print(count,L1.mean(),L2.mean(),L3.mean(),L4.mean(),R1.mean(),R2.mean(),R3.mean(),R4.mean()) 
          del img
          del disp
          del toShow
          
        cam.release()        

if __name__ == '__main__':
    tf.app.run()
