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
from pygame import mixer
import os
import argparse
import time
import datetime
from utils import *
from pydnet import *
import png
import math
import time

def mapFunction(value, min_orig, max_orig, min_new, max_new):
  result = (min_new + ((max_new - min_new)/(max_orig - min_orig)) * (value - min_orig))
  if(result > 6300):
    result = 6300
  if(result < 200):
    result = 200
  upperhundred =  int(math.ceil(result/100.0)) * 100
  lowerhundred = upperhundred - 100
  
  if(abs(result - upperhundred) > abs(result - lowerhundred)):
    return lowerhundred
  else:
    return upperhundred

def mapFunction2(value, min_orig, max_orig, min_new, max_new):
  result = (min_new + ((max_new - min_new)/(max_orig - min_orig)) * (value - min_orig))
  upperhundred =  int(math.ceil(result/100.0)) * 100
  lowerhundred = upperhundred - 100
  return upperhundred

def play_audio(L1,L2,L3,L4,R1,R2,R3,R4):
  sec1 = "%s_100_L.wav" % L1
  sec2 = "%s_080_L.wav" % L2
  sec3 = "%s_050_L.wav" % L3
  sec4 = "%s_030_L.wav" % L4
  sec5 = "%s_030_R.wav" % R1
  sec6 = "%s_050_R.wav" % R2
  sec7 = "%s_080_R.wav" % R3
  sec8 = "%s_100_R.wav" % R4
  LOCATION = "../../audio-files/"
  SHORT_WAIT = 0.3
  LONG_WAIT = 0.75
  MIN_VAL = 1000
  
  mixer.init()
  if(L1 > MIN_VAL):
    mixer.music.load(LOCATION + sec1)
    mixer.music.play()
    time.sleep(SHORT_WAIT)

  if(L2 > MIN_VAL):
    mixer.music.load(LOCATION + sec2)
    mixer.music.play()
    time.sleep(SHORT_WAIT)

  if(L3 > MIN_VAL):
    mixer.music.load(LOCATION + sec3)
    mixer.music.play()
    time.sleep(SHORT_WAIT)

  if(L4 > MIN_VAL):
    mixer.music.load(LOCATION + sec4)
    mixer.music.play()
    time.sleep(SHORT_WAIT)

  if(R1 > MIN_VAL):
    mixer.music.load(LOCATION + sec5)
    mixer.music.play()
    time.sleep(SHORT_WAIT)

  if(R2 > MIN_VAL):
    mixer.music.load(LOCATION + sec6)
    mixer.music.play()
    time.sleep(SHORT_WAIT)

  if(R3 > MIN_VAL):
    mixer.music.load(LOCATION + sec7)
    mixer.music.play()
    time.sleep(SHORT_WAIT)

  if(R4 > MIN_VAL):
    mixer.music.load(LOCATION + sec8)
    mixer.music.play()
    time.sleep(LONG_WAIT)
  
# forces tensorflow to run on CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

parser = argparse.ArgumentParser(description='Argument parser')

""" Arguments related to network architecture"""
parser.add_argument('--width', dest='width', type=int, default=512, help='width of input images')
parser.add_argument('--height', dest='height', type=int, default=256, help='height of input images')
parser.add_argument('--resolution', dest='resolution', type=int, default=3, help='resolution [1:H, 2:Q, 3:E]')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', type=str, default='checkpoint/IROS18/pydnet', help='checkpoint directory')
parser.add_argument('--filename', dest='filename', type=str, default='roxas1.mp4', help='file name of the video file to be used in inference')

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
    #cam = cv2.VideoCapture('../test-videos/' + filename)
    cam = cv2.VideoCapture(0)

    with tf.Session() as sess:
        sess.run(init)
        loader.restore(sess, args.checkpoint_dir)
        count=0
        while True:
          for i in range(4):
            cam.grab()
          ret_val, img = cam.read() 
          # if count % 10 != 0:
          #   count += 1
          #   continue
          # uncomment next line to generate images
          # cv2.imwrite(str(count) + '-' + filename[:-4] + '.jpg', img)
          
          img = cv2.resize(img, (width, height)).astype(np.float32) / 255.
          img = np.expand_dims(img, 0)
          
          start = time.time()
          disp = sess.run(model.results[args.resolution-1], feed_dict={placeholders['im0']: img})
          end = time.time()

          disp_color = applyColorMap(disp[0,:,:,0]*20, 'plasma')
          toShow = (np.concatenate((img[0], disp_color), 0)*255.).astype(np.uint8)
          toShow = cv2.resize(toShow, (width//2, height))
          #os.mkdir(filename[:-4])

          # uncomment for saving data
          # np.savetxt(str(count) + '-' + filename[:-4] + '-savetxt-' + '.txt',disp, delimiter=', ', fmt='%s', header="load file using => np.loadtxt('filename', dtype=int)")
          # np.save(str(count) + '-' + filename[:-4] + '-save-' + '.npy',disp)

          count += 1

          cv2.imshow('pydnet', toShow)
          k = cv2.waitKey(1)         
          if k == 1048603 or k == 27: 
            break  # esc to quit
          if k == 1048688:
            cv2.waitKey(0) # 'p' to pause

          #print("Time: " + str(end - start))
          dc = disp_color * 255
          min_orig = 50.0
          max_orig = 150.0
          min_new = 200.0
          max_new = 6300.0
          
          unprocessed_L1 = mapFunction2(dc[: ,0: 64].mean(), min_orig, max_orig, min_new, max_new)
          unprocessed_L2 = mapFunction2(dc[: ,64: 128].mean(), min_orig, max_orig, min_new, max_new)
          unprocessed_L3 = mapFunction2(dc[: ,128: 192].mean(), min_orig, max_orig, min_new, max_new)
          unprocessed_L4 = mapFunction2(dc[: ,192: 256].mean(), min_orig, max_orig, min_new, max_new)
          unprocessed_R1 = mapFunction2(dc[: ,256: 320].mean(), min_orig, max_orig, min_new, max_new)
          unprocessed_R2 = mapFunction2(dc[: ,320: 384].mean(), min_orig, max_orig, min_new, max_new)
          unprocessed_R3 = mapFunction2(dc[: ,384: 448].mean(), min_orig, max_orig, min_new, max_new)
          unprocessed_R4 = mapFunction2(dc[: ,448: 512].mean(), min_orig, max_orig, min_new, max_new)

          L1 = mapFunction(dc[: ,0: 64].mean(), min_orig, max_orig, min_new, max_new)
          L2 = mapFunction(dc[: ,64: 128].mean(), min_orig, max_orig, min_new, max_new)
          L3 = mapFunction(dc[: ,128: 192].mean(), min_orig, max_orig, min_new, max_new)
          L4 = mapFunction(dc[: ,192: 256].mean(), min_orig, max_orig, min_new, max_new)
          R1 = mapFunction(dc[: ,256: 320].mean(), min_orig, max_orig, min_new, max_new)
          R2 = mapFunction(dc[: ,320: 384].mean(), min_orig, max_orig, min_new, max_new)
          R3 = mapFunction(dc[: ,384: 448].mean(), min_orig, max_orig, min_new, max_new)
          R4 = mapFunction(dc[: ,448: 512].mean(), min_orig, max_orig, min_new, max_new)

          print(count,L1,L2,L3,L4,R1,R2,R3,R4)
          print(count,unprocessed_L1,unprocessed_L2,unprocessed_L3,unprocessed_L4,unprocessed_R1,unprocessed_R2,unprocessed_R3,unprocessed_R4)
          print(end-start)
          print()

          play_audio(L1,L2,L3,L4,R1,R2,R3,R4)

          # arr_out = np.array([count,L1.mean(),L2.mean(),L3.mean(),L4.mean(),R1.mean(),R2.mean(),R3.mean(),R4.mean()])
#         #uncomment for saving data
#         np.savetxt(str(count) + '-' + 'readings' + '-savetxt-' + '.txt',arr_out, delimiter=', ', fmt='%s', header="load file using => np.loadtxt('filename', dtype=int)")
#         np.save(str(count) + '-' + 'readings' + '-save-' + '.npy',arr_out)

          del img
          del disp
          del toShow
          
        cam.release()        

if __name__ == '__main__':
    tf.app.run()

