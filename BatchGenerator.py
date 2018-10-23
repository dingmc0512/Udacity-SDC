import tensorflow as tf
import numpy as np
import PIL.Image as Image
import os
import cv2
from DataAugmenter import DataAugmenter
from settings import *
DA=DataAugmenter()


class BatchGenerator(object):
    def __init__(self, sequence, seq_len, batch_size, dir_path='../train_val/'):
        self.sequence = sequence
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.dir_path = dir_path
        chunk_size = 1 + (len(sequence) - 1) // batch_size
        self.indices = [(i*chunk_size) % len(sequence) for i in range(batch_size)]
        # |i+0 - - -|i+1 - - -|i+2 - - -|i+3 - - -| ... |


    #is_da:True if using data augmentation. 
    def decode_image(self, filenames, is_da=True):
        images = []
        for filename in filenames:
            img = Image.open(self.dir_path + filename)
            img = np.array(cv2.resize(np.array(img),(HEIGHT, WIDTH)))
            images.append(img)
        if is_da:
            images = DA.Apply(images)
        else:
            images = images
        return np.array(images).astype(np.float32)


    def next(self):
        while True:
            output = []
            for i in range(self.batch_size):
                idx = self.indices[i]
                left_pad = self.sequence[idx - LEFT_CONTEXT:idx]

                # less than left context, copy the most left one
                if len(left_pad) < LEFT_CONTEXT:
                    left_pad = [self.sequence[0]] * (LEFT_CONTEXT - len(left_pad)) + left_pad
                assert len(left_pad) == LEFT_CONTEXT
                
                # less than seq length, use the beginning ones
                leftover = len(self.sequence) - idx
                if leftover >= self.seq_len:
                    result = self.sequence[idx:idx + self.seq_len]
                else:
                    result = self.sequence[idx:] + self.sequence[:self.seq_len - leftover]
                assert len(result) == self.seq_len

                # update index of current batch
                self.indices[i] = (idx + self.seq_len) % len(self.sequence)

                # unzip lines to fields
                images, targets = zip(*result)
                images_left_pad, targets_left_pad = zip(*left_pad)

                path_seq = np.stack(images_left_pad + images)
                image_seq = self.decode_image(path_seq)
                target_seq = np.stack(tuple([targets_left_pad[-1]]) + targets)
                output.append((path_seq, image_seq, np.stack(target_seq)))

            output = list(zip(*output))
            output[0] = np.stack(output[0]) # batch_size x (LEFT_CONTEXT + seq_len)
            output[1] = np.stack(output[1]) # batch_size x (LEFT_CONTEXT + seq_len) x height x width x channel
            output[2] = np.stack(output[2]) # batch_size x (1 + seq_len) x OUTPUT_DIM
            return (output[0], output[1]), output[2]