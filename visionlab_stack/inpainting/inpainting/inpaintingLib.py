
from warnings import simplefilter 
simplefilter(action='ignore', category=FutureWarning)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
#os.environ["CUDA_VISIBLE_DEVICES"]="-1" # -1 null

#https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
import logging
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import tensorflow as tf
import numpy as np
import cv2

# this repalce ng.Config inside  env/lib/python3.6/site-packages/neuralgym/utils/config.py
from visionlab_stack.inpainting.inpainting.utils.ngconfig import Config  # we remove the print
from visionlab_stack.inpainting.inpainting.utils.inpaint_model import InpaintCAModel

class inpaintingClass:
    
    def __init__(self,
            inpaint_yml = "./data/models/inpainting/inpainting/inpaint.yml",
            inpaint_checkpoint = "./data/models/inpainting/inpainting/release_places2_256_deepfill_v2",
            inpaint_size = [256,256]
        ):
        self.inpaint_checkpoint = inpaint_checkpoint
        self.FLAGS = Config(inpaint_yml)

        self.size = inpaint_size
        self.input_image_ph, self.output, self.sess = self.deepfill_model(self.inpaint_checkpoint)

    def deepfill_model(self,checkpoint_dir):
        # Line tor un with gpu
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        sess = tf.Session(config=sess_config)

        model = InpaintCAModel()

        input_image_ph = tf.placeholder(tf.float32, shape=(1, self.size[0], self.size[1]*2, 3))
        #output = model.build_server_graph(input_image_ph, dynamic=True)
        output = model.build_server_graph(self.FLAGS, input_image_ph,reuse=tf.AUTO_REUSE)
        output = (output + 1.) * 127.5
        output = tf.reverse(output, [-1])
        output = tf.saturate_cast(output, tf.uint8)
        
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        for var in vars_list:
            vname = var.name
            from_name = vname
            var_value = tf.contrib.framework.load_variable(checkpoint_dir, from_name)
            assign_ops.append(tf.assign(var, var_value))
        sess.run(assign_ops)
        #print('Model loaded.')
        return input_image_ph, output, sess
        
    def doInpainting(self, image, mask):        
        #inp_image = cv.resize(inp_image, (608,608))
        #image = cv.resize(image, (608,608))
        #mask = cv.resize(mask, (608,608))

        res = np.concatenate([image,mask])        
        # ng.get_gpus(1)
        #image = cv2.imread(args.image)
        #mask = cv2.imread(args.mask)
        # mask = cv2.resize(mask, (0,0), fx=0.5, fy=0.5)
      
        assert image.shape == mask.shape

        h, w, _ = image.shape
        grid = 8
        image = image[:h//grid*grid, :w//grid*grid, :]
        mask = mask[:h//grid*grid, :w//grid*grid, :]
        #h, w, _ = image.shape


        # Added
        image = cv2.resize(image,(self.size[0],self.size[1]))
        mask = cv2.resize(mask,(self.size[0],self.size[1]))

        image = np.expand_dims(image, 0)
        mask = np.expand_dims(mask, 0)
        input_image = np.concatenate([image, mask], axis=2)
        #print('Shape of image: {}'.format(image.shape))

        '''
        input_image_ph = tf.placeholder(tf.float32, shape=(1, image.shape[1], image.shape[2]*2, 3))
        #output = model.build_server_graph(input_image_ph, dynamic=True)
        output = self.model.build_server_graph(self.FLAGS, input_image_ph) #,reuse=tf.AUTO_REUSE)
        output = (output + 1.) * 127.5
        output = tf.reverse(output, [-1])
        output = tf.saturate_cast(output, tf.uint8)
        '''
        result = self.sess.run(self.output, feed_dict={self.input_image_ph: input_image})


        #cv2.imwrite("out.png", result[0][:, :, ::-1])
        res = result[0][:, :, ::-1]
        res = cv2.resize(res,(w,h))
        #cv2.imshow("dsi",res)
        #cv2.waitKey(0)

        #res = np.concatenate([inp_image,image[0],res])

        '''
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        with tf.compat.v1.Session(config=sess_config) as sess:
            input_image = tf.constant(input_image, dtype=tf.float32)
            output = self.model.build_server_graph(self.FLAGS, input_image,reuse=tf.AUTO_REUSE)
            output = (output + 1.) * 127.5
            output = tf.reverse(output, [-1])
            output = tf.saturate_cast(output, tf.uint8)
            
            # load pretrained model
            vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            assign_ops = []
            for var in vars_list:
                vname = var.name
                from_name = vname
                var_value = tf.contrib.framework.load_variable(self.inpaint_checkpoint, from_name)
                assign_ops.append(tf.compat.v1.assign (var, var_value))
            sess.run(assign_ops)

            result = sess.run(output)
                
            #cv2.imwrite("out.png", result[0][:, :, ::-1])
            res = result[0][:, :, ::-1]
            #res = np.concatenate([inp_image,image[0],res])
        '''
        return res
