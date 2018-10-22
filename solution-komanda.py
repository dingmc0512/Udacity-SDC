import tensorflow as tf
import numpy as np
import os
import time
import model
import shutil
from BatchGenerator import BatchGenerator
from ProcessDataSet import *
from settings import *
slim = tf.contrib.slim

import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--train_txt', default='../train_val/interpolated.csv', type=str)
parser.add_argument('--test_txt', default='../test/interpolated.csv', type=str)
parser.add_argument('--model', default=None, type=str)
args=parser.parse_args()


(train_seq, valid_seq), (mean, std) = process_csv(filename=args.train_txt, val=5) 
# concatenated interpolated.csv from rosbags 
test_seq = list(read_csv(args.test_txt)) 
# interpolated.csv for testset filled with dummy values


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, v in grad_and_vars:
            if g is not None:
                expanded_g = tf.expand_dims(g, 0)
                grads.append(expanded_g)
        if grads:
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)
        else:
            grad = None
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def compute_loss(name_scope,out_gt,out_regress,targets_normalized,aux_cost_weight):
    mse_gt = tf.reduce_mean(tf.squared_difference(out_gt, targets_normalized))
    mse_autoregressive = tf.reduce_mean(tf.squared_difference(out_regress, targets_normalized))
    mse_autoregressive_steering = tf.reduce_mean(tf.squared_difference(out_regress[:, :, 0], targets_normalized[:, :, 0]))
    
    tf.summary.scalar(name_scope+"_rmse_infer_steering", tf.sqrt(mse_autoregressive_steering))
    tf.summary.scalar(name_scope+"_rmse_gt", tf.sqrt(mse_gt))
    tf.summary.scalar(name_scope+"_rmse_infer", tf.sqrt(mse_autoregressive))

    # T3D
    '''
    weight_loss = tf.losses.get_regularization_loss()
    tf.summary.scalar(name_scope+"_weight_loss", weight_loss)
    '''

    # P3D
    '''
    weight_loss=tf.get_collection('weightdecay_losses')
    tf.summary.scalar(name_scope+'_weight_loss',tf.reduce_mean(weight_loss))
    '''

    #tf.add_to_collection('weight_loss', tf.reduce_mean(weight_loss))
    total_loss  = mse_autoregressive_steering + aux_cost_weight * (1 * mse_gt + 4 * mse_autoregressive)
    total_loss  = total_loss #+ weight_loss

    tf.summary.scalar(name_scope+'_total_loss',tf.sqrt(total_loss))
    return total_loss


def placeholder_inputs(batch_size):
    # image from the central camera
    inputs_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                            LEFT_CONTEXT+SEQ_LEN,
                                                            HEIGHT,
                                                            WIDTH,
                                                            RGB_CHANNEL))
    # seq_len x batch_size x OUTPUT_DIM 
    labels_placeholder = tf.placeholder(tf.float32, shape=(batch_size,SEQ_LEN,OUTPUT_DIM))
    return inputs_placeholder, labels_placeholder


def get_optimizer(loss, lrate):
    optimizer = tf.train.AdamOptimizer(learning_rate=lrate)

    gradvars = optimizer.compute_gradients(loss)
    gradients, v = zip(*gradvars)
    #print([x.name for x in v])
    gradients, _ = tf.clip_by_global_norm(gradients, 15.0)
    return optimizer.apply_gradients(zip(gradients, v))


graph = tf.Graph()
with graph.as_default():
    # inputs  
    global_step=tf.get_variable('global_step',[],initializer=tf.constant_initializer(0),trainable=False)
    learning_rate = tf.placeholder_with_default(input=LEARNING_RATE, shape=())
    keep_prob = tf.placeholder_with_default(input=1.0, shape=())
    aux_cost_weight = tf.placeholder_with_default(input=0.1, shape=())

    folder_path = tf.placeholder(shape=(), dtype=tf.string)
    inputs_placeholder, labels_placeholder = placeholder_inputs(BATCH_SIZE * GPU_NUM)

    input_images = inputs_placeholder
    #input_images = tf.stack([tf.image.resize_images(tf.image.decode_png(tf.read_file('../' + folder_path + x)), (HEIGHT, WIDTH))
    #                         for x in tf.unstack(tf.reshape(inputs_placeholder, shape=[(LEFT_CONTEXT+SEQ_LEN) * BATCH_SIZE * GPU_NUM]))])
    targets_normalized = (labels_placeholder - mean) / std

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    tower_grads = []
    predictions = []
    state_gt = []
    state_regress = []

    for gpu_index in range(0, GPU_NUM):
        print("gpu_index:", gpu_index)
        with tf.variable_scope('SDC', reuse=bool(gpu_index != 0)):
            with tf.device('/gpu:%d' % gpu_index):
                out_gt, final_state_gt, out_regress, final_state_regress=model.inference(
                        input_images[gpu_index * BATCH_SIZE:(gpu_index + 1) * BATCH_SIZE,:,:,:,:],
                        targets_normalized[gpu_index * BATCH_SIZE:(gpu_index + 1) * BATCH_SIZE],
                        keep_prob)
                loss_name_scope = ('gpud_%d_loss' % gpu_index)
                loss = compute_loss(
                                loss_name_scope,
                                out_gt,
                                out_regress,
                                targets_normalized[gpu_index * BATCH_SIZE:(gpu_index + 1) * BATCH_SIZE],
                                aux_cost_weight)
                        
                # use last tower statistics to update the moving mean/variance 
                batchnorm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

                # Reuse variables for the next tower.
                # tf.get_variable_scope().reuse_variables()

                varlist=tf.trainable_variables()       
                grads = optimizer.compute_gradients(loss, varlist)

                tower_grads.append(grads)
                predictions.append(out_regress)
                state_gt.append(final_state_gt)
                state_regress.append(final_state_regress)

    predictions = tf.concat(predictions,0)      
    grads = average_gradients(tower_grads)
    apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)
    mse_autoregressive_steering = tf.reduce_mean(tf.squared_difference(predictions[:, :, 0], targets_normalized[:, :, 0]))
    steering_predictions = (predictions[:, :, 0] * std[0]) + mean[0]
    
    tf.summary.scalar('mse_steering',tf.sqrt(mse_autoregressive_steering))

    with tf.control_dependencies(batchnorm_updates):
        optim_op_group=tf.group(apply_gradient_op)
    
    summaries = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./visual_logs/'+args.model+'/train', graph=graph)
    valid_writer = tf.summary.FileWriter('./visual_logs/'+args.model+'/test', graph=graph)
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)


global_train_step = 0
global_valid_step = 0
loss_cur = None

def do_epoch(session, sequences, mode):
    global global_train_step, global_valid_step
    global loss_cur
    test_predictions = {}
    valid_predictions = {}
    final_state_gt_cur, final_state_regress_cur = None, None
    acc_loss = np.float128(0.0)

    '''
    if not os.path.exists("./bad_case"):
        os.mkdir("./bad_case")
    '''

    duration=0
    batch_generator = BatchGenerator(sequence=sequences, seq_len=SEQ_LEN, batch_size=BATCH_SIZE * GPU_NUM)
    total_num_steps = 1 + (batch_generator.indices[1] - 1) // SEQ_LEN
    for step in range(total_num_steps):
        feed_inputs, feed_targets = batch_generator.next()
        image_path = feed_inputs[0]
        image_data = feed_inputs[1]
        feed_dict = {inputs_placeholder : image_data, labels_placeholder : feed_targets}
        
        #print('weight_loss: ' + str(session.run(tf.get_collection('weight_loss')[0])))

        if final_state_regress_cur is not None:           
            feed_dict.update({final_state_regress : final_state_regress_cur})
        if final_state_gt_cur is not None:          
            feed_dict.update({final_state_gt : final_state_gt_cur})
        
        if mode == "train":
            start_time=time.time()

            feed_dict.update({keep_prob : KEEP_PROB_TRAIN})
            summary, _, loss, final_state_gt_cur, final_state_regress_cur = \
                session.run([summaries, optim_op_group, mse_autoregressive_steering, final_state_gt, final_state_regress],
                           feed_dict = feed_dict)
            train_writer.add_summary(summary, global_train_step)
            global_train_step += 1

            duration+=time.time()-start_time
            print('Step %d: %.2f sec -->loss : %.4f' % (step+1, duration, np.mean(loss)))
            duration=0  

        elif mode == "valid":
            start_time=time.time()

            model_predictions, summary, loss, final_state_regress_cur = \
                session.run([steering_predictions, summaries, mse_autoregressive_steering, final_state_regress],
                           feed_dict = feed_dict)
            valid_writer.add_summary(summary, global_valid_step)
            global_valid_step += 1  
            image_path = image_path[:, LEFT_CONTEXT:].flatten()
            steering_targets = feed_targets[:, :, 0].flatten()
            model_predictions = model_predictions.flatten()
            stats = np.stack([steering_targets, model_predictions, (steering_targets - model_predictions)**2])
            for i, img in enumerate(image_path):
                valid_predictions[img] = stats[:, i]
			
            duration+=time.time()-start_time
            print('Step %d: %.2f sec -->loss : %.4f ==== rsme: %.4f' % (step+1, duration, np.mean(loss), np.sqrt(np.mean(stats[-1]))))
            duration=0 

            '''
            if loss_cur is None: 
                loss_cur = loss
            if loss_cur * 1.4 < loss:
            	for img in image_path:
            		shutil.copyfile(os.path.join('../train_val',img), os.path.join('./bad_case',img.split('/')[1] + '_' + str(global_valid_step)))
            loss_cur = loss
            '''

        elif mode == "test":
            feed_dict.update({folder_path : 'test/'})
            model_predictions, final_state_regress_cur = \
                session.run([steering_predictions, final_state_regress],
                           feed_dict = feed_dict)           
            image_path = image_path[:, LEFT_CONTEXT:].flatten()
            model_predictions = model_predictions.flatten()
            for i, img in enumerate(image_path):
                test_predictions[img] = model_predictions[i]

        if mode != "test":
            acc_loss += loss
            #print('\r', step + 1, "/", total_num_steps, np.sqrt(acc_loss / (step+1)))
    print()
    return (np.sqrt(acc_loss / total_num_steps), valid_predictions) if mode != "test" else (None, test_predictions)
    

def run():
    #checkpoint_dir = os.getcwd() + "/visual_logs"
    MODEL_PATH='./visual_logs/'+args.model
    best_validation_score = None

    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True)) as session:

        # initial network varibles
        session.run(tf.global_variables_initializer())
        print('Initialized')

        # restore model parameters
        ckpt = tf.train.latest_checkpoint(MODEL_PATH)
        if ckpt:
            print("Restoring from", ckpt)
            saver.restore(sess=session, save_path=ckpt)

        # train model
        for epoch in range(NUM_EPOCHS):
            print("Starting epoch %d" % epoch)
            print("Validation:")
            valid_score, valid_predictions = do_epoch(session=session, sequences=valid_seq, mode="valid")

            if best_validation_score is None: 
                best_validation_score = valid_score
            if valid_score < best_validation_score:
                saver.save(session, MODEL_PATH+'/checkpoint-sdc')
                best_validation_score = valid_score
                print('\r', "SAVED at epoch %d" % epoch)

                # write validation result to file
                '''
                with open(MODEL_PATH+"/valid-predictions-epoch%d" % epoch, "w") as out:
                    result = np.float128(0.0)
                    for img, stats in valid_predictions.items():
                        print(img, stats, file=out)
                        result += stats[-1]
                '''
                result = np.float128(0.0)
                for img, stats in valid_predictions.items():
                    result += stats[-1]
                with open(MODEL_PATH+"/best_valid_score", "w") as out:
                	print("Validation unnormalized RMSE:", np.sqrt(result / len(valid_predictions)), file=out)

                # test process
                '''
                with open(MODEL_PATH+"/test-predictions-epoch%d" % epoch, "w") as out:
                    _, test_predictions = do_epoch(session=session, sequences=test_seq, mode="test")
                    print("frame_id,steering_angle", file=out)
                    for img, pred in test_predictions.items():
                        img = img.replace("challenge_2/Test-final/center/", "")
                        print("%s,%f" % (img, pred), file=out)
                '''

            if epoch != NUM_EPOCHS - 1:
                print("Training")
                do_epoch(session=session, sequences=train_seq, mode="train")


if __name__=='__main__':
    print('Preparing for dataset,this may take several seconds.')
    run()  