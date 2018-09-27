import tensorflow as tf
import numpy as np
import os
import model
from BatchGenerator import BatchGenerator
from settings import *
slim = tf.contrib.slim

        
'''       
def read_csv(filename):
    with open(filename, 'r') as f:
        lines = [ln.strip().split(",")[-7:-3] for ln in f.readlines()]
        lines = map(lambda x: (x[0], np.float32(x[1:])), lines) # imagefile, outputs
        return lines
'''

def read_csv(filename):
    with open(filename, 'r') as f:
        lines = []
        for ln in f.readlines():
            x = ln.strip().split(",")[-7:-3]
            if x[0].startswith('center'):
                lines.append((x[0],np.float32(x[1:])))
        return lines


def process_csv(filename, val=5):
    sum_f = np.float128([0.0] * OUTPUT_DIM)
    sum_sq_f = np.float128([0.0] * OUTPUT_DIM)
    lines = read_csv(filename)
    # leave val% for validation
    train_seq = []
    valid_seq = []
    cnt = 0
    for ln in lines:
        if cnt < SEQ_LEN * BATCH_SIZE * (100 - val): 
            train_seq.append(ln)
            sum_f += ln[1]
            sum_sq_f += ln[1] * ln[1]
        else:
            valid_seq.append(ln)
        cnt += 1
        cnt %= SEQ_LEN * BATCH_SIZE * 100
    mean = sum_f / len(train_seq)
    var = sum_sq_f / len(train_seq) - mean * mean
    std = np.sqrt(var)
    #print(len(train_seq), len(valid_seq))
    #print(mean, std) # we will need these statistics to normalize the outputs (and ground truth inputs)
    return (train_seq, valid_seq), (mean, std)


(train_seq, valid_seq), (mean, std) = process_csv(filename="../train_val/interpolated.csv", val=5) 
# concatenated interpolated.csv from rosbags 
test_seq = list(read_csv("../test/interpolated.csv")) 
# interpolated.csv for testset filled with dummy values


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
    learning_rate = tf.placeholder_with_default(input=LEARNING_RATE, shape=())
    keep_prob = tf.placeholder_with_default(input=1.0, shape=())
    aux_cost_weight = tf.placeholder_with_default(input=0.1, shape=())
    
    inputs = tf.placeholder(shape=(BATCH_SIZE,LEFT_CONTEXT+SEQ_LEN), dtype=tf.string) # pathes to png files from the central camera
    targets = tf.placeholder(shape=(BATCH_SIZE,SEQ_LEN,OUTPUT_DIM), dtype=tf.float32) # seq_len x batch_size x OUTPUT_DIM
    targets_normalized = (targets - mean) / std
    
    folder_path = tf.placeholder(shape=(), dtype=tf.string)
    input_images = tf.stack([tf.image.resize_images(tf.image.decode_png(tf.read_file('../' + folder_path + x)), (CROP_SIZE,CROP_SIZE))
                            for x in tf.unstack(tf.reshape(inputs, shape=[(LEFT_CONTEXT+SEQ_LEN) * BATCH_SIZE]))])

    out_gt, final_state_gt, out_regress, final_state_regress = model.inference(input_images, targets_normalized, keep_prob)

    mse_gt = tf.reduce_mean(tf.squared_difference(out_gt, targets_normalized))
    mse_autoregressive = tf.reduce_mean(tf.squared_difference(out_regress, targets_normalized))
    mse_autoregressive_steering = tf.reduce_mean(tf.squared_difference(out_regress[:, :, 0], targets_normalized[:, :, 0]))
    steering_predictions = (out_regress[:, :, 0] * std[0]) + mean[0]
    
    total_loss = mse_autoregressive_steering + aux_cost_weight * (1 * mse_gt + 4 * mse_autoregressive)
    
    optimizer = get_optimizer(total_loss, learning_rate)

    tf.summary.scalar("MAIN TRAIN METRIC: rmse_autoregressive_steering", tf.sqrt(mse_autoregressive_steering))
    tf.summary.scalar("rmse_gt", tf.sqrt(mse_gt))
    tf.summary.scalar("rmse_autoregressive", tf.sqrt(mse_autoregressive))
    
    summaries = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('v3/train_summary', graph=graph)
    valid_writer = tf.summary.FileWriter('v3/valid_summary', graph=graph)
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)



global_train_step = 0
global_valid_step = 0
KEEP_PROB_TRAIN = 0.5

def do_epoch(session, sequences, mode):
    global global_train_step, global_valid_step
    test_predictions = {}
    valid_predictions = {}
    batch_generator = BatchGenerator(sequence=sequences, seq_len=SEQ_LEN, batch_size=BATCH_SIZE)
    total_num_steps = 1 + (batch_generator.indices[1] - 1) // SEQ_LEN
    final_state_gt_cur, final_state_regress_cur = None, None
    acc_loss = np.float128(0.0)
    for step in range(total_num_steps):
        feed_inputs, feed_targets = batch_generator.next()
        feed_dict = {inputs : feed_inputs, targets : feed_targets}
        
        if final_state_regress_cur is not None:           
            feed_dict.update({final_state_regress : final_state_regress_cur})
        if final_state_gt_cur is not None:          
            feed_dict.update({final_state_gt : final_state_gt_cur})
        
        if mode == "train":
            feed_dict.update({folder_path : 'train_val/'})
            feed_dict.update({keep_prob : KEEP_PROB_TRAIN})
            summary, _, loss, final_state_gt_cur, final_state_regress_cur = \
                session.run([summaries, optimizer, mse_autoregressive_steering, final_state_gt, final_state_regress],
                           feed_dict = feed_dict)
            train_writer.add_summary(summary, global_train_step)
            global_train_step += 1
        elif mode == "valid":
            feed_dict.update({folder_path : 'train_val/'})
            model_predictions, summary, loss, final_state_regress_cur = \
                session.run([steering_predictions, summaries, mse_autoregressive_steering, final_state_regress],
                           feed_dict = feed_dict)
            valid_writer.add_summary(summary, global_valid_step)
            global_valid_step += 1  
            feed_inputs = feed_inputs[:, LEFT_CONTEXT:].flatten()
            steering_targets = feed_targets[:, :, 0].flatten()
            model_predictions = model_predictions.flatten()
            stats = np.stack([steering_targets, model_predictions, (steering_targets - model_predictions)**2])
            for i, img in enumerate(feed_inputs):
                valid_predictions[img] = stats[:, i]
        elif mode == "test":
            feed_dict.update({folder_path : 'test/'})
            model_predictions, final_state_regress_cur = \
                session.run([steering_predictions, final_state_regress],
                           feed_dict = feed_dict)           
            feed_inputs = feed_inputs[:, LEFT_CONTEXT:].flatten()
            model_predictions = model_predictions.flatten()
            for i, img in enumerate(feed_inputs):
                test_predictions[img] = model_predictions[i]
        if mode != "test":
            acc_loss += loss
            print('\r', step + 1, "/", total_num_steps, np.sqrt(acc_loss / (step+1)))
    print()
    return (np.sqrt(acc_loss / total_num_steps), valid_predictions) if mode != "test" else (None, test_predictions)
    


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
checkpoint_dir = os.getcwd() + "/v3"

NUM_EPOCHS=20
best_validation_score = None

with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as session:
    session.run(tf.global_variables_initializer())
    print('Initialized')
    ckpt = tf.train.latest_checkpoint(checkpoint_dir)
    if ckpt:
        print("Restoring from", ckpt)
        saver.restore(sess=session, save_path=ckpt)
    for epoch in range(NUM_EPOCHS):
        print("Starting epoch %d" % epoch)
        print("Validation:")
        valid_score, valid_predictions = do_epoch(session=session, sequences=valid_seq, mode="valid")
        if best_validation_score is None: 
            best_validation_score = valid_score
        if valid_score < best_validation_score:
            saver.save(session, 'v3/checkpoint-sdc-ch2')
            best_validation_score = valid_score
            print('\r', "SAVED at epoch %d" % epoch)
            with open("v3/valid-predictions-epoch%d" % epoch, "w") as out:
                result = np.float128(0.0)
                for img, stats in valid_predictions.items():
                    print(img, stats, file=out)
                    result += stats[-1]
            with open("v3/best_valid_score", "w") as out:
            	print("Validation unnormalized RMSE:", np.sqrt(result / len(valid_predictions)), file=out)
            with open("v3/test-predictions-epoch%d" % epoch, "w") as out:
                _, test_predictions = do_epoch(session=session, sequences=test_seq, mode="test")
                print("frame_id,steering_angle", file=out)
                for img, pred in test_predictions.items():
                    img = img.replace("challenge_2/Test-final/center/", "")
                    print("%s,%f" % (img, pred), file=out)
        if epoch != NUM_EPOCHS - 1:
            print("Training")
            do_epoch(session=session, sequences=train_seq, mode="train")
