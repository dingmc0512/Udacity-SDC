# define some constants

# RNNs are typically trained using (truncated) backprop through time. SEQ_LEN here is the length of BPTT. 
# Batch size specifies the number of sequence fragments used in a sigle optimization step.
# (Actually we can use variable SEQ_LEN and BATCH_SIZE, they are set to constants only for simplicity).
# LEFT_CONTEXT is the number of extra frames from the past that we append to the left of our input sequence.
# We need to do it because 3D convolution with "VALID" padding "eats" frames from the left, decreasing the sequence length.
# One should be careful here to maintain the model's causality.

# P3D Settings
BLOCK_EXPANSION = 4

# T3D Settings
BN_SIZE=4  #Expansion rate for Bottleneck structure.
GROWTH_RATE=32 #Fixed number of out_channels for some layers.
START_CHANNEL=64  #number of channels before entering first block structure.

# Common Settings
GPU_NUM = 2
VISION_FEATURE_SIZE = 128
CROP_SIZE=160
SEQ_LEN = 5 
BATCH_SIZE = 8 
LEFT_CONTEXT = 5
LEARNING_RATE = 1e-5
KEEP_PROB_TRAIN = 0.5
#IS_DA=False #whether or not to use data augmentation

# These are the input image parameters.
HEIGHT = 480
WIDTH = 640
RGB_CHANNEL = 3

# The parameters of the LSTM that keeps the model state.
RNN_SIZE = 32
RNN_PROJ = 32

# Our training data follows the "interpolated.csv" format from Ross Wightman's scripts.
CSV_HEADER = "index,timestamp,width,height,frame_id,filename,angle,torque,speed,lat,long,alt".split(",")
OUTPUTS = CSV_HEADER[-6:-3] # angle,torque,speed
OUTPUT_DIM = len(OUTPUTS) # predict all features: steering angle, torque and vehicle speed
