import tensorflow as tf
from layers import shufflenet_unit, conv3d, max_pool_3d, avg_pool_3d, avg_pool_2d, dense, dropout, flatten
from settings import *

IS_TRAIN = None

class ShuffleNet:
	"""ShuffleNet is implemented here!"""
	def __init__(self, X, is_training):
		self.X = X
		self.is_training = is_training
		
		# A number stands for the num_groups
		# Output channels for conv1 layer
		self.output_channels = {'1': [144, 288, 576], '2': [200, 400, 800], '3': [240, 480, 960], '4': [272, 544, 1088],
								'8': [384, 768, 1536], 'conv1': 24}


	def __stage(self, x, stage=2, repeat=3):
		if 2 <= stage <= 4:
			stage_layer = shufflenet_unit('stage' + str(stage) + '_0', x=x, w=None,
										  num_groups=NUM_GROUPS,
										  group_conv_bottleneck=not (stage == 2),
										  num_filters=
										  self.output_channels[str(NUM_GROUPS)][
											  stage - 2],
										  stride=(2, 2, 2),
										  fusion='concat', l2_strength=L2_DECAY,
										  bias=0.0,
										  batchnorm_enabled=True,
										  is_training=self.is_training)
			for i in range(1, repeat + 1):
				stage_layer = shufflenet_unit('stage' + str(stage) + '_' + str(i),
											  x=stage_layer, w=None,
											  num_groups=NUM_GROUPS,
											  group_conv_bottleneck=True,
											  num_filters=self.output_channels[
												  str(NUM_GROUPS)][stage - 2],
											  stride=(1, 1, 1),
											  fusion='add',
											  l2_strength=L2_DECAY,
											  bias=0.0,
											  batchnorm_enabled=True,
											  is_training=self.is_training)
			return stage_layer
		else:
			raise ValueError("Stage should be from 2 -> 4")

	def inference(self):
		print('input_shape:', self.X.shape.as_list())
		conv1 = conv3d('conv1', x=self.X, w=None, num_filters=self.output_channels['conv1'], kernel_size=(3, 3, 3),
					   stride=(1, 2, 2), l2_strength=L2_DECAY, bias=0.0,
					   batchnorm_enabled=True, is_training=self.is_training,
					   activation=tf.nn.relu, padding='SAME')
		print('conv1_shape:', conv1.shape.as_list())
		max_pool = max_pool_3d(conv1, size=(3, 3, 3), stride=(2, 2, 2), name='max_pool')
		print('max3d_shape:', max_pool.shape.as_list())
		stage2 = self.__stage(max_pool, stage=2, repeat=3)
		print('stag2_shape:', stage2.shape.as_list())
		stage3 = self.__stage(stage2, stage=3, repeat=7)
		print('stag3_shape:', stage3.shape.as_list())
		stage4 = self.__stage(stage3, stage=4, repeat=3)
		print('stag4_shape:', stage4.shape.as_list())
		global_pool = avg_pool_3d(stage4, size=(1, 5, 5), stride=(1, 1, 1), name='global_pool', padding='VALID')
		print('avg3d_shape:', global_pool.shape.as_list())

		drop_out = dropout(global_pool, is_training=self.is_training, keep_prob=0.5)
		logits_unflattened = conv3d('fc', drop_out, w=None, num_filters=NUM_CLASS,
									kernel_size=(1, 1, 1),
									l2_strength=L2_DECAY,
									bias=0.0,
									is_training=self.is_training)
		print('convn_shape:', logits_unflattened.shape.as_list())
		logits = flatten(logits_unflattened)
		print('fc_re_shape:', logits.shape.as_list())
		return logits
