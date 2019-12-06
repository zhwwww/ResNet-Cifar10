import argparse
import mxnet as mx
import time
import os
import logging

def score(model, data_test, metrics, gpus, batch_size, image_shape, max_num_examples, rgb_std='1,1,1', rgb_mean='0,0,0' , data_nthreads=4, label_name='softmax_label'):

	data_shape = tuple([int(i) for i in image_shape.split(',')])
	rgb_mean = [float(i) for i in rgb_mean.split(',')]
	rgb_std = [float(i) for i in rgb_std.split(',')]
	data = mx.io.ImageRecordIter(
        path_imgrec        = data_test,
        label_width        = 1,
        preprocess_threads = data_nthreads,
        batch_size         = batch_size,
        data_shape         = data_shape,
        label_name         = label_name,
        rand_crop          = False,
        rand_mirror        = False,
		mean_r              = rgb_mean[0],
        mean_g              = rgb_mean[1],
        mean_b              = rgb_mean[2],
        std_r               = rgb_std[0],
        std_g               = rgb_std[1],
        std_b               = rgb_std[2]
		)

	if isinstance(model, tuple) or isinstance(model, list):
		assert len(model) == 3
		(sym, arg_params, aux_params) = model
	else:
		raise TypeError('model type [%s] is not supported' % str(type(model)))

	if gpus == '':
		devs = mx.cpu()
	else:
		devs = [mx.gpu(int(i)) for i in gpus.split(',')]

	mod = mx.mod.Module(symbol=sym, context=devs, label_names=[label_name,])
	mod.bind(for_training=False,
             data_shapes=data.provide_data,
             label_shapes=data.provide_label)
	mod.set_params(arg_params, aux_params)
	if not isinstance(metrics, list):
		metrics = [metrics,]
	tic = time.time()
	num = 0
	for batch in data:
		mod.forward(batch, is_train=False)
		for m in metrics:
			mod.update_metric(m, batch.label)
		num += batch_size
		if max_num_examples is not None and num > max_num_examples:
			break
	return (num / (time.time() - tic), )

def _load_model(load_epoch, model_prefix, rank=0):
	if rank > 0 and os.path.exists("%s-%d-symbol.json" % (model_prefix, rank)):
		model_prefix += "-%d" % (rank)
	sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, load_epoch)
	logging.info('Loaded model %s_%04d.params', model_prefix, load_epoch)
	return (sym, arg_params, aux_params)

if __name__ == '__main__':
    
	load_epoch = 200
	model_prefix = 'mx_resnet8'
	data_test = './data/train_val.rec'
	gpus = '0'
	batch_size = 128
	image_shape = '3,32,32'
	max_num_examples = None
	rgb_mean = '125.307,122.961,113.8575'
	rgb_std  = '51.5865,50.847,51.255'
	data_nthreads = 4 

	logging.basicConfig(level=logging.DEBUG)
	metrics = [mx.metric.create('acc')]
    
	model = _load_model(load_epoch,model_prefix)

	(speed,) = score(model=model, data_test=data_test, metrics = metrics, gpus=gpus, batch_size=batch_size, 
                     image_shape=image_shape, max_num_examples=max_num_examples, rgb_std=rgb_std, rgb_mean=rgb_mean,
                     data_nthreads=data_nthreads)
	logging.info('Finished with %f images per second', speed)
	for m in metrics:
		logging.info(m.get())
