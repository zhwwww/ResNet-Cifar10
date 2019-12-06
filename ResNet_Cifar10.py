import mxnet as mx
import argparse
import logging
import math
import data
import resnet_8layers

def _get_lr_scheduler(args, kv):
    epoch_size = get_epoch_size(args, kv)
    begin_epoch = args.load_epoch if args.load_epoch else 0
    step_epochs = [int(l) for l in args.lr_step_epochs.split(',')]
    lr = args.lr
    for s in step_epochs:
        if begin_epoch >= s:
            lr *= args.lr_factor
    if lr != args.lr:
        logging.info('Adjust learning rate to %e for epoch %d', lr, begin_epoch)
    steps = [epoch_size * (x - begin_epoch)
             for x in step_epochs if x - begin_epoch > 0]
    if steps:
        return (lr, mx.lr_scheduler.MultiFactorScheduler(step=steps, factor=args.lr_factor,
                                                         base_lr=args.lr))
    else:
        return (lr, None)

def _save_model(args, rank=0):
    if args.model_prefix is None:
        return None
    return mx.callback.do_checkpoint(args.model_prefix if rank == 0 else "%s-%d" % (
        args.model_prefix, rank), period=args.save_period)

def get_epoch_size(args, kv):
    return math.ceil(int(args.num_examples / kv.num_workers) / args.batch_size)

def _load_model(args, rank=0):
    if 'load_epoch' not in args or args.load_epoch is None:
        return (None, None, None)
    assert args.model_prefix is not None
    model_prefix = args.model_prefix
    if rank > 0 and os.path.exists("%s-%d-symbol.json" % (model_prefix, rank)):
        model_prefix += "-%d" % (rank)
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, args.load_epoch)
    logging.info('Loaded model %s_%04d.params', model_prefix, args.load_epoch)
    return (sym, arg_params, aux_params)


def add_data_aug_arg(parser):
    aug = parser.add_argument_group('DataAugmentation','data augmentation')
    aug.add_argument('--random-crop', type=bool, default=True, help='if or not randomly crop the image')
    aug.add_argument('--min-crop-size', type=int, default=32, help='Crop both width and height into a random size')
    aug.add_argument('--max-crop-size', type=int, default=32, help='Crop both width and height into a random size')	
    #Resize into [width*s, height*s] with s randomly chosen from 
    #[min_random_scale, max_random_scale]. Ignored if random_resized_crop is True.		 
    aug.add_argument('--max-random-scale', type=float, default=1, help='max ratio to scale')
    aug.add_argument('--min-random-scale', type=float, default=1, help='min ratio to scale, should >= img_size/input_shape. '
                          'otherwise use --pad-size')				 					  
    aug.add_argument('--random-resized-crop', type=bool, default=False,
                     help='whether to use random resized crop')
    aug.add_argument('--max-random-area', type=float, default=1,
                     help='max area to crop in random resized crop, whose range is [0, 1]')
    aug.add_argument('--min-random-area', type=float, default=1,
                     help='min area to crop in random resized crop, whose range is [0, 1]')	 
    aug.add_argument('--random-mirror', type=bool, default=True,
                     help='if or not randomly flip horizontally')				 
    aug.add_argument('--min-random-aspect-ratio', type=float, default=None,
                     help='min value of aspect ratio, whose value is either None or a positive value.')
    aug.add_argument('--max-random-aspect-ratio', type=float, default=0,
                     help='max value of aspect ratio. If min_random_aspect_ratio is None, '
                          'the aspect ratio range is [1-max_random_aspect_ratio, '
                          '1+max_random_aspect_ratio], otherwise it is '
                          '[min_random_aspect_ratio, max_random_aspect_ratio].')    
    aug.add_argument('--max-random-rotate-angle', type=int, default=0,
                     help='max angle to rotate, whose range is [0, 360]')
    aug.add_argument('--max-random-shear-ratio', type=float, default=0,
                     help='max ratio to shear, whose range is [0, 1]')
    aug.add_argument('--brightness', type=float, default=0,
                     help='brightness jittering, whose range is [0, 1]')
    aug.add_argument('--contrast', type=float, default=0,
                     help='contrast jittering, whose range is [0, 1]')
    aug.add_argument('--saturation', type=float, default=0,
                     help='saturation jittering, whose range is [0, 1]')
    aug.add_argument('--max-random-h', type=int, default=0,
                     help='max change of hue, whose range is [0, 180]')
    aug.add_argument('--max-random-s', type=int, default=0,
                     help='max change of saturation, whose range is [0, 255]')
    aug.add_argument('--max-random-l', type=int, default=0,
                     help='max change of intensity, whose range is [0, 255]')
    aug.add_argument('--pca-noise', type=float, default=0,
                     help='pca noise, whose range is [0, 1]')			 
    aug.add_argument('--pad-size', type=int, default=4,help='padding the input image')
    aug.add_argument('--fill-value', type=int, default=127,help='Set the padding pixels value to fill_value')

def add_data_arg(parser):
    data_arg = parser.add_argument_group('Data','data')
    data_arg.add_argument('--data-train',type=str)
    data_arg.add_argument('--data-val',type=str)
    data_arg.add_argument('--image-shape',type=str)
    data_arg.add_argument('--rgb-mean',type=str)
    data_arg.add_argument('--rgb-std',type=str)
    data_arg.add_argument('--num-classes',type=int)
    data_arg.add_argument('--num-examples',type=int)
    data_arg.add_argument('--data-nthreads',type=int)

def add_train_arg(parser):
    train_arg = parser.add_argument_group('Trainging','model trainging')
    train_arg.add_argument('--batch-size', type=int)
    train_arg.add_argument('--gpus', type=str)
    train_arg.add_argument('--lr', type=float)
    train_arg.add_argument('--wd', type=float,help='weight decay for sgd')
    train_arg.add_argument('--optimizer', type=str,help='the optimizer type')
    train_arg.add_argument('--mom', type=float,help='momentum for sgd')
    train_arg.add_argument('--top-k', type=int)
    train_arg.add_argument('--loss', type=str)
    train_arg.add_argument('--num-epochs', type=int)
    train_arg.add_argument('--lr-step-epochs', type=str)
    train_arg.add_argument('--lr-factor', type=float,help='the ratio to reduce lr on each step')
    train_arg.add_argument('--disp-batches', type=int)
    train_arg.add_argument('--model-prefix', type=str,help='model prefix')
    train_arg.add_argument('--save-period', type=int,help='params saving period')
    train_arg.add_argument('--load-epoch', type=int,help='load the model on an epoch using the model-load-prefix')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train cifar10 with 8 layers resnet',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_data_arg(parser)
    add_train_arg(parser)
    add_data_aug_arg(parser)
    parser.set_defaults(
        # data
        data_train     = './data/train_train.rec',
        data_val       = './data/train_val.rec',
        image_shape    = '3,32,32',
        rgb_mean       = '125.307,122.961,113.8575',
        rgb_std        = '51.5865,50.847,51.255',
        num_classes    = 10,
        num_examples  = 45000,
        data_nthreads = 4,
        # train
        batch_size     = 128,
        gpus           = '2,3,4,5',
        lr             = 0.1,
        wd             = 0.0001,
        optimizer      = 'sgd',
        mom            = 0.9,
        top_k          = 1,
        # cross-entropy or negative likelihood loss
        loss           = 'ce,nll',
        num_epochs     = 200,
        lr_step_epochs = '100,150',
        lr_factor      = 0.1,
        disp_batches   = 20,
        model_prefix   = 'mx_resnet8',
        save_period    = 10,
        load_epoch     = None
    )
    
    args = parser.parse_args()
    # Aggregates gradients and updates weights on GPUs. With this setting,
    # the KVStore also attempts to use GPU peer-to-peer communication,
    # potentially accelerating the communication.
    kv = mx.kvstore.create('device')
    # logging
    head = '%(asctime)-15s Node[' + str(kv.rank) + '] %(message)s'
    logging.basicConfig(level=logging.DEBUG,format=head)
    logging.info('start...')
    # epoch_size * batch_size =  num_examples / num_workers
    epoch_size = get_epoch_size(args,kv)
    logging.info('epoch_size = %d',epoch_size)
    
    (train, val) = data.get_rec_iter(args, kv)
    
    devs = mx.cpu() if args.gpus is None or args.gpus == "" else [
        mx.gpu(int(i)) for i in args.gpus.split(',')]

    lr, lr_scheduler= _get_lr_scheduler(args,kv)

    network = resnet_8layers.get_symbol()

    model = mx.module.Module(
            context = devs,
            symbol = network
    )
    optimizer_params = {
        'learning_rate': lr,
        'wd': args.wd,
        'lr_scheduler': lr_scheduler,
        'multi_precision': True
        }
    has_momentum = {'sgd', 'dcasgd', 'nag', 'signum', 'lbsgd'}
    if args.optimizer in has_momentum:
        optimizer_params['momentum'] = args.mom
    initializer = mx.init.Xavier(
        rnd_type='gaussian', factor_type="in", magnitude=2)
    eval_metrics = ['accuracy']
    # use accuracy if top_k is no more than 1
    if args.top_k > 1:
        eval_metrics.append(mx.metric.create('top_k_accuracy', top_k=args.top_k))
    supported_loss = ['ce', 'nll_loss']
    if len(args.loss) > 0:
        # ce or nll loss is only applicable to softmax output
        loss_type_list = args.loss.split(',')
        if 'softmax_output' in network.list_outputs():
            for loss_type in loss_type_list:
                loss_type = loss_type.strip()
                if loss_type == 'nll':
                    loss_type = 'nll_loss'
                if loss_type not in supported_loss:
                    logging.warning(loss_type + ' is not an valid loss type, only cross-entropy or ' \
                                    'negative likelihood loss is supported!')
                else:
                    eval_metrics.append(mx.metric.create(loss_type))
        else:
            logging.warning("The output is not softmax_output, loss argument will be skipped!")

    batch_end_callbacks = [mx.callback.Speedometer(args.batch_size, args.disp_batches)]
    checkpoint = _save_model(args, kv.rank)
    load_symbol, arg_params, aux_params = _load_model(args, kv.rank)
    if load_symbol is not None:
        assert load_symbol.tojson() == network.tojson()
    model.fit(train,
            begin_epoch = args.load_epoch if args.load_epoch else 0,
            num_epoch=args.num_epochs,
            eval_data=val,
            eval_metric=eval_metrics,
            kvstore=kv,
            optimizer=args.optimizer,
            optimizer_params=optimizer_params,
            initializer=initializer,
            arg_params=arg_params,
            aux_params=aux_params,
            batch_end_callback=batch_end_callbacks,
            epoch_end_callback=checkpoint,
            allow_missing=True
           )

