from mxnet.io import DataBatch, DataIter
import mxnet as mx

def get_rec_iter(args,kv=None):
    image_shape = tuple([int(l) for l in args.image_shape.split(',')])
    if kv:
        (rank, nworker) = (kv.rank, kv.num_workers)
    else:
        (rank, nworker) = (0, 1)
    rgb_mean = [float(i) for i in args.rgb_mean.split(',')]
    rgb_std = [float(i) for i in args.rgb_std.split(',')]
    train = mx.io.ImageRecordIter(
        path_imgrec         = args.data_train,
        label_width         = 1,
        mean_r              = rgb_mean[0],
        mean_g              = rgb_mean[1],
        mean_b              = rgb_mean[2],
        std_r               = rgb_std[0],
        std_g               = rgb_std[1],
        std_b               = rgb_std[2],
        data_name           = 'data',
        label_name          = 'softmax_label',
        data_shape          = image_shape,
        batch_size          = args.batch_size,
        preprocess_threads  = args.data_nthreads,
        shuffle             = True,
        num_parts           = nworker,
        part_index          = rank,
        # random crop
        rand_crop           = args.random_crop,
        rand_mirror         = args.random_mirror,
        min_random_scale    = args.min_random_scale,
        max_random_scale    = args.max_random_scale,
        pad                 = args.pad_size,
        fill_value          = args.fill_value,
        random_resized_crop = args.random_resized_crop,
        min_aspect_ratio    = args.min_random_aspect_ratio,
        max_aspect_ratio    = args.max_random_aspect_ratio,
        min_random_area     = args.min_random_area,
        max_random_area     = args.max_random_area,
        min_crop_size       = args.min_crop_size,
        max_crop_size       = args.max_crop_size,
        brightness          = args.brightness,
        contrast            = args.contrast,
        saturation          = args.saturation,
        pca_noise           = args.pca_noise,
        random_h            = args.max_random_h,
        random_s            = args.max_random_s,
        random_l            = args.max_random_l,
        max_rotate_angle    = args.max_random_rotate_angle,
        max_shear_ratio     = args.max_random_shear_ratio
        )
    val = mx.io.ImageRecordIter(
        path_imgrec         = args.data_val,
        label_width         = 1,
        mean_r              = rgb_mean[0],
        mean_g              = rgb_mean[1],
        mean_b              = rgb_mean[2],
        std_r               = rgb_std[0],
        std_g               = rgb_std[1],
        std_b               = rgb_std[2],
        data_name           = 'data',
        label_name          = 'softmax_label',
        batch_size          = args.batch_size,
        data_shape          = image_shape,
        preprocess_threads  = args.data_nthreads,
        num_parts           = nworker,
        part_index          = rank,
        rand_crop           = False,
        rand_mirror         = False
        )
    return (train, val)