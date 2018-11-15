"""
Simple script to run a forward pass with SSD-6D on a SIXD dataset with a trained model.
Usage:
    run.py [options]
    run.py (-h | --help)

Options:
    -d, --dataset=<string>   Path to SIXD dataset [default: /Users/kehl/Desktop/sixd/hinterstoisser]
    -s, --sequence=<int>     Number of the sequence [default: 1]
    -f, --frames=<int>       Number of frames to load [default: 10]
    -n, --network=<string>   Path to trained network [default: /Users/kehl/Dropbox/iccv-models/hinterstoisser_obj_01.pb]
    -t, --threshold=<float>  Threshold for the detection confidence [default: 0.5]
    -v, --views=<int>        Views to parse for 6D pose pooling [default: 3]
    -i, --inplanes=<int>     In-plane rotations to parse for 6D pose pooling [default: 3]
    -h --help                Show this message and exit

"""


import cv2
from docopt import docopt
import numpy as np
import tensorflow as tf

from .ssd.ssd_utils import load_frozen_graph, NMSUtility, process_detection_output
from .rendering.utils import precompute_projections, build_6D_poses, verify_6D_poses
from .rendering.utils import draw_detections_2D, draw_detections_3D
from .utils.sixd import load_sixd


def init_networks(path):
    # Build detection and NMS networks
    load_frozen_graph(path)
    return NMSUtility(max_output_size=100, iou_threshold=0.45)

def load_model_map(path, sess, seq, bench):

    models = sess.run(sess.graph.get_tensor_by_name('models:0'))
    models = [m.decode('utf-8') for m in models]  # Strings are byte-encoded
    views = sess.run(sess.graph.get_tensor_by_name('views:0'))
    inplanes = sess.run(sess.graph.get_tensor_by_name('inplanes:0'))

    if len(models) == 1:  # If single-object network
        models = ['obj_{:02d}'.format(seq)]  # Overwrite model name

    print('Models:', models)
    print('Views:', len(views))
    print('Inplanes:', len(inplanes))

    print('Precomputing projections for each used model...')
    model_map = bench.models  # Mapping from name to model3D instance
    for model_name in models:
        m = model_map[model_name]
        m.projections = precompute_projections(views, inplanes, bench.cam, m)

    return (model_map, models)

def predict_pose(sess, img, model_map, models, nms, threshold, views, inplanes, bench):

    # Read out constant information
    priors = sess.run(sess.graph.get_tensor_by_name('priors:0'))
    variances = sess.run(sess.graph.get_tensor_by_name('variances:0'))
    priors = np.concatenate((priors, variances), axis=1)

    # Get tensor handles
    tensor_in = sess.graph.get_tensor_by_name('input:0')
    tensor_loc = sess.graph.get_tensor_by_name('locations:0')
    tensor_cla = sess.graph.get_tensor_by_name('class_probs:0')
    tensor_view = sess.graph.get_tensor_by_name('view_probs:0')
    tensor_inpl = sess.graph.get_tensor_by_name('inplane_probs:0')

     # print('Priors:', priors.shape)


    input_shape = (1, 299, 299, 3)
    image = cv2.resize(img, (input_shape[2], input_shape[1]))
    image = image[np.newaxis, :]  # Bring image into 4D batch shape

    # Get the raw network output
    run = [tensor_loc, tensor_cla, tensor_view, tensor_inpl]
    encoded_boxes, cla_probs, view_probs, inpl_probs = sess.run(run, {tensor_in: image})

    # Extend rank because of buggy TF 1.0 softmax
    cla_probs = cla_probs[np.newaxis, :]
    view_probs = view_probs[np.newaxis, :]
    inpl_probs = inpl_probs[np.newaxis, :]

    # Read out the detections in proper format for us
    dets_2d = process_detection_output(sess, priors, nms, models,
                                       encoded_boxes, cla_probs, view_probs, inpl_probs,
                                       threshold, views, inplanes)

    # Convert the 2D detections with their view/inplane IDs into 6D poses
    dets_6d = build_6D_poses(dets_2d, model_map, bench.cam)[0]

    # (NOT INCLUDED HERE) Run pose refinement for each pose in pool

    # Pick for each detection the best pose from the 6D pose pool
    final = verify_6D_poses(dets_6d, model_map, bench.cam, img)

    return final, dets_6d, dets_2d

def visualize_pose(img, final, dets_6d, cam, model_map):
    cv2.imshow('2D boxes', draw_detections_2D(img, final))
    cv2.imshow('6D pools', draw_detections_3D(img, dets_6d, cam, model_map))
    cv2.imshow('Final poses', draw_detections_3D(img, final, cam, model_map))
    cv2.waitKey()


if __name__ == '__main__':

    args = docopt(__doc__)
    sixd_base = args["--dataset"]
    sequence = int(args["--sequence"])
    nr_frames = int(args["--frames"])
    network = args["--network"]
    threshold = float(args["--threshold"])
    views_to_parse = int(args["--views"])
    inplanes_to_parse = int(args["--inplanes"])

    nms = init_networks(path=network)

    with tf.Session() as sess:

        bench = load_sixd(sixd_base, nr_frames=nr_frames, seq=sequence)

        model_map, models = load_model_map(path=sixd_base, sess=sess, seq=sequence, bench=bench)


        # Process each frame separately
        for f in bench.frames:

            import pdb; pdb.set_trace()
            final, dets_6d, dets_2d = predict_pose(sess=sess, img=f.color, model_map=model_map, models=models,
                                                   nms=nms, threshold=threshold, views=views_to_parse,
                                                   inplanes=inplanes_to_parse, bench=bench)

            visualize_pose(img=f.color, final=final, dets_6d=dets_6d, cam=bench.cam, model_map=model_map)
