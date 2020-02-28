import os
import cv2
import json
import numpy as np


LABELS_FULL = {
    'Sunglasses': [170, 0, 51],
    'LeftArm': [51, 170, 221],
    'RightArm': [0, 255, 255],
    'LeftLeg': [85, 255, 170],
    'RightLeg': [170, 255, 85],
    'LeftShoe': [255, 255, 0],
    'RightShoe': [255, 170, 0],
}

LABELS_CLOTHING= {
    'Face': [0, 0, 255],
    'Arms': [51, 170, 221],
    'Legs': [85, 255, 170],
    'Shoes': [255, 255, 0]
}


def read_segmentation(file):
    segm = cv2.imread(file)[:, :, ::-1]

    segm[np.all(segm == LABELS_FULL['Sunglasses'], axis=2)] = LABELS_CLOTHING['Face']
    segm[np.all(segm == LABELS_FULL['LeftArm'], axis=2)] = LABELS_CLOTHING['Arms']
    segm[np.all(segm == LABELS_FULL['RightArm'], axis=2)] = LABELS_CLOTHING['Arms']
    segm[np.all(segm == LABELS_FULL['LeftLeg'], axis=2)] = LABELS_CLOTHING['Legs']
    segm[np.all(segm == LABELS_FULL['RightLeg'], axis=2)] = LABELS_CLOTHING['Legs']
    segm[np.all(segm == LABELS_FULL['LeftShoe'], axis=2)] = LABELS_CLOTHING['Shoes']
    segm[np.all(segm == LABELS_FULL['RightShoe'], axis=2)] = LABELS_CLOTHING['Shoes']

    return segm[:, :, ::-1] / 255.


def openpose_from_file(file, resolution=(1080, 1080), person=0):
    with open(file) as f:
        data = json.load(f)['people'][person]

        pose = np.array(data['pose_keypoints_2d']).reshape(-1, 3)
        pose[:, 2] /= np.expand_dims(np.mean(pose[:, 2][pose[:, 2] > 0.1]), -1)
        pose = pose * np.array([2. / resolution[1], -2. / resolution[0], 1.]) + np.array([-1., 1., 0.])
        pose[:, 0] *= 1. * resolution[1] / resolution[0]

        face = np.array(data['face_keypoints_2d']).reshape(-1, 3)
        face = face * np.array([2. / resolution[1], -2. / resolution[0], 1.]) + np.array([-1., 1., 0.])
        face[:, 0] *= 1. * resolution[1] / resolution[0]

        return pose, face


def write_mesh(filename, v, f):
    with open(filename, 'w') as fp:
        fp.write(('v {:f} {:f} {:f}\n' * len(v)).format(*v.reshape(-1)))
        fp.write(('f {:d} {:d} {:d}\n' * len(f)).format(*(f.reshape(-1) + 1)))


def write_mesh2(filename, v, f, vt=None, ft=None, vn=None, vc=None, texture=None):
    with open(filename, 'w') as fp:
        if texture is not None:
            mat_file = filename.replace('obj', 'mtl')

            fp.write('mtllib {}\n'.format(os.path.basename(mat_file)))
            fp.write('usemtl mat\n')

            with open(mat_file, 'w') as mfp:
                mfp.write('newmtl mat\n')
                mfp.write('Ka 1.0 1.0 1.0\n')
                mfp.write('Kd 1.0 1.0 1.0\n')
                mfp.write('Ks 0.0 0.0 0.0\n')
                mfp.write('d 1.0\n')
                mfp.write('Ns 0.0\n')
                mfp.write('illum 0\n')
                mfp.write('map_Kd {}\n'.format(texture))

        if vc is not None:
            fp.write(('v {:f} {:f} {:f} {:f} {:f} {:f}\n' * len(v)).format(*np.hstack((v, vc)).reshape(-1)))
        else:
            fp.write(('v {:f} {:f} {:f}\n' * len(v)).format(*v.reshape(-1)))

        if vn is not None:
            fp.write(('vn {:f} {:f} {:f}\n' * len(vn)).format(*vn.reshape(-1)))

        if vt is not None:
            fp.write(('vt {:f} {:f}\n' * len(vt)).format(*vt.reshape(-1)))

        if ft is not None:
            fp.write(('f {:d}/{:d}/{:d} {:d}/{:d}/{:d} {:d}/{:d}/{:d}\n' * len(f)).format(*np.hstack((f.reshape(-1, 1), ft.reshape(-1, 1), f.reshape(-1, 1))).reshape(-1) + 1))
        else:
            fp.write(('f {:d}//{:d} {:d}//{:d} {:d}//{:d}\n' * len(f)).format(*np.repeat(f.reshape(-1) + 1, 2)))
