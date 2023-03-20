# Copyright (c) OpenMMLab. All rights reserved.
import torch

from .camera_base import CAMERAS, SingleCameraBase


@CAMERAS.register_module()
class SimpleCameraTorch(SingleCameraBase):
    """Camera model to calculate coordinate transformation with given
    intrinsic/extrinsic camera parameters.

    Notes:
        The keypoint coordinate should be an np.ndarray with a shape of
    [...,J, C] where J is the keypoint number of an instance, and C is
    the coordinate dimension. For example:

        [J, C]: shape of joint coordinates of a person with J joints.
        [N, J, C]: shape of a batch of person joint coordinates.
        [N, T, J, C]: shape of a batch of pose sequences.

    Args:
        param (dict): camera parameters including:
            - R: 3x3, camera rotation matrix (camera-to-world)
            - T: 3x1, camera translation (camera-to-world)
            - K: (optional) 2x3, camera intrinsic matrix
            - k: (optional) nx1, camera radial distortion coefficients
            - p: (optional) mx1, camera tangential distortion coefficients
            - f: (optional) 2x1, camera focal length
            - c: (optional) 2x1, camera center
        if K is not provided, it will be calculated from f and c.

    Methods:
        world_to_camera: Project points from world coordinates to camera
            coordinates
        camera_to_pixel: Project points from camera coordinates to pixel
            coordinates
        world_to_pixel: Project points from world coordinates to pixel
            coordinates
    """

    def __init__(self, param, device):

        self.param = {}
        # extrinsic param
        R = torch.tensor(param['R'], device=device)
        T = torch.tensor(param['T'], device=device)

        assert R.shape == (3, 3)
        assert T.shape == (3, 1)
        # The camera matrices are transposed in advance because the joint
        # coordinates are stored as row vectors.
        self.param['R_c2w'] = R.T
        self.param['T_c2w'] = T.T
        self.param['R_w2c'] = R
        self.param['T_w2c'] = -self.param['T_c2w'] @ self.param['R_w2c']

        # intrinsic param
        if 'K' in param:
            K = torch.tensor(param['K'], device=device)
            assert K.shape == (2, 3)
            self.param['K'] = K.T
            self.param['f'] = torch.tensor([[K[0, 0]], [K[1, 1]]],
                                           device=device)
            self.param['c'] = torch.tensor([[K[0, 2]], [K[1, 2]]],
                                           device=device)
        elif 'f' in param and 'c' in param:
            f = torch.tensor(param['f'], device=device)
            c = torch.tensor(param['c'], device=device)
            assert f.shape == (2, 1)
            assert c.shape == (2, 1)
            self.param['K'] = torch.cat([torch.diagflat(f), c], dim=-1).T
            self.param['f'] = f
            self.param['c'] = c
        else:
            raise ValueError('Camera intrinsic parameters are missing. '
                             'Either "K" or "f"&"c" should be provided.')

        # distortion param
        if 'k' in param and 'p' in param:
            self.undistortion = True
            self.param['k'] = torch.tensor(param['k'], device=device).view(-1)
            self.param['p'] = torch.tensor(param['p'], device=device).view(-1)
            assert len(self.param['k']) in {3, 6}
            assert len(self.param['p']) == 2
        else:
            self.undistortion = False

    def world_to_camera(self, X):
        assert isinstance(X, torch.Tensor)
        assert X.ndim >= 2 and X.shape[-1] == 3
        return X @ self.param['R_w2c'] + self.param['T_w2c']

    def camera_to_world(self, X):
        assert isinstance(X, torch.Tensor)
        assert X.ndim >= 2 and X.shape[-1] == 3
        return X @ self.param['R_c2w'] + self.param['T_c2w']

    def camera_to_pixel(self, X):
        assert isinstance(X, torch.Tensor)
        assert X.ndim >= 2 and X.shape[-1] == 3

        _X = X / X[..., 2:]

        if self.undistortion:
            k = self.param['k']
            p = self.param['p']
            _X_2d = _X[..., :2]
            r2 = (_X_2d**2).sum(-1)
            radial = 1 + sum(ki * r2**(i + 1) for i, ki in enumerate(k[:3]))
            if k.size == 6:
                radial /= 1 + sum(
                    (ki * r2**(i + 1) for i, ki in enumerate(k[3:])))

            tangential = 2 * (p[1] * _X[..., 0] + p[0] * _X[..., 1])

            _X[..., :2] = _X_2d * (radial + tangential)[..., None] + torch.ger(
                r2, p.flip([0])).reshape(_X_2d.shape)
        return _X @ self.param['K']
