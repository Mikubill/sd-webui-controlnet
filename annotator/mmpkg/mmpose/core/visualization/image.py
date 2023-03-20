# Copyright (c) OpenMMLab. All rights reserved.
import math
import os
import warnings

import cv2
import annotator.mmpkg.mmcv as mmcv
import numpy as np
from matplotlib import pyplot as plt
from annotator.mmpkg.mmcv.utils.misc import deprecated_api_warning
from annotator.mmpkg.mmcv.visualization.color import color_val

try:
    import trimesh
    has_trimesh = True
except (ImportError, ModuleNotFoundError):
    has_trimesh = False

try:
    os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
    import pyrender
    has_pyrender = True
except (ImportError, ModuleNotFoundError):
    has_pyrender = False


def imshow_bboxes(img,
                  bboxes,
                  labels=None,
                  colors='green',
                  text_color='white',
                  thickness=1,
                  font_scale=0.5,
                  show=True,
                  win_name='',
                  wait_time=0,
                  out_file=None):
    """Draw bboxes with labels (optional) on an image. This is a wrapper of
    mmcv.imshow_bboxes.

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): ndarray of shape (k, 4), each row is a bbox in
            format [x1, y1, x2, y2].
        labels (str or list[str], optional): labels of each bbox.
        colors (list[str or tuple or :obj:`Color`]): A list of colors.
        text_color (str or tuple or :obj:`Color`): Color of texts.
        thickness (int): Thickness of lines.
        font_scale (float): Font scales of texts.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str, optional): The filename to write the image.

    Returns:
        ndarray: The image with bboxes drawn on it.
    """

    # adapt to mmcv.imshow_bboxes input format
    bboxes = np.split(
        bboxes, bboxes.shape[0], axis=0) if bboxes.shape[0] > 0 else []
    if not isinstance(colors, list):
        colors = [colors for _ in range(len(bboxes))]
    colors = [mmcv.color_val(c) for c in colors]
    assert len(bboxes) == len(colors)

    img = mmcv.imshow_bboxes(
        img,
        bboxes,
        colors,
        top_k=-1,
        thickness=thickness,
        show=False,
        out_file=None)

    if labels is not None:
        if not isinstance(labels, list):
            labels = [labels for _ in range(len(bboxes))]
        assert len(labels) == len(bboxes)

        for bbox, label, color in zip(bboxes, labels, colors):
            if label is None:
                continue
            bbox_int = bbox[0, :4].astype(np.int32)
            # roughly estimate the proper font size
            text_size, text_baseline = cv2.getTextSize(label,
                                                       cv2.FONT_HERSHEY_DUPLEX,
                                                       font_scale, thickness)
            text_x1 = bbox_int[0]
            text_y1 = max(0, bbox_int[1] - text_size[1] - text_baseline)
            text_x2 = bbox_int[0] + text_size[0]
            text_y2 = text_y1 + text_size[1] + text_baseline
            cv2.rectangle(img, (text_x1, text_y1), (text_x2, text_y2), color,
                          cv2.FILLED)
            cv2.putText(img, label, (text_x1, text_y2 - text_baseline),
                        cv2.FONT_HERSHEY_DUPLEX, font_scale,
                        mmcv.color_val(text_color), thickness)

    if show:
        mmcv.imshow(img, win_name, wait_time)
    if out_file is not None:
        mmcv.imwrite(img, out_file)
    return img


@deprecated_api_warning({'pose_limb_color': 'pose_link_color'})
def imshow_keypoints(img,
                     pose_result,
                     skeleton=None,
                     kpt_score_thr=0.3,
                     pose_kpt_color=None,
                     pose_link_color=None,
                     radius=4,
                     thickness=1,
                     show_keypoint_weight=False):
    """Draw keypoints and links on an image.

    Args:
            img (str or Tensor): The image to draw poses on. If an image array
                is given, id will be modified in-place.
            pose_result (list[kpts]): The poses to draw. Each element kpts is
                a set of K keypoints as an Kx3 numpy.ndarray, where each
                keypoint is represented as x, y, score.
            kpt_score_thr (float, optional): Minimum score of keypoints
                to be shown. Default: 0.3.
            pose_kpt_color (np.array[Nx3]`): Color of N keypoints. If None,
                the keypoint will not be drawn.
            pose_link_color (np.array[Mx3]): Color of M links. If None, the
                links will not be drawn.
            thickness (int): Thickness of lines.
    """

    img = mmcv.imread(img)
    img_h, img_w, _ = img.shape

    for kpts in pose_result:

        kpts = np.array(kpts, copy=False)

        # draw each point on image
        if pose_kpt_color is not None:
            assert len(pose_kpt_color) == len(kpts)

            for kid, kpt in enumerate(kpts):
                x_coord, y_coord, kpt_score = int(kpt[0]), int(kpt[1]), kpt[2]

                if kpt_score < kpt_score_thr or pose_kpt_color[kid] is None:
                    # skip the point that should not be drawn
                    continue

                color = tuple(int(c) for c in pose_kpt_color[kid])
                if show_keypoint_weight:
                    img_copy = img.copy()
                    cv2.circle(img_copy, (int(x_coord), int(y_coord)), radius,
                               color, -1)
                    transparency = max(0, min(1, kpt_score))
                    cv2.addWeighted(
                        img_copy,
                        transparency,
                        img,
                        1 - transparency,
                        0,
                        dst=img)
                else:
                    cv2.circle(img, (int(x_coord), int(y_coord)), radius,
                               color, -1)

        # draw links
        if skeleton is not None and pose_link_color is not None:
            assert len(pose_link_color) == len(skeleton)

            for sk_id, sk in enumerate(skeleton):
                pos1 = (int(kpts[sk[0], 0]), int(kpts[sk[0], 1]))
                pos2 = (int(kpts[sk[1], 0]), int(kpts[sk[1], 1]))

                if (pos1[0] <= 0 or pos1[0] >= img_w or pos1[1] <= 0
                        or pos1[1] >= img_h or pos2[0] <= 0 or pos2[0] >= img_w
                        or pos2[1] <= 0 or pos2[1] >= img_h
                        or kpts[sk[0], 2] < kpt_score_thr
                        or kpts[sk[1], 2] < kpt_score_thr
                        or pose_link_color[sk_id] is None):
                    # skip the link that should not be drawn
                    continue
                color = tuple(int(c) for c in pose_link_color[sk_id])
                if show_keypoint_weight:
                    img_copy = img.copy()
                    X = (pos1[0], pos2[0])
                    Y = (pos1[1], pos2[1])
                    mX = np.mean(X)
                    mY = np.mean(Y)
                    length = ((Y[0] - Y[1])**2 + (X[0] - X[1])**2)**0.5
                    angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
                    stickwidth = 2
                    polygon = cv2.ellipse2Poly(
                        (int(mX), int(mY)), (int(length / 2), int(stickwidth)),
                        int(angle), 0, 360, 1)
                    cv2.fillConvexPoly(img_copy, polygon, color)
                    transparency = max(
                        0, min(1, 0.5 * (kpts[sk[0], 2] + kpts[sk[1], 2])))
                    cv2.addWeighted(
                        img_copy,
                        transparency,
                        img,
                        1 - transparency,
                        0,
                        dst=img)
                else:
                    cv2.line(img, pos1, pos2, color, thickness=thickness)

    return img


def imshow_keypoints_3d(
    pose_result,
    img=None,
    skeleton=None,
    pose_kpt_color=None,
    pose_link_color=None,
    vis_height=400,
    kpt_score_thr=0.3,
    num_instances=-1,
    *,
    axis_azimuth=70,
    axis_limit=1.7,
    axis_dist=10.0,
    axis_elev=15.0,
):
    """Draw 3D keypoints and links in 3D coordinates.

    Args:
        pose_result (list[dict]): 3D pose results containing:
            - "keypoints_3d" ([K,4]): 3D keypoints
            - "title" (str): Optional. A string to specify the title of the
                visualization of this pose result
        img (str|np.ndarray): Opptional. The image or image path to show input
            image and/or 2D pose. Note that the image should be given in BGR
            channel order.
        skeleton (list of [idx_i,idx_j]): Skeleton described by a list of
            links, each is a pair of joint indices.
        pose_kpt_color (np.ndarray[Nx3]`): Color of N keypoints. If None, do
            not nddraw keypoints.
        pose_link_color (np.array[Mx3]): Color of M links. If None, do not
            draw links.
        vis_height (int): The image height of the visualization. The width
                will be N*vis_height depending on the number of visualized
                items.
        kpt_score_thr (float): Minimum score of keypoints to be shown.
            Default: 0.3.
        num_instances (int): Number of instances to be shown in 3D. If smaller
            than 0, all the instances in the pose_result will be shown.
            Otherwise, pad or truncate the pose_result to a length of
            num_instances.
        axis_azimuth (float): axis azimuth angle for 3D visualizations.
        axis_dist (float): axis distance for 3D visualizations.
        axis_elev (float): axis elevation view angle for 3D visualizations.
        axis_limit (float): The axis limit to visualize 3d pose. The xyz
            range will be set as:
            - x: [x_c - axis_limit/2, x_c + axis_limit/2]
            - y: [y_c - axis_limit/2, y_c + axis_limit/2]
            - z: [0, axis_limit]
            Where x_c, y_c is the mean value of x and y coordinates
        figsize: (float): figure size in inch.
    """

    show_img = img is not None
    if num_instances < 0:
        num_instances = len(pose_result)
    else:
        if len(pose_result) > num_instances:
            pose_result = pose_result[:num_instances]
        elif len(pose_result) < num_instances:
            pose_result += [dict()] * (num_instances - len(pose_result))
    num_axis = num_instances + 1 if show_img else num_instances

    plt.ioff()
    fig = plt.figure(figsize=(vis_height * num_axis * 0.01, vis_height * 0.01))

    if show_img:
        img = mmcv.imread(img, channel_order='bgr')
        img = mmcv.bgr2rgb(img)
        img = mmcv.imrescale(img, scale=vis_height / img.shape[0])

        ax_img = fig.add_subplot(1, num_axis, 1)
        ax_img.get_xaxis().set_visible(False)
        ax_img.get_yaxis().set_visible(False)
        ax_img.set_axis_off()
        ax_img.set_title('Input')
        ax_img.imshow(img, aspect='equal')

    for idx, res in enumerate(pose_result):
        dummy = len(res) == 0
        kpts = np.zeros((1, 3)) if dummy else res['keypoints_3d']
        if kpts.shape[1] == 3:
            kpts = np.concatenate([kpts, np.ones((kpts.shape[0], 1))], axis=1)
        valid = kpts[:, 3] >= kpt_score_thr

        ax_idx = idx + 2 if show_img else idx + 1
        ax = fig.add_subplot(1, num_axis, ax_idx, projection='3d')
        ax.view_init(
            elev=axis_elev,
            azim=axis_azimuth,
        )
        x_c = np.mean(kpts[valid, 0]) if sum(valid) > 0 else 0
        y_c = np.mean(kpts[valid, 1]) if sum(valid) > 0 else 0
        ax.set_xlim3d([x_c - axis_limit / 2, x_c + axis_limit / 2])
        ax.set_ylim3d([y_c - axis_limit / 2, y_c + axis_limit / 2])
        ax.set_zlim3d([0, axis_limit])
        ax.set_aspect('auto')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.dist = axis_dist

        if not dummy and pose_kpt_color is not None:
            pose_kpt_color = np.array(pose_kpt_color)
            assert len(pose_kpt_color) == len(kpts)
            x_3d, y_3d, z_3d = np.split(kpts[:, :3], [1, 2], axis=1)
            # matplotlib uses RGB color in [0, 1] value range
            _color = pose_kpt_color[..., ::-1] / 255.
            ax.scatter(
                x_3d[valid],
                y_3d[valid],
                z_3d[valid],
                marker='o',
                color=_color[valid],
            )

        if not dummy and skeleton is not None and pose_link_color is not None:
            pose_link_color = np.array(pose_link_color)
            assert len(pose_link_color) == len(skeleton)
            for link, link_color in zip(skeleton, pose_link_color):
                link_indices = [_i for _i in link]
                xs_3d = kpts[link_indices, 0]
                ys_3d = kpts[link_indices, 1]
                zs_3d = kpts[link_indices, 2]
                kpt_score = kpts[link_indices, 3]
                if kpt_score.min() > kpt_score_thr:
                    # matplotlib uses RGB color in [0, 1] value range
                    _color = link_color[::-1] / 255.
                    ax.plot(xs_3d, ys_3d, zs_3d, color=_color, zdir='z')

        if 'title' in res:
            ax.set_title(res['title'])

    # convert figure to numpy array
    fig.tight_layout()
    fig.canvas.draw()
    img_w, img_h = fig.canvas.get_width_height()
    img_vis = np.frombuffer(
        fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(img_h, img_w, -1)
    img_vis = mmcv.rgb2bgr(img_vis)

    plt.close(fig)

    return img_vis


def imshow_mesh_3d(img,
                   vertices,
                   faces,
                   camera_center,
                   focal_length,
                   colors=(76, 76, 204)):
    """Render 3D meshes on background image.

    Args:
        img(np.ndarray): Background image.
        vertices (list of np.ndarray): Vetrex coordinates in camera space.
        faces (list of np.ndarray): Faces of meshes.
        camera_center ([2]): Center pixel.
        focal_length ([2]): Focal length of camera.
        colors (list[str or tuple or Color]): A list of mesh colors.
    """

    H, W, C = img.shape

    if not has_pyrender:
        warnings.warn('pyrender package is not installed.')
        return img

    if not has_trimesh:
        warnings.warn('trimesh package is not installed.')
        return img

    try:
        renderer = pyrender.OffscreenRenderer(
            viewport_width=W, viewport_height=H)
    except (ImportError, RuntimeError):
        warnings.warn('pyrender package is not installed correctly.')
        return img

    if not isinstance(colors, list):
        colors = [colors for _ in range(len(vertices))]
    colors = [color_val(c) for c in colors]

    depth_map = np.ones([H, W]) * np.inf
    output_img = img
    for idx in range(len(vertices)):
        color = colors[idx]
        color = [c / 255.0 for c in color]
        color.append(1.0)
        vert = vertices[idx]
        face = faces[idx]

        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2, alphaMode='OPAQUE', baseColorFactor=color)

        mesh = trimesh.Trimesh(vert, face)
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(ambient_light=(0.5, 0.5, 0.5))
        scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        camera = pyrender.IntrinsicsCamera(
            fx=focal_length[0],
            fy=focal_length[1],
            cx=camera_center[0],
            cy=camera_center[1],
            zfar=1e5)
        scene.add(camera, pose=camera_pose)

        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)
        light_pose = np.eye(4)

        light_pose[:3, 3] = np.array([0, -1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([0, 1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([1, 1, 2])
        scene.add(light, pose=light_pose)

        color, rend_depth = renderer.render(
            scene, flags=pyrender.RenderFlags.RGBA)

        valid_mask = (rend_depth < depth_map) * (rend_depth > 0)
        depth_map[valid_mask] = rend_depth[valid_mask]
        valid_mask = valid_mask[:, :, None]
        output_img = (
            valid_mask * color[:, :, :3] + (1 - valid_mask) * output_img)

    return output_img


def imshow_multiview_keypoints_3d(
    pose_result,
    skeleton=None,
    pose_kpt_color=None,
    pose_link_color=None,
    space_size=[8000, 8000, 2000],
    space_center=[0, -500, 800],
    kpt_score_thr=0.0,
):
    """Draw 3D keypoints and links in 3D coordinates.

    Args:
        pose_result (list[kpts]): The poses to draw. Each element kpts is
            a set of K keypoints as an Kx4 numpy.ndarray, where each
            keypoint is represented as x, y, z, score.
        skeleton (list of [idx_i,idx_j]): Skeleton described by a list of
            links, each is a pair of joint indices.
        pose_kpt_color (np.ndarray[Nx3]`): Color of N keypoints. If None, do
            not nddraw keypoints.
        pose_link_color (np.array[Mx3]): Color of M links. If None, do not
            draw links.
        space_size: (list). Default: [8000, 8000, 2000].
        space_center: (list). Default: [0, -500, 800].
        kpt_score_thr (float): Minimum score of keypoints to be shown.
            Default: 0.0.
    """
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlim3d(space_center[0] - space_size[0] * 0.5,
                  space_center[0] + space_size[0] * 0.5)
    ax.set_ylim3d(space_center[1] - space_size[1] * 0.5,
                  space_center[1] + space_size[1] * 0.5)
    ax.set_zlim3d(space_center[2] - space_size[2] * 0.5,
                  space_center[2] + space_size[2] * 0.5)
    pose_kpt_color = np.array(pose_kpt_color)
    pose_kpt_color = pose_kpt_color[..., ::-1] / 255.

    for kpts in pose_result:
        # draw each point on image
        xs, ys, zs, scores = kpts.T
        valid = scores > kpt_score_thr
        ax.scatter(
            xs[valid],
            ys[valid],
            zs[valid],
            marker='o',
            color=pose_kpt_color[valid])

        for link, link_color in zip(skeleton, pose_link_color):
            link_indices = [_i for _i in link]
            xs_3d = kpts[link_indices, 0]
            ys_3d = kpts[link_indices, 1]
            zs_3d = kpts[link_indices, 2]
            kpt_score = kpts[link_indices, 3]
            if kpt_score.min() > kpt_score_thr:
                # matplotlib uses RGB color in [0, 1] value range
                _color = np.array(link_color[::-1]) / 255.
                ax.plot(xs_3d, ys_3d, zs_3d, color=_color)

    # convert figure to numpy array
    fig.tight_layout()
    fig.canvas.draw()
    img_w, img_h = fig.canvas.get_width_height()
    img_vis = np.frombuffer(
        fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(img_h, img_w, -1)
    img_vis = mmcv.rgb2bgr(img_vis)

    plt.close(fig)

    return img_vis
