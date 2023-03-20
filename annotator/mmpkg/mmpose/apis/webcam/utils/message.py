# Copyright (c) OpenMMLab. All rights reserved.
import time
import uuid
import warnings
from typing import Callable, Dict, List, Optional

import numpy as np

Filter = Callable[[Dict], bool]


class Message():
    """Message base class.

    All message class should inherit this class. The basic use of a Message
    instance is to carray a piece of text message (self.msg) and a dict that
    stores structured data (self.data), e.g. frame image, model prediction,
    et al.

    A message may also hold route information, which is composed of
    information of all nodes the message has passed through.

    Parameters:
        msg (str): The text message.
        data (dict, optional): The structured data.
    """

    def __init__(self, msg: str = '', data: Optional[Dict] = None):
        self.msg = msg
        self.data = data if data else {}
        self.route_info = []
        self.timestamp = time.time()
        self.id = uuid.uuid1()

    def update_route_info(self,
                          node=None,
                          node_name: Optional[str] = None,
                          node_type: Optional[str] = None,
                          info: Optional[Dict] = None):
        """Append new node information to the route information.

        Args:
            node (Node, optional): An instance of Node that provides basic
                information like the node name and type. Default: ``None``.
            node_name (str, optional): The node name. If node is given,
                node_name will be ignored. Default: ``None``.
            node_type (str, optional): The class name of the node. If node
                is given, node_type will be ignored. Default: ``None``.
            info (dict, optional): The node information, which is usually
                given by node.get_node_info(). Default: ``None``.
        """
        if node is not None:
            if node_name is not None or node_type is not None:
                warnings.warn(
                    '`node_name` and `node_type` will be overridden if node'
                    'is provided.')
            node_name = node.name
            node_type = node.__class__.__name__

        node_info = {'node': node_name, 'node_type': node_type, 'info': info}
        self.route_info.append(node_info)

    def set_route_info(self, route_info: List[Dict]):
        """Directly set the entire route information.

        Args:
            route_info (list): route information to set to the message.
        """
        self.route_info = route_info

    def merge_route_info(self, route_info: List[Dict]):
        """Merge the given route information into the original one of the
        message. This is used for combining route information from multiple
        messages. The node information in the route will be reordered according
        to their timestamps.

        Args:
            route_info (list): route information to merge.
        """
        self.route_info += route_info
        self.route_info.sort(key=lambda x: x.get('timestamp', np.inf))

    def get_route_info(self) -> List:
        return self.route_info.copy()


class VideoEndingMessage(Message):
    """The special message to indicate the ending of the input video."""


class FrameMessage(Message):
    """The message to store information of a video frame."""

    def __init__(self, img):
        super().__init__(data=dict(image=img, objects={}, model_cfgs={}))

    def get_image(self) -> np.ndarray:
        """Get the frame image.

        Returns:
            np.ndarray: The frame image.
        """
        return self.data.get('image', None)

    def set_image(self, img):
        """Set the frame image to the message.

        Args:
            img (np.ndarray): The frame image.
        """
        self.data['image'] = img

    def set_objects(self, objects: List[Dict]):
        """Set the object information. The old object information will be
        cleared.

        Args:
            objects (list[dict]): A list of object information

        See also :func:`update_objects`.
        """
        self.data['objects'] = {}
        self.update_objects(objects)

    def update_objects(self, objects: List[Dict]):
        """Update object information.

        Each object will be assigned an unique ID if it does not has one. If
        an object's ID already exists in ``self.data['objects']``, the object
        information will be updated; otherwise it will be added as a new
        object.

        Args:
            objects (list[dict]): A list of object information
        """
        for obj in objects:
            if '_id_' in obj:
                # get the object id if it exists
                obj_id = obj['_id_']
            else:
                # otherwise assign a new object id
                obj_id = uuid.uuid1()
                obj['_id_'] = obj_id
            self.data['objects'][obj_id] = obj

    def get_objects(self, obj_filter: Optional[Filter] = None) -> List[Dict]:
        """Get object information from the frame data.

        Default to return all objects in the frame data. Optionally, filters
        can be set to retrieve objects with specific keys and values. The
        filters are represented as a dict. Each key in the filters specifies a
        required key of the object. Each value in the filters is a tuple that
        enumerate the required values of the corresponding key in the object.

        Args:
            obj_filter (callable, optional): A filter function that returns a
                bool value from a object (dict). If provided, only objects
                that return True will be retrieved. Otherwise all objects will
                be retrieved. Default: ``None``.

        Returns:
            list[dict]: A list of object information.


        Example::
            >>> objects = [
            ...     {'_id_': 2, 'label': 'dog'}
            ...     {'_id_': 1, 'label': 'cat'},
            ... ]
            >>> frame = FrameMessage(img)
            >>> frame.set_objects(objects)
            >>> frame.get_objects()
            [
                {'_id_': 1, 'label': 'cat'},
                {'_id_': 2, 'label': 'dog'}
            ]
            >>> frame.get_objects(obj_filter=lambda x:x['label'] == 'cat')
            [{'_id_': 1, 'label': 'cat'}]
        """

        objects = [
            obj.copy()
            for obj in filter(obj_filter, self.data['objects'].values())
        ]

        return objects
