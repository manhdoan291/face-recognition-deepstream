
#!/usr/bin/env python3

################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################
from detection_info import *
import pyds
import ctypes
from apps.common.FPS import GETFPS
from apps.common.bus_call import bus_call
from apps.common.is_aarch_64 import is_aarch64
import time
import numpy as np
import math
from ctypes import *
from gi.repository import GLib
from gi.repository import GLib, Gst
import configparser
import gi
import sys
sys.path.append('../')
gi.require_version('Gst', '1.0')
import torch
import torchvision
CONF_THRESH = 0.75
IOU_THRESHOLD = 0.4
ctypes.cdll.LoadLibrary('/ws/fr_recognition_ds/face-detect-deepstream/yolov5s/libYoloV5Decoder.so')
ctypes.cdll.LoadLibrary('/ws/fr_recognition_ds/retinaface-deepstream-python/libdecodeplugin.so')
# ctypes.cdll.LoadLibrary(
#     '/ws/fr_recognition_ds/apps/deepstream-face-recognition/models/arcface/libArcFaceDecoder.so')
MAX_DISPLAY_LEN = 64
MUXER_OUTPUT_WIDTH = 1280
MUXER_OUTPUT_HEIGHT = 720
MUXER_BATCH_TIMEOUT_USEC = 4000000
TILED_OUTPUT_WIDTH = 1280
TILED_OUTPUT_HEIGHT = 720
GST_CAPS_FEATURES_NVMM = "memory:NVMM"
OSD_PROCESS_MODE = 1
OSD_DISPLAY_TEXT = 1
SECOND_DETECTOR_IS_SECONDARY = 0
MAX_ELEMENTS_IN_DISPLAY_META = 16
COORDINATES_SCALE_PARAM = 1
PRIMARY_DETECTOR_UID = 1
SECONDARY_DETECTOR_UID = 2
PGIE = 1
SGIE = 2
TGIE = 3
import scale_from_txt
import ptvsd

PERSON_DETECTED = {}
fps_streams = {}
RFACE_POOL = []
RFACE_POOL_MAX = 4
start_time = 0
end_time = 0
is_first = True
STREAMMUX_MAX_WIDTH = 1920
STREAMMUX_MAX_HEIGHT = 1080
def coor_scale(input_height, input_width, output_height, output_width):
    return max(output_height/input_height, output_width/input_width)

def parse_objects_from_tensor_meta(layer):
    num_detection = 0
    if layer.buffer:
        num_detection = int(pyds.get_detections(layer.buffer, 0))
        print(num_detection)
    
    num = num_detection * 15 + 1

    output = []
    for i in range(num):
        output.append(pyds.get_detections(layer.buffer, i))
    
    pred = np.reshape(output[1:], (-1, 15))[:num, :]

    pred = torch.Tensor(pred).cuda()
    # Get the boxes
    boxes = pred[:, :4]
    # Get the scores
    scores = pred[:, 4]
    # Get the landmark
    landmark = pred[:,5:15]
    # Choose those boxes that score > CONF_THRESH
    si = scores >= CONF_THRESH
    boxes = boxes[si, :]
    scores = scores[si]

    landmark = landmark[si,:]

    # Do nms
    indices = torchvision.ops.nms(boxes, scores, iou_threshold=IOU_THRESHOLD).cpu()
    result_boxes = boxes[indices, :].cpu()
    result_scores = scores[indices].cpu()
    result_landmark = landmark[indices].cpu()

    return result_landmark
    # return result_boxes, result_scores, result_landmark

def tiler_sink_pad_buffer_probe(pad, info, u_data):
    ptvsd.debug_this_thread()
    frame_number = 0
    num_rects = 0
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break
        frame_number = frame_meta.frame_num
        num_rects = frame_meta.num_obj_meta
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                # obj_meta = pyds.glist_get_nvds_object_meta(l_obj.data)
                obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            if obj_meta.unique_component_id == PGIE and obj_meta.class_id == 0 and (obj_meta.object_id in PERSON_DETECTED) == False:
                print("++New person id: {} detected on frame: {}.".format(
                    obj_meta.object_id, frame_number))
                PERSON_DETECTED[obj_meta.object_id] = [None, None]
            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                # obj_meta=pyds.glist_get_nvds_object_meta(l_obj.data)
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            if obj_meta.unique_component_id == SGIE and obj_meta.class_id == 0:
                # print("object id: "+str(obj_meta.object_id),
                #       "  ", str(obj_meta.parent.object_id))
                if obj_meta.parent.object_id in PERSON_DETECTED:
                    if PERSON_DETECTED[obj_meta.parent.object_id][0] is None:
                        print(
                            "face of person-{} detected in frame{}".format(obj_meta.parent.object_id, frame_number))
                        PERSON_DETECTED[obj_meta.parent.object_id][0] = frame_number

                        l_user = obj_meta.obj_user_meta_list
                        print("l_user co none ko: "+str(l_user))
                        while l_user is not None:
                            try:
                                # Note that l_user.data needs a cast to pyds.NvDsUserMeta
                                # The casting is done by pyds.NvDsUserMeta.cast()
                                # The casting also keeps ownership of the underlying memory
                                # in the C code, so the Python garbage collector will leave
                                # it alone
                                user_meta = pyds.NvDsUserMeta.cast(l_user.data)
                            except StopIteration:
                                break

                            # Check data type of user_meta
                            if (user_meta and user_meta.base_meta.meta_type == pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META):
                                try:
                                    tensor_meta = pyds.NvDsInferTensorMeta.cast(
                                        user_meta.user_meta_data)
                                except StopIteration:
                                    break

                                layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)
                                output = []
                                for i in range(512):
                                    output.append(
                                        pyds.get_detections(layer.buffer, i))

                                res = np.reshape(output, (512, -1))
                                norm = np.linalg.norm(res)
                                normal_array = res / norm
                                PERSON_DETECTED[obj_meta.parent.object_id][1] = normal_array
                                print("get facial features of person {}".format(
                                    obj_meta.parent.object_id))
                                try:
                                    l_user = l_user.next
                                except StopIteration:
                                    break

            try:
                l_obj = l_obj.next
            except StopIteration:
                break
        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def osd_sink_pad_buffer_probe(pad, info, u_data):
    ptvsd.debug_this_thread()
    frame_number = 0
    e_height, e_width = scale_from_txt.get_network_scale("/ws/fr_recognition_ds/face-detect-deepstream/retina_network_config.txt")
    # set the scale param for later display
    scale = coor_scale(e_height, e_width, STREAMMUX_MAX_HEIGHT, STREAMMUX_MAX_WIDTH)
    num_rects = 0
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    # print("l_frame co none ko:", str(l_frame))
    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break
        frame_number=frame_meta.frame_num
        result_landmark = []
        l_user=frame_meta.frame_user_meta_list
        l_obj = frame_meta.obj_meta_list
        while l_user is not None:
            try:
                # Note that l_user.data needs a cast to pyds.NvDsUserMeta
                # The casting is done by pyds.NvDsUserMeta.cast()
                # The casting also keeps ownership of the underlying memory
                # in the C code, so the Python garbage collector will leave
                # it alone
                user_meta=pyds.NvDsUserMeta.cast(l_user.data) 
            except StopIteration:
                break
            
            # Check data type of user_meta 
            if(user_meta and user_meta.base_meta.meta_type==pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META): 
                try:
                    tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
                except StopIteration:
                    break
                
                layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)
                result_landmark = parse_objects_from_tensor_meta(layer)
                      
            try:
                l_user=l_user.next
            except StopIteration:
                break    
        display_meta=pyds.nvds_acquire_display_meta_from_pool(batch_meta)

        
        num_rects = frame_meta.num_obj_meta
        l_obj=frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta = pyds.glist_get_nvds_object_meta(l_obj.data)
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            obj_meta.rect_params.border_color.set(1.0, 1.0, 1.0, 1.0)
            print("class id & conf: "+ obj_meta.class_id,
                  " ", obj_meta.confidence)
            try:
                l_obj = l_obj.next
            except StopIteration:
                break
            if obj_meta.unique_component_id == PRIMARY_DETECTOR_UID:
                # set bbox color in rgba
                obj_meta.rect_params.border_color.set(1.0, 1.0, 1.0, 0.0)
                # set the border width in pixel
                obj_meta.rect_params.border_width=2
                obj_meta.rect_params.has_bg_color=1
                # obj_meta.rect_params.bg_color.set(0.0, 0.5, 0.3, 0.4)
            if obj_meta.unique_component_id == SECONDARY_DETECTOR_UID:
                obj_meta.rect_params.border_color.set(1.0, 1.0, 1.0, 0.0)
                obj_meta.rect_params.border_width=2
                obj_meta.rect_params.has_bg_color=1
                # obj_meta.rect_params.bg_color.set(0.0, 0.5, 0.3, 0.4)
            try: 
                l_obj=l_obj.next

            except StopIteration:
                break

        # Acquiring a display meta object. The memory ownership remains in
        # the C code so downstream plugins can still access it. Otherwise
        # the garbage collector will claim it when this probe function exits.

        # draw 5 landmarks for each rect
        display_meta.num_circles = len(result_landmark) * 5
        ccount = 0
        for i in range(len(result_landmark)):
            # scale coordinates
            landmarks = result_landmark[i] * scale

            # nvosd struct can only draw MAX 16 elements once 
            # so acquire a new display meta for every face detected
            display_meta=pyds.nvds_acquire_display_meta_from_pool(batch_meta)   
            display_meta.num_circles = 5
            ccount = 0
            for j in range(5):
                py_nvosd_circle_params = display_meta.circle_params[ccount]
                py_nvosd_circle_params.circle_color.set(1.0, 1.0, 1.0, 1.0)
                py_nvosd_circle_params.has_bg_color = 2
                py_nvosd_circle_params.bg_color.set(0.0, 0.0, 0.0, 1.0)
                py_nvosd_circle_params.xc = int(landmarks[j * 2])
                py_nvosd_circle_params.yc = int(landmarks[j * 2 + 1])
                py_nvosd_circle_params.radius=10
                ccount = ccount + 1
            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

        display_meta=pyds.nvds_acquire_display_meta_from_pool(batch_meta)       
        display_meta.num_labels = 1
        py_nvosd_text_params = display_meta.text_params[0]
        # Setting display text to be shown on screen
        # Note that the pyds module allocates a buffer for the string, and the
        # memory will not be claimed by the garbage collector.
        # Reading the display_text field here will return the C address of the
        # allocated string. Use pyds.get_string() to get the string content.
        py_nvosd_text_params.display_text = "Frame Number={} Number of Objects={}".format(frame_number, num_rects)

        # Now set the offsets where the string should appear
        py_nvosd_text_params.x_offset = 10
        py_nvosd_text_params.y_offset = 12

        # Font , font-color and font-size
        py_nvosd_text_params.font_params.font_name = "Serif"
        py_nvosd_text_params.font_params.font_size = 10
        # set(red, green, blue, alpha); set to White
        py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)

        # Text background color
        py_nvosd_text_params.set_bg_clr = 1
        # set(red, green, blue, alpha); set to Black
        py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)
        # Using pyds.get_string() to get display_text as string
        print(pyds.get_string(py_nvosd_text_params.display_text))
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
        # Get frame rate through this probe
        fps_streams["stream{0}".format(frame_meta.pad_index)].get_fps()
        try:
            l_frame=l_frame.next
        except StopIteration:
            break
			
    return Gst.PadProbeReturn.OK	
        
        


def sgie_sink_pad_buffer_probe(pad, info, u_data):
    '''
    '''
    global RFACE_POOL_MAX, RFACE_POOL
    global start_time, end_time
    global is_first
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        if is_first == True:
            start_time = time.time()
            is_first = False
        if (end_time - start_time) >= 1:
            RFACE_POOL.clear()
            start_time = end_time
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            print("start time: "+str(start_time), "  ", "end time: " + str(end_time))
        except StopIteration:
            break
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            remove_flag = 1
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            # if obj_meta.class_id!=0:
                #remove_flag = 0
            if obj_meta.class_id == 0 and obj_meta.unique_component_id == PGIE and (obj_meta.rect_params.width < 120 or obj_meta.rect_params.height < 120):
                remove_flag = 1
            elif obj_meta.class_id == 0 and (obj_meta.object_id in PERSON_DETECTED) and (PERSON_DETECTED[obj_meta.object_id][0] is not None):
                if obj_meta.object_id in RFACE_POOL:
                    RFACE_POOL.remove(obj_meta.object_id)
                remove_flag = 1

            elif obj_meta.class_id == 0 and (obj_meta.object_id in PERSON_DETECTED) == False:
                if obj_meta.object_id in RFACE_POOL:
                    remove_flag = 1
                if obj_meta.object_id not in RFACE_POOL:
                    if len(RFACE_POOL) >= RFACE_POOL_MAX:
                        remove_flag = 1
                    else:
                        RFACE_POOL.append(obj_meta.object_id)
                        print("++Retina Face get in person id: "+str(RFACE_POOL))
                        remove_flag = 0
            try:
                l_obj = l_obj.next
                if remove_flag:
                    pyds.nvds_remove_obj_meta_from_frame(frame_meta, obj_meta)
            except StopIteration:
                break

        try:
            end_time = time.time()
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def main(args):
    # Check input arguments
    if len(args) < 2:
        sys.stderr.write("usage: %s <uri1> [uri2] ... [uriN]\n" % args[0])
        sys.exit(1)

    for i in range(0, len(args)-1):
        fps_streams["stream{0}".format(i)] = GETFPS(i)
    print("fps_streams: " + str(fps_streams))
    number_sources = len(args)-1

    # Standard GStreamer initialization
    GLib.threads_init()
    Gst.init(None)

    # Create gstreamer elements */
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()
    is_live = False

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")
    print("Creating streamux \n ")

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")

    pipeline.add(streammux)
    for i in range(number_sources):
        print("Creating source_bin ", i, " \n ")
        uri_name = args[i+1]
        if uri_name.find("rtsp://") == 0:
            is_live = True
        source_bin = create_source_bin(i, uri_name)
        if not source_bin:
            sys.stderr.write("Unable to create source bin \n")
        pipeline.add(source_bin)
        padname = "sink_%u" % i
        sinkpad = streammux.get_request_pad(padname)
        if not sinkpad:
            sys.stderr.write("Unable to create sink pad bin \n")
        srcpad = source_bin.get_static_pad("src")
        if not srcpad:
            sys.stderr.write("Unable to create src pad bin \n")
        srcpad.link(sinkpad)
    queue1 = Gst.ElementFactory.make("queue", "queue1")
    queue2 = Gst.ElementFactory.make("queue", "queue2")
    queue3 = Gst.ElementFactory.make("queue", "queue3")
    queue4 = Gst.ElementFactory.make("queue", "queue4")
    queue5 = Gst.ElementFactory.make("queue", "queue5")
    queue6 = Gst.ElementFactory.make("queue", "queue6")
    queue7 = Gst.ElementFactory.make("queue", "queue7")
    queue8 = Gst.ElementFactory.make("queue", "queue8")
    pipeline.add(queue1)
    pipeline.add(queue2)
    pipeline.add(queue3)
    pipeline.add(queue4)
    pipeline.add(queue5)
    pipeline.add(queue6)
    pipeline.add(queue7)
    pipeline.add(queue8)
    print("Creating Pgie \n ")
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create pgie \n")
    print("Creating sgie \n ")
    sgie = Gst.ElementFactory.make("nvinfer", "Secondary-inference")
    if not sgie:
        sys.stderr.write(" Unable to create sgie \n")

    print("Creating tgie \n ")
    tgie = Gst.ElementFactory.make("nvinfer", "Third-inference")
    if not tgie:
        sys.stderr.write(" Unable to create tgie \n")
    print("Creating tracker \n ")
    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    if not tracker:
        sys.stderr.write(" Unable to create tracker \n")
    print("Creating tiler \n ")
    tiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    if not tiler:
        sys.stderr.write(" Unable to create tiler \n")
    print("Creating nvvidconv \n ")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        sys.stderr.write(" Unable to create nvvidconv \n")
    print("Creating nvosd \n ")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    if not nvosd:
        sys.stderr.write(" Unable to create nvosd \n")
    nvosd.set_property('process-mode', OSD_PROCESS_MODE)
    nvosd.set_property('display-text', OSD_DISPLAY_TEXT)
    if (is_aarch64()):
        print("Creating transform \n ")
        transform = Gst.ElementFactory.make(
            "queue", "queue")
        if not transform:
            sys.stderr.write(" Unable to create transform \n")

    print("Creating EGLSink \n")
    #sink = Gst.ElementFactory.make("fakevideosink", "nvvideo-renderer")
    sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")

    if not sink:
        sys.stderr.write(" Unable to create egl sink \n")

    if is_live:
        print("Atleast one of the sources is live")
        streammux.set_property('live-source', 1)

    streammux.set_property('width', 1920)
    streammux.set_property('height', 1080)
    streammux.set_property('batch-size', 1)
    streammux.set_property('batched-push-timeout', 4000000)

    pgie.set_property(
        'config-file-path', "/ws/fr_recognition_ds/apps/deepstream-face-recognition/config_yolov5.txt")
    pgie_batch_size = pgie.get_property("batch-size")
    if (pgie_batch_size != number_sources):
        print("WARNING: Overriding infer-config batch-size",
              pgie_batch_size, " with number of sources ", number_sources, " \n")
        pgie.set_property("batch-size", number_sources)

    sgie.set_property(
        'config-file-path', "/ws/fr_recognition_ds/apps/deepstream-face-recognition/config_retinaface.txt")
    tgie.set_property(
        'config-file-path', "/ws/fr_recognition_ds/apps/deepstream-face-recognition/config_arcface.txt")

    tiler_rows = int(math.sqrt(number_sources))
    tiler_columns = int(math.ceil((1.0*number_sources)/tiler_rows))
    tiler.set_property("rows", tiler_rows)
    tiler.set_property("columns", tiler_columns)
    tiler.set_property("width", TILED_OUTPUT_WIDTH)
    tiler.set_property("height", TILED_OUTPUT_HEIGHT)
    sink.set_property("qos", 0)
    sink.set_property("sync", 1)

    # Set properties of tracker
    config = configparser.ConfigParser()
    config.read('/ws/fr_recognition_ds/retinaface-deepstream-python/config_tracker.txt')
    config.sections()

    for key in config['tracker']:
        if key == 'tracker-width':
            tracker_width = config.getint('tracker', key)
            tracker.set_property('tracker-width', tracker_width)
        if key == 'tracker-height':
            tracker_height = config.getint('tracker', key)
            tracker.set_property('tracker-height', tracker_height)
        if key == 'gpu-id':
            tracker_gpu_id = config.getint('tracker', key)
            tracker.set_property('gpu_id', tracker_gpu_id)
        if key == 'll-lib-file':
            tracker_ll_lib_file = config.get('tracker', key)
            tracker.set_property('ll-lib-file', tracker_ll_lib_file)
        if key == 'll-config-file':
            tracker_ll_config_file = config.get('tracker', key)
            tracker.set_property('ll-config-file', tracker_ll_config_file)
        if key == 'enable-past-frame':
            tracker_enable_past_frame = config.getint('tracker', key)
            tracker.set_property('enable_past_frame',
                                 tracker_enable_past_frame)

    print("Adding elements to Pipeline \n")
    pipeline.add(pgie)
    pipeline.add(sgie)
    pipeline.add(tgie)
    pipeline.add(tracker)
    pipeline.add(tiler)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    if is_aarch64():
        pipeline.add(transform)
    pipeline.add(sink)

    print("Linking elements in the Pipeline \n")
    
    streammux.link(queue1) #
    queue1.link(pgie)
    pgie.link(queue2)
    queue2.link(tracker)
    tracker.link(queue3)
    queue3.link(sgie)
    sgie.link(queue4)
    queue4.link(tgie)
    tgie.link(queue5)
    queue5.link(tiler)
    tiler.link(queue6)
    queue6.link(nvvidconv)
    nvvidconv.link(queue7)
    queue7.link(nvosd)
    nvosd.link(queue8)
    queue8.link(sink)

    # create an event loop and feed gstreamer bus mesages to it
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)
    result = Global_result()

    tiler_sink_pad = tiler.get_static_pad("sink")
    if not tiler_sink_pad:
        sys.stderr.write(" Unable to get sink pad of tiler \n")
    else:
        i = 1
        tiler_sink_pad.add_probe(
            Gst.PadProbeType.BUFFER, tiler_sink_pad_buffer_probe, 0)

    sgie_sink_pad = queue3.get_static_pad("sink")
    if not sgie_sink_pad:
        sys.stderr.write(" Unable to get sink pad of tiler \n")
    else:
        i = 2
        sgie_sink_pad.add_probe(Gst.PadProbeType.BUFFER,
                                sgie_sink_pad_buffer_probe, 0)

    osd_sink_pad = pgie.get_static_pad("sink")
    if not osd_sink_pad:
        sys.stderr.write(" Unable to get sink pad of osd \n")
    else:
        i = 3
        osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER,
                               osd_sink_pad_buffer_probe, 0)

    # List the sources
    print("Now playing...")
    for i, source in enumerate(args):
        if (i != 0):
            print("index uri video: "+str(i), ": ", source)

    print("Starting pipeline \n")
    # start play back and listed to events
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    # cleanup
    print("Exiting app\n")
    pipeline.set_state(Gst.State.NULL)


def cb_newpad(decodebin, decoder_src_pad, data):
    print("In cb_newpad\n")
    caps = decoder_src_pad.get_current_caps()
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    source_bin = data
    features = caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not
    # audio.
    print("gstname=", gstname)
    if (gstname.find("video") != -1):
        # Link the decodebin pad only if decodebin has picked nvidia
        # decoder plugin nvdec_*. We do this by checking if the pad caps contain
        # NVMM memory features.
        print("features=", features)
        if features.contains("memory:NVMM"):
            # Get the source bin ghost pad
            bin_ghost_pad = source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write(
                    "Failed to link decoder src pad to source bin ghost pad\n")
        else:
            sys.stderr.write(
                " Error: Decodebin did not pick nvidia decoder plugin.\n")


def decodebin_child_added(child_proxy, Object, name, user_data):
    print("Decodebin child added:", name, "\n")
    if (name.find("decodebin") != -1):
        Object.connect("child-added", decodebin_child_added, user_data)

    # if "source" in name:
    #     Object.set_property("drop-on-latency", True)


def create_source_bin(index, uri):
    print("Creating source bin")

    # Create a source GstBin to abstract this bin's content from the rest of the
    # pipeline
    bin_name = "source-bin-%02d" % index
    print(bin_name)
    nbin = Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write(" Unable to create source bin \n")

    # Source element for reading from the uri.
    # We will use decodebin and let it figure out the container format of the
    # stream and the codec and plug the appropriate demux and decode plugins.
    uri_decode_bin = Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    if not uri_decode_bin:
        sys.stderr.write(" Unable to create uri decode bin \n")
    # We set the input uri to the source element
    uri_decode_bin.set_property("uri", uri)
    # Connect to the "pad-added" signal of the decodebin which generates a
    # callback once a new pad for raw data has beed created by the decodebin
    uri_decode_bin.connect("pad-added", cb_newpad, nbin)
    uri_decode_bin.connect("child-added", decodebin_child_added, nbin)

    # We need to create a ghost pad for the source bin which will act as a proxy
    # for the video decoder src pad. The ghost pad will not have a target right
    # now. Once the decode bin creates the video decoder and generates the
    # cb_newpad callback, we will set the ghost pad target to the video decoder
    # src pad.
    Gst.Bin.add(nbin, uri_decode_bin)
    bin_pad = nbin.add_pad(
        Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
    if not bin_pad:
        sys.stderr.write(" Failed to add ghost pad in source bin \n")
        return None
    return nbin


if __name__ == '__main__':
    sys.exit(main(sys.argv))
