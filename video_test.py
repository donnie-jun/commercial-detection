import sys
import os
import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import time
from PIL import Image

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

################### Definitions for model #############################

PATH_TO_CKPT = 'exported_model_trained/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('training/config', 'object-detection.pbtxt')
NUM_CLASSES = 2
#PATH_TO_CKPT = 'ssd_mobilenet_v1_ppn_coco/frozen_inference_graph.pb'
#PATH_TO_LABELS = 'ssd_mobilenet_v1_ppn_coco/mscoco_label_map.pbtxt'
#NUM_CLASSES = 90

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)
print(category_index)

#######################################################################

CONFIDENCE_FACTOR = 0.4

def detect_alert(boxes, classes, scores, category_index,
                 max_boxes_to_draw=10, min_score_thresh=0.6):
    returnlist = []
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores is None or scores[i] > min_score_thresh:
            classname = None
            classscore = None
            if category_index[classes[i]]['name']:
                classname = category_index[classes[i]]['name']
                classscore = int(100 * scores[i])
            line = {}
            line[classname] = classscore
            returnlist.append(line)
    return returnlist

def detect_objects(image_np, sess, detection_graph):
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Actual detection.
    tic=time.time()
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    toc=time.time()
    alert_array = detect_alert(np.squeeze(boxes), np.squeeze(classes).astype(np.int32), np.squeeze(scores),
                               category_index, max_boxes_to_draw=10, min_score_thresh=CONFIDENCE_FACTOR)
    return alert_array, toc-tic

def process_image(image, sess, detection_graph):    
    alert_array, tictoc = detect_objects(image, sess, detection_graph)
    alert = False
    prob=0
    for q in alert_array:
        if 'car' in q:
            if q['car'] > CONFIDENCE_FACTOR*100 and q['car'] > prob: #ruling for max prob
                alert = True
                prob = q['car']    
    return alert,prob, tictoc

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def load_image_into_numpy_array_video(image):
  (im_width, im_height, _) = image.shape
  return np.array(image).reshape(
      (im_width, im_height, 3)).astype(np.uint8)

def run_inference_for_single_image(image, sess, graph):
  with graph.as_default():
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

def main():
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    config = tf.ConfigProto(
        device_count = {'GPU': 1}
    )
    sess=tf.Session(graph=detection_graph,config=config)

    video = cv2.VideoCapture('video.mp4') #change the path of the video
    count = 0
    success = True
        
    cartimelist=[]
    carscorelist=[]
    with detection_graph.as_default():

        if True: # Toggle on/off an image-test step before video-test
            image = Image.open('testimage/image.jpg')
            image_np = load_image_into_numpy_array(image)
            output_dict = run_inference_for_single_image(image_np, sess, detection_graph)
            print(output_dict)
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                category_index,
                #instance_masks=output_dict.get('detection_masks'),
                use_normalized_coordinates=True,
                min_score_thresh=0.5,
                line_thickness=3)

            # Saving the labeled image
            width=image_np.shape[1]
            height=image_np.shape[0]
            AR=width/height
            imgsize=(8*AR,8)
            fig=plt.figure(figsize=imgsize,frameon=False)
            fig.add_subplot(111)
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            plt.axis('off')
            plt.imshow(image_np)
            fig.savefig('testimage/outputfigure.jpg')
            plt.show()
            plt.close()

            print("Image test completed, enter continue to video test...")
            input() # Wait input to continue
        
        while success:
            if count==1: tic0=time.time()
            count += 1
            tic=time.time()
            success, image = video.read()

            #image = cv2.resize(image,(300,300))
            
            if count%1==0 and success:
                imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                frametime=video.get(cv2.CAP_PROP_POS_MSEC)/1000
                #plt.figure(figsize=IMAGE_SIZE,frameon=False)
                #plt.imshow(image)
                #plt.axis('off')
                #plt.savefig('image_series/Time'+str(frametime)+'s.jpg',
                #            bbox_inches='tight')
                #plt.close()
                
                output_dict = run_inference_for_single_image(imageRGB, sess, detection_graph)
                toc=time.time()
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image,
                    output_dict['detection_boxes'],
                    output_dict['detection_classes'],
                    output_dict['detection_scores'],
                    category_index,
                    #instance_masks=output_dict.get('detection_masks'),
                    use_normalized_coordinates=True,
                    min_score_thresh = CONFIDENCE_FACTOR,
                    line_thickness=4)
                """
                plt.figure(figsize=IMAGE_SIZE)
                plt.imshow(image)
                plt.savefig('Time'+str(frametime)+'s.jpg')
                plt.close()
                """
                
                cv2.imshow("result", image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                print("Inference time: {}, detected: {}".format(toc-tic, output_dict['num_detections']))

                prob = max(output_dict['detection_scores'])
                if prob<CONFIDENCE_FACTOR: prob=0
                    
                list.append(cartimelist,frametime)
                list.append(carscorelist,prob)
    toc0=time.time()
    print('total time: '+str(toc0-tic0)+'s ,average:'+str(1000*(toc0-tic0)/(count-1))+' ms')
    plt.figure()
    plt.plot(cartimelist,carscorelist)
    plt.xlabel('Time(s)')
    plt.ylabel('Confidence(%)')
    plt.savefig('testimage/plot.jpg')
    plt.close()
    
if __name__ == '__main__':
    main()
