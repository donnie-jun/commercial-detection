# commercial-detection 

### Objective
Here the challenge is to:
1. Identify when an embedded commercial appears;
2. Identify the company name of this commercial;
3. Identify the product that the commercial is selling;
4. Calculate screen occupation of the embedded commercial;

### Methods
A video of a TV advertisement is selected as an example target (video.mp4) 

Training dataset (data_generation/images) was obtained from google images. 

Some images were screenshots of the web page, the random stuff around the object can serve as a blank object reference, which turns out improved accuracy than simply saving those images.

API: Tensorflow object detection

Starting Model: ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03

Image labelling tool: LabelImg

Data format: tfrecord (converted from xml using xml_to_csv.py and generate_tfrecord.py)

Dependencies:
- Tensorflow
- Object detection API
- openCV
- pillow
- protobuf
- pandas
- etc...

### Steps
1. Label images using LabelImg
2. Run xml_to_csv.py to get generated_training_data/training_data.csv
3. Run generate_tfrecord.py to get generated_training_data/training_data.record
4. Prepare the training/config/object-detection.pbtxt file as the label map
5. Edit the training/config/ppn_pipeline.config file to prepare training
6. Set all path accordingly in the config file and train_ppn.py, then run train_ppn.py
7. Monitor training process using tensorboard --logdir="training_output"
8. Run export_inference_graph_ppn.py to generate frozen graph
9. Run video_test.py to test result

### Results
Obtained 50 fps on i5-6400 + GTX 970. 
Still need to improve accuracy, but the exported_model_trained froze graph (trained with only 91 labeled images) can already mostly detect all advertised product

![](https://github.com/donnie-jun/commercial-detection/blob/master/testimage/outputfigure1.jpg)
> Detected cup soup (an advertised product)

![](https://github.com/donnie-jun/commercial-detection/blob/master/testimage/outputfigure2.jpg)
> Detected overlapped cup soup and its box

![](https://github.com/donnie-jun/commercial-detection/blob/master/testimage/plot.jpg)
> Showing the appearance timing of the advertised product and detection confidence

![](https://github.com/donnie-jun/commercial-detection/blob/master/testimage/training_loss.JPG)
> Training loss over time (exported model was picked at step 17150)
