# commercial-detection 

# It is nowadays a common practice to embed commercial advertisement into various kinds of video contents, 
# including conventional scripted TV ads, drama series, movies, variaty shows, or other stream media like youtube videos. 
# Here the challenge is to:
#   1. Identify when an embedded commercials appears;
#   2. Identify the company name of this commercial;
#   3. Identify the product that the commercial is selling;
#   4. Calculate screen occupation of the embedded commercial;

# A video of a TV advertisement is selected as an example target (video.mp4)
# Training data set (data_generation/images) was obtained from google images. 
# Some images were screen shots of the web page, the random stuff around the object can serve as blank object reference, 
# which turns out improved accuracy than simply saving those images.

# Method: Tensorflow object detection API
# Starting Model: ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03
# Image labelling tool: LabelImg
# Data format: tfrecord (converted from xml using xml_to_csv.py and generate_tfrecord.py)

# Steps: 
#   1. Label images using LabelImg
#   2. Run xml_to_csv.py to get generated_training_data/training_data.csv
#   3. Run generate_tfrecord.py to get generated_training_data/training_data.record
#   4. Prepare the training/config/object-detection.pbtxt file as the label map
#   5. Edit the training/config/ppn_pipeline.config file to prepare training
#   6. Set all path accordingly in the config file and train_ppn.py, then run train_ppn.py
#   7. Monitor training process using tensorboard --logdir="training_output"
#   8. Run export_inference_graph_ppn.py to generate frozen graph
#   9. Run video_test.py to test result

# Obtained 50 fps on i5-6400+GTX 970
# Still need to improve accuracy, but the exported_model_hougumixi0727 froze graph (trained with only 91 labeled images) can already mostly detect all advertized product
