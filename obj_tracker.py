import sys
sys.path.insert(1,'/home/atharva/traffic_detection_system/backend')
import time, random
import numpy as np
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs, convert_boxes

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from PIL import Image
from flask_pymongo import PyMongo
import json
from bson import ObjectId
import time

flags.DEFINE_string('classes', '/home/atharva/traffic_detection_system/backend/data/labels/coco.names', 'path to classes file')
flags.DEFINE_string('weights', '/home/atharva/traffic_detection_system/backend/weights/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video', '/home/atharva/traffic_detection_system/backend/data/video/video2.mp4',
                    'path to video file or number for webcam)')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')

vehicle_info = []

def main(f_app):
    class JSONEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, ObjectId):
                return str(o)
            return json.JSONEncoder.default(self, o)
    class vehicle:
        def __init__(self,vehicle_id,vehicle_sframe,center_y,vehicle_label='car',vehicle_speed=0):
            self.vehicle_id = vehicle_id
            self.vehicle_sframe = vehicle_sframe
            self.center_y = center_y
            self.vehicle_label = vehicle_label
            self.vehicle_speed = vehicle_speed
            self.vehicle_id = vehicle_id            
    

    # Definition of the parameters
    max_cosine_distance = 0.5
    nn_budget = None
    nms_max_overlap = 1.0
    
    #initialize deep sort
    model_filename = '/home/atharva/traffic_detection_system/backend/model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    yolo = YoloV3(classes=80)

    yolo.load_weights('/home/atharva/traffic_detection_system/backend/weights/yolov3.tf')
    logging.info('weights loaded')

    class_names = [c.strip() for c in open('/home/atharva/traffic_detection_system/backend/data/labels/coco.names').readlines()]
    logging.info('classes loaded')

    try:
        vid = cv2.VideoCapture(int('/home/atharva/traffic_detection_system/backend/data/video/cam5.mkv'))
    except:
        vid = cv2.VideoCapture('/home/atharva/traffic_detection_system/backend/data/video/cam5.mkv')

    out = None

    # if FLAGS.output:
    #     # by default VideoCapture returns float instead of int
    #     width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    #     height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #     fps = int(vid.get(cv2.CAP_PROP_FPS))
    #     codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
    #     out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))
    #     list_file = open('detection.txt', 'w')
    #     frame_index = -1 
    mongo = PyMongo(f_app, uri="mongodb+srv://testUser:testUser@gql-firstlast.teksa.mongodb.net/traffic?retryWrites=true&w=majority")
    db = mongo.db
    fps = 0.0
    fps_real = 20
    count = 0 
    car = 0
    truck = 0
    van = 0
    bus = 0
    max_car =0
    max_truck =0
    max_van =0
    max_bus =0
    frame = 0
    pos_line1 = 258
    pos_line2 = 335
    offset = 4
    str1 = "/home/atharva/traffic_detection_system/dr_test/cam5/cam5_fr"

    vehicle_obj = []
    vehicle_speed = []
    ids = []

    while True:
        _, img = vid.read()

        if img is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            count+=1
            if count < 3:
                continue
            else: 
                break
        
        frame+=1

        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, 416)

        t1 = time.time()
        boxes, scores, classes, nums = yolo.predict(img_in)
        classes = classes[0]
        names = []
        for i in range(len(classes)):
            names.append(class_names[int(classes[i])])
        names = np.array(names)
        converted_boxes = convert_boxes(img, boxes[0])
        features = encoder(img, converted_boxes)    
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(converted_boxes, scores[0], names, features)]
        
        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima suppresion
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]        
        
        # print(boxs)
        # Call the tracker
        tracker.predict()
        tracker.update(detections)
    
        #cam1
        # cv2.line(img, (280, pos_line1), (400, pos_line1), (0,255,0), 1)
        # cv2.line(img, (180, pos_line2), (375, pos_line2), (0,255,0), 1)

        # cv2.line(img,(0,480),(365,190),(0,255,0),1)
        # cv2.line(img,(320,480),(420,190),(0,255,0),1)

        #cam3
        # cv2.line(img,(10,480),(210,210),(0,255,0),1)
        # cv2.line(img,(340,480),(340,210),(0,255,0),1)

        #cam5
        cv2.line(img,(0,480),(200,170),(0,255,0),1)
        cv2.line(img,(340,480),(320,170),(0,255,0),1)

        # cv2.line(img, (650, pos_line1), (1200, pos_line1), (255,127,0), 1)
        # cv2.line(img, (670, pos_line2), (1220, pos_line2), (255,127,0), 1)


        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            center_y = (bbox[1]+bbox[3])/2
            center_x = (bbox[0]+bbox[2])/2
            # print(bbox)
            global vehicle_info

            if center_x<600 and center_y<=(pos_line1+offset) and center_y>=(pos_line1-offset):
                vehicle_obj.append(vehicle(track.track_id,frame,center_y))
                file_name = str(track.track_id)+".jpg"
                cv2.imwrite(file_name,img[int(max(bbox[1],0)):int(bbox[3]),int(max(bbox[0],0)):int(bbox[2]),:])
            elif center_x<600 and center_y<=(pos_line2+offset) and center_y>=(pos_line2-offset):
                for i in vehicle_obj:
                    if i.vehicle_id == track.track_id and track.track_id not in ids:
                        ids.append(track.track_id)
                        speed = 3.6*(20+(abs(i.center_y-pos_line1)+abs(center_y-pos_line2))*2/15)*fps_real/(frame-i.vehicle_sframe)
                        print("Vehicle id : ",track.track_id,"Lane : 1\nSpeed : ",speed," km/hr")
                        # vehicle_speed.append(speed)
                        # localtime = time.asctime( time.localtime(time.time()) )
                        # vehicle_data = {"vehicleId":track.track_id,"vehicleClass":track.get_class(),"vehicleSpeed":speed,"rec_time":localtime,"location":"Barstow Fwy, CA", "lane":1}            
                        # vehicle_info.append(vehicle_data)
                        # vehicle_data = json.dumps(vehicle_data)
                        # vehicle_data = json.loads(vehicle_data)
                        # #Insert into database
                        # db.vehicles.insert_one(vehicle_data)
                        break
                try:
                    vehicle_obj.remove(vehicle(track.track_id,frame))
                except:
                    pass

            if center_x>600 and center_y<=(pos_line2+offset) and center_y>=(pos_line2-offset):
                vehicle_obj.append(vehicle(track.track_id,frame,center_y))
                file_name = str(track.track_id)+".jpg"
                cv2.imwrite(file_name,img[int(max(bbox[1],0)):int(bbox[3]),int(max(bbox[0],0)):int(bbox[2]),:])
            elif center_x>600 and center_y<=(pos_line1+offset) and center_y>=(pos_line1-offset):
                for i in vehicle_obj:
                    if i.vehicle_id == track.track_id and track.track_id not in ids:
                        ids.append(track.track_id)
                        speed = 3.6*(20+(abs(i.center_y-pos_line2)+abs(center_y-pos_line1))*2/15)*fps_real/(frame-i.vehicle_sframe)
                        print("Vehicle id : ",track.track_id,"Lane : 2\nSpeed : ",speed," km/hr")
                        # vehicle_speed.append(speed)
                        # localtime = time.asctime( time.localtime(time.time()) )
                        # vehicle_data = {"vehicleId":track.track_id,"vehicleClass":track.get_class(),"vehicleSpeed":speed,"rec_time":localtime, "location":"Barstow Fwy, CA", "lane":2}
                        # vehicle_info.append(vehicle_data)
                        # vehicle_data = json.dumps(vehicle_data)
                        # vehicle_data = json.loads(vehicle_data)
                        # #Insert into database
                        # db.vehicles.insert_one(vehicle_data)
                        break
                try:
                    vehicle_obj.remove(vehicle(track.track_id,frame))
                except:
                    pass
            
            class_name = track.get_class()
            if class_name == 'car' and track.track_id>max_car:
                max_car = track.track_id
                car+=1
            elif class_name == 'truck' and track.track_id>max_truck:
                max_truck = track.track_id
                truck+=1
            elif class_name == 'van' and track.track_id>max_van:
                max_van = track.track_id
                van+=1
            elif class_name == 'bus' and track.track_id>max_bus:
                max_bus = track.track_id
                bus+=1
            # print(track.track_id)
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]

            if class_name == 'car' or class_name == 'bus' or class_name == 'van' or class_name == 'truck':
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255,127,0), 1)
                # cv2.rectangle(img, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                img = cv2.putText(img, str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.3, (255,255,255),1)
            
        ### UNCOMMENT BELOW IF YOU WANT CONSTANTLY CHANGING YOLO DETECTIONS TO BE SHOWN ON SCREEN
        #for det in detections:
        #    bbox = det.to_tlbr() 
        #    cv2.rectangle(img,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)
        
        # print fps on screen 
        fps  = ( fps + (1./(time.time()-t1)) ) / 2

        if frame%5==0:
            str2 = str1 + str(frame) + ".png"
            cv2.imwrite(str2,img)
        # cv2.imshow('output', img)

        # if FLAGS.output:
        #     out.write(img)
        #     frame_index = frame_index + 1
        #     list_file.write(str(frame_index)+' ')
        #     if len(converted_boxes) != 0:
        #         for i in range(0,len(converted_boxes)):
        #             list_file.write(str(converted_boxes[i][0]) + ' '+str(converted_boxes[i][1]) + ' '+str(converted_boxes[i][2]) + ' '+str(converted_boxes[i][3]) + ' ')
        #     list_file.write('\n')


        # print("cars : ",car,"/ntrucks : ",truck,"/nbus : ",bus,"/nvan : ",van)
        
        ret, buffer = cv2.imencode('.jpg', img)
        img_frame = buffer.tobytes()
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img_frame + b'\r\n')  # concat frame one by one and show result

        # press q to quit
        if cv2.waitKey(1) == ord("q"):
            break
    vid.release()
    if FLAGS.ouput:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass
