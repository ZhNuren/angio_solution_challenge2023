import numpy as np
from pydicom import dcmread
import argparse
from keras.models import load_model
import cv2
import easygui
from utils.region_segmentation.segment_regions import segment_regions
from utils.stenosis_segmentation.segment_stenosis import segment_stenosis
from utils.binary_segmentation.frame_extractor import extract_frame
import supervision as sv
from utils.preprocess.contrast_func import improve_contrast
from utils.region_segmentation.extract_cat import calculate_pixel_size, calculate_region_width
parser = argparse.ArgumentParser(
                    prog='Coronary Angiography Analysis Pipeline',
                    description='This program identifies coronary regions according to the Syntax Score methodology, detects atherosclerotic plaques inside them, and provides health report',
                    epilog='Distributed as a part of Angio Solution Challenge 2023 application')

parser.add_argument('--frame_weights', '-fw', default="utils/binary_segmentation/weights/binary_sample.h5")
parser.add_argument('--region_weights', '-rw', default="utils/region_segmentation/weights/cat_yolov8x.pt")
parser.add_argument('--stenosis_weights', '-sw', default="utils/stenosis_segmentation/weights/best.pt")
parser.add_argument('--output', '-o', default="out_binary.png")
args = parser.parse_args()

while True:
    img = cv2.imread("imgs/solution.jpg")
    cv2.putText(img, "u - upload DICOM file", (55, 55), cv2.FONT_HERSHEY_SIMPLEX,0.6, (0,0,255), 1, cv2.LINE_AA)
    cv2.putText(img, "q - quit", (55, 75), cv2.FONT_HERSHEY_SIMPLEX,0.6, (0,0,255), 1, cv2.LINE_AA)

    cv2.imshow("ARCADE", img)
    if cv2.waitKey() == ord('u'):
        file = easygui.fileopenbox()
        dcm = dcmread(file).pixel_array
    elif cv2.waitKey() == ord('q'):
        break

    best_mat, best_ids = extract_frame(dcm, args.frame_weights, 3)
    best_mat = np.expand_dims(best_mat, 3)
    print("Found 3 best frames: ", best_mat.shape)
    mat_frame = np.ones((512,1024,3), dtype=np.uint8)*255
    d3 = np.concatenate([best_mat, best_mat, best_mat], axis=3).astype(np.uint8)
    print("Converted to 3-dimensional RGB", d3.shape)
    id = 0
    command=0

    while True:
        if command == ord('f'):
            id += 1
            command=0
        elif command == ord('a'):
            command = 0
            pred, names = segment_regions(args.region_weights, d3[id%3])
            masks = pred.masks.masks
            class_mat, mat = segment_stenosis(args.stenosis_weights, improve_contrast(d3[id%3]))
            msks = cv2.resize(masks.permute(1,2,0).numpy(), (512,512)) * cv2.resize(mat, (512,512))[:,:,np.newaxis]
            pos = np.where(msks.sum(axis=0).sum(axis=0)>0)  
            stenosis = np.sum(msks[:,:,pos[0]], axis=2)
            report = ""
            for cl_name, rect in class_mat:
                for i, mask in enumerate(masks):
                    if (cv2.resize(mask, (512,512))*cv2.resize(rect, (512,512))).sum()
                    report += 

            
            pix_size = calculate_pixel_size(pred)
            print(calculate_region_width(stenosis, pix_size))
            cls = pred.boxes.cls[pos].numpy()      

            poly_annotator = sv.BoxAnnotator(
                    thickness=1,
                    text_thickness=1,
                    text_scale=0.4,
                    text_padding=2
                )
            detections = sv.Detections.from_yolov8(pred)
            detections.xyxy = detections.xyxy[pos]
            detections.class_id = detections.class_id[pos]
            detections.confidence = detections.confidence[pos]
            labels = [
                f"{names[class_id]} {confidence:0.2f}"
                for _, confidence, class_id, _
                in detections
            ]
            frame = poly_annotator.annotate(
                scene=d3[id%3], 
                detections=detections, 
                labels=labels,
                skip_label=False
            ) 
            print(frame.shape, frame[stenosis>0].shape)
            frame[stenosis>0] = np.array([0,0,255])


        mat_frame[0:512,0:512] = d3[id%3]
        cv2.putText(mat_frame, "a - analyze", (mat_frame.shape[1]-250, 55), cv2.FONT_HERSHEY_SIMPLEX,0.6, (0,0,255), 1, cv2.LINE_AA)
        cv2.putText(mat_frame, "f - next frame", (mat_frame.shape[1]-250, 75), cv2.FONT_HERSHEY_SIMPLEX,0.6, (0,0,255), 1, cv2.LINE_AA)
        cv2.putText(mat_frame, "b - back", (mat_frame.shape[1]-250, 95), cv2.FONT_HERSHEY_SIMPLEX,0.6, (0,0,255), 1, cv2.LINE_AA)
        cv2.imshow("ARCADE", mat_frame)
        k = cv2.waitKey(1)
        if k == ord('b'):
            break
        if k != -1 and k != ord('q'):
            command = k

        



    
