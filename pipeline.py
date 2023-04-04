import numpy as np
from pydicom import dcmread
import argparse
from keras.models import load_model
import cv2
import easygui
import matplotlib.pyplot as plt
from utils.region_segmentation.segment_regions import segment_regions
from utils.stenosis_segmentation.segment_stenosis import segment_stenosis
from utils.binary_segmentation.frame_extractor import extract_frame
import supervision as sv
from utils.preprocess.contrast_func import improve_contrast
from utils.region_segmentation.extract_cat import calculate_pixel_size, calculate_region_width
from scipy.interpolate import make_interp_spline
import time
import jinja2
import pdfkit
import os
from datetime import datetime
parser = argparse.ArgumentParser(
                    prog='Coronary Angiography Analysis Pipeline',
                    description='This program identifies coronary regions according to the Syntax Score methodology, detects atherosclerotic plaques inside them, and provides health report',
                    epilog='Distributed as a part of Angio Solution Challenge 2023 application')

parser.add_argument('--frame_weights', '-fw', default="utils/binary_segmentation/weights/binary_sample.h5")
parser.add_argument('--region_weights', '-rw', default="utils/region_segmentation/weights/cat_yolov8x.pt")
parser.add_argument('--stenosis_weights', '-sw', default="utils/stenosis_segmentation/weights/best.pt")
parser.add_argument('--output', '-o', default="out_binary.png")
args = parser.parse_args()
analyzepressed = [0,0,0]
visited = [0,0,0]
while True:
    img = cv2.imread("imgs/solution.jpg")
    cv2.putText(img, "u - upload DICOM file", (55, 55), cv2.FONT_HERSHEY_SIMPLEX,0.6, (0,0,255), 1, cv2.LINE_AA)
    cv2.putText(img, "q - quit", (55, 75), cv2.FONT_HERSHEY_SIMPLEX,0.6, (0,0,255), 1, cv2.LINE_AA)

    cv2.imshow("ARCADE", img)
    if cv2.waitKey() == ord('u'):
        file = easygui.fileopenbox()
        dcm_data = dcmread(file)
        dcm = dcm_data.pixel_array
        best_mat, best_ids = extract_frame(dcm, args.frame_weights, 3)
        best_mat = np.expand_dims(best_mat, 3)
        print("Found 3 best frames: ", best_mat.shape)
        mat_frame = np.ones((512,1024,3), dtype=np.uint8)*255
        d3_orig = np.concatenate([best_mat, best_mat, best_mat], axis=3).astype(np.uint8)
        d3 = d3_orig.copy()
        print("Converted to 3-dimensional RGB", d3.shape)
        id = 0
        command=0
        reports = None
        report_part = np.ones((3,512,512,3))*255
        while True:
            if command == ord('f'): 
                id += 1
                command=0
                
            elif command == ord('a'):
                analyzepressed[id%3] = 1
                command = 0
                pred, names = segment_regions(args.region_weights, d3[id%3])
                masks = pred.masks.masks
                class_mat, mat = segment_stenosis(args.stenosis_weights, improve_contrast(d3[id%3]))
                msks = cv2.resize(masks.permute(1,2,0).numpy(), (512,512)) * cv2.resize(mat, (512,512))[:,:,np.newaxis]
                pos = np.where(msks.sum(axis=0).sum(axis=0)>0)  
                stenosis = np.sum(msks[:,:,pos[0]], axis=2)
                report = ""
                html_report = ""
                reports = []
                print(len(class_mat), "instances of stenosis found")
                sten_intercepts = np.zeros((len(class_mat), 512, 512))

                for i, (cl_name, rect, conf) in enumerate(class_mat):
                    sten_intercepts[i] = (cv2.resize(masks.permute(1,2,0).numpy(), (512,512)).sum(axis=2)>0) * cv2.resize(rect, (512,512))

                hold = np.ones((len(class_mat)))
                for i, sten1 in enumerate(sten_intercepts):
                    if sten1.sum() != 0:
                        for j, sten2 in enumerate(sten_intercepts[i+1:]):
                            if sten2.sum() != 0:
                                if sten1[(sten1>0) & (sten2>0)].shape[0]/sten1[sten1>0].shape[0]>0.5 or sten1[(sten1>0) & (sten2>0)].shape[0]/sten2[sten2>0].shape[0]>0.5:
                                    if class_mat[i][2] > class_mat[i+j+1][2]:
                                        hold[i+j+1] = 0
                                    else:
                                        hold[i] = 0
                            else:
                                hold[i+j+1] = 0
                    else:
                        hold[i] = 0
                pix_size = calculate_pixel_size(pred)
                for idx, mask_intercept in zip(np.where(hold>0)[0], sten_intercepts[hold>0]):
                    widths = calculate_region_width(mask_intercept, pix_size)[5:-5]
                    X_Y_Spline = make_interp_spline(np.arange(len(widths)), widths) 
                    X_ = np.linspace(np.arange(len(widths)).min(), np.arange(len(widths)).max(), 500) 
                    Y_ = X_Y_Spline(X_) 
                    plt.figure(time.time_ns())
                    plt.plot(X_, Y_)
                    plt.ylabel("width of stenosis")
                    plt.savefig(f"tmp/graph{id%3}_{idx}.png")

                path = os.path.abspath(".")
                for j, (cl_name, rect, conf) in enumerate(class_mat):
                    if hold[j] == 1:
                        for i, mask in enumerate(masks): 
                            if rect[(rect>0) & (mask.numpy()>0)].shape[0]>0:

                                report += f"Stenosis type {cl_name} found in {names[pred.boxes.cls[i].item()]} with confidence {conf*100:.2f}%. Area covered: {rect[(rect>0) & (mask.numpy()>0)].shape[0]/mask[mask.numpy()>0].numpy().shape[0]*100:.2f}%.\n"
                                reports+=[f"Stenosis type {cl_name} found in {names[pred.boxes.cls[i].item()]}", f"with confidence {conf*100:.2f}%.", f"Area covered: {rect[(rect>0) & (mask.numpy()>0)].shape[0]/mask[mask.numpy()>0].numpy().shape[0]*100:.2f}%."]
                                rectangle = np.array(np.where(best_mat[id%3, :, :, 0] * cv2.resize(rect, (512,512))>0)).T
                                w,h = rectangle[-1] - rectangle[0]
                                export_rect = best_mat[id%3, rectangle[0,0]-10:rectangle[0,0]+w+10, rectangle[0,1]-10:rectangle[0,1]+h+10]
                                cv2.imwrite(f"tmp/rect{id%3}_{j}.png", cv2.resize(export_rect, (400,400)))
                                
                                html_report += f"<li>Stenosis type {cl_name} found in {names[pred.boxes.cls[i].item()]} with confidence {conf*100:.2f}%. Area covered: {rect[(rect>0) & (mask.numpy()>0)].shape[0]/mask[mask.numpy()>0].numpy().shape[0]*100:.2f}%.\n<br>"
                                html_report += f'<img src="{path}\\tmp\\rect{id%3}_{j}.png"><img src="{path}\\tmp\\graph{id%3}_{j}.png" width="400" height="400"><br></li>\n'
                print(report)

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
                    scene=d3_orig[id%3].copy(), 
                    detections=detections, 
                    labels=labels,
                    skip_label=False
                ) 

                for j, bbxvessel in zip(labels,detections.xyxy):
                        box = best_mat[id%3, :, :, 0][int(bbxvessel[1]):int(bbxvessel[3]),int(bbxvessel[0]):int(bbxvessel[2])]
                        cv2.imwrite(f"tmp/bbxvessel{id%3}_{j.split()[0]}.png", cv2.resize(box, (int(box.shape[1]*4),int(box.shape[0]*4))))

                frame[sten_intercepts[hold>0].sum(axis=0)>0] = np.array([0,0,255])
                cv2.imwrite(f"tmp/d3{id%3}.png", frame)

                report_frame = np.ones((512,512,3))*255
                if reports:
                    offset = 95
                    for i, s in enumerate(reports):
                        offset+=20+((i)%3==0)*20
                        cv2.putText(report_frame, s, (12,offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
                report_part[id%3] = report_frame

                    

                d3[id%3] = frame
            mat_frame[0:512,0:512] = d3[id%3]
            mat_frame[:, 512:1024] = report_part[id%3]
            if(analyzepressed[id%3]):
                cv2.putText(mat_frame, "r - generate report", (mat_frame.shape[1]-350, 95), cv2.FONT_HERSHEY_SIMPLEX,0.6, (0,0,255), 1, cv2.LINE_AA)
                if command == ord('r'):
                    command = 0
                    if(os.path.exists(f'{path}/tmp/report{id%3}_Patient{dcm_data.PatientID}.pdf')):
                        os.system(f'{path}/tmp/report{id%3}_Patient{dcm_data.PatientID}.pdf')
                    else:
                        template_loader = jinja2.FileSystemLoader('./')
                        template_env = jinja2.Environment(loader=template_loader)
                        template = template_env.get_template('report.html')

                        context = {'name': dcm_data.PatientID, 'dob': dcm_data.PatientBirthDate, 'doe': f"{dcm_data.StudyDate[:4]} \
                                {dcm_data.StudyDate[4:6]} {dcm_data.StudyDate[6:]}, {dcm_data.StudyTime[:2]}:{dcm_data.StudyTime[2:4]}:{dcm_data.StudyTime[4:]}", 'date':time.strftime('%Y.%m.%d-%H:%M:%S')}
                        output_text = template.render(context)
                        output_text = output_text.split("\n")
                        output_text = output_text[:8] + [f'<img src="{path}\\tmp\\d3{id%3}.png" class="center"><br>'] + [html_report] + output_text[8:]
                        output_text = '''<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"/></head><style>p { font-size: 24px; } .center{
  display: block;
  margin-left: auto;
  margin-right: auto;
  width: 50%;
}
            li { font-size: 24px; }</style><body>''' + "\n".join(output_text) + "</body>"

                        config = pdfkit.configuration(wkhtmltopdf=r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe')
                        pdfkit.from_string(output_text, f'tmp/report{id%3}_Patient{dcm_data.PatientID}.pdf', configuration=config, css='style.css', options={"enable-local-file-access": ""})
                        os.system(f'{path}/tmp/report{id%3}_Patient{dcm_data.PatientID}.pdf')




            
            cv2.putText(mat_frame, "a - analyze", (mat_frame.shape[1]-350, 35), cv2.FONT_HERSHEY_SIMPLEX,0.6, (0,0,255), 1, cv2.LINE_AA)
            cv2.putText(mat_frame, "f - next frame", (mat_frame.shape[1]-350, 55), cv2.FONT_HERSHEY_SIMPLEX,0.6, (0,0,255), 1, cv2.LINE_AA)
            cv2.putText(mat_frame, "b - back", (mat_frame.shape[1]-350, 75), cv2.FONT_HERSHEY_SIMPLEX,0.6, (0,0,255), 1, cv2.LINE_AA)

           

            cv2.imshow("ARCADE", mat_frame)
            k = cv2.waitKey(1)
            if k == ord('b'):
                break
            if k != -1 and k != ord('q'):
                command = k
    elif cv2.waitKey() == ord('q'):
        break



from glob import glob
for i in glob("tmp/*"):
    os.remove(i)