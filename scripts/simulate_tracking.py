import cv2
import os
import json
import numpy as np
import sys
import datetime
import time

FPS = 59.9

#Simple layout based on aquarium mechanical layout
box_coords_1 = [
    ((10,75,130,225), 'A'),
    ((130,75,250,225), 'B'),
    ((250,75,386,225), 'C'),
    ((386,75,506,225), 'D'),
    ((506,75,626,225), 'E'),
    ((250,225,386,350), 'F'),
    ((250,350,386,475), 'G'),
    ((250,475,386,600), 'H')
]

#Our layout based on fish interest areas
box_coords_2 = [
    ((10,75,130,225), 'A'),
    ((130,75,166,225), 'B1'),     #30% B
    ((166,75,250,225), 'B2'),     #70% B
    ((250,75,386,225), 'C'),      #C
    ((386,75,572,225), 'D1'),     #D, 50% E
    ((572,75,626,225), 'D2'),     #50% E        
    ((250,225,386,475), 'E1'),    #F+G
    ((250,475,386,600), 'E2')     #H
]

#Apply the layout
def draw_layout(img, layout, limit, frame = None):
    #draw the limits of the aquarium surface
    cv2.rectangle(img, (5, 5), (img.shape[0]-90, img.shape[1]+80), (255,255,255), 1)
    #draw selected layout
    for i, ((x1,y1,x2,y2), lbl) in enumerate(layout):
        cv2.rectangle(img, (x1, y1), (x2, y2), (255,255,255), 1)                                
        if i < limit:
            cv2.putText(img, lbl, ((x1 + x2) // 2 - 20, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)        
        else:
            cv2.putText(img, lbl, (x1 - 50, (y1 + y2) // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)                 
    #Show passing time
    if frame is not None:
        cv2.rectangle(img, (250, 610), (400,660), (0,0,0), -1)
        passed = datetime.datetime.strftime(datetime.datetime.fromtimestamp(frame/FPS), "%M:%S")
        cv2.putText(img, passed, (270,650), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
    #highlight and label the moving fish subject place
    cv2.rectangle(img, (layout[0][0][0]+1, layout[0][0][1]+1), (layout[0][0][2]-1, layout[0][0][3]-1), (128,128,128), -1)
    cv2.putText(img, "moving", (layout[0][0][0]+20, layout[0][0][1]+50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, "fish", (layout[0][0][0]+20, layout[0][0][1]+80), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, "subjects", (layout[0][0][0]+20, layout[0][0][1]+110), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, cv2.LINE_AA)

#Execute simulation with specified FPS (FPS here may be much higher compared to initial recording)
def simulate(json_path, layout, show_output):
    #load data set with fish tracking data
    data = json.load(open(json_path, 'r'))
    #fish tracking line
    img = np.zeros((720,635,3), np.uint8)
    #moving fish
    fish_move = np.zeros((720,635,3), np.uint8)
    #heatmap highlighting the most visited areas
    heat_map = np.zeros([img.shape[0], img.shape[1]])
    limit = 5 if layout[1][1] == 'B' else 6
    _, fname = os.path.split(json_path)
    name, _ = os.path.splitext(fname)
    name = name.split('_')[0]
    start_time = time.time()
    for i in range(len(data)):
        #plot fish tracking
        if i == 0:
            cv2.line(img, (data[i][1], data[i][2]), (data[i][1], data[i][2]), (128, 255, 128), 1) 
        else:
            cv2.line(img, (data[i-1][1], data[i-1][2]), (data[i][1], data[i][2]), (128, 255, 128), 1) 
        
        #plot the moving fish
        if i > 0:
            cv2.circle(fish_move, (data[i-1][1], data[i-2][2]), 20, (0, 0, 0), -1)
        cv2.circle(fish_move, (data[i][1], data[i][2]), 10, (0, 255, 0), -1)
        
        #compute heatmap
        heat_map[np.all(fish_move == [0, 255, 0], 2)] += 1
        heat_map[heat_map < 0] = 0
        heat_map[heat_map > 255] = 255
        color_map = cv2.applyColorMap(heat_map.astype("uint8"), cv2.COLORMAP_JET)

        #redraw layout for all images
        draw_layout(img, layout, limit, i)
        draw_layout(color_map, layout, limit)
        draw_layout(fish_move, layout, limit)

        #show images
        if show_output:
            final_img = cv2.hconcat([fish_move, img, color_map])
            cv2.imshow(f"Tracking {name}", final_img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    processing_time = time.time() - start_time
    recording_time = data[-1][0] * (1/FPS)

    #write the resulting images
    cv2.imwrite(f"{name}_tracking.png", img)
    cv2.imwrite(f"{name}_heatmap.png", color_map)

    return processing_time, recording_time

def process_all_in_path(path_to_jsons, layout):
    traces = [
        os.path.join(path_to_jsons, trace)
        for trace in os.listdir(path_to_jsons)
        if trace.endswith(".json")
    ]

    recording_times = []
    processing_times = []
    for trace in traces:
        proc, rec = simulate(trace, layout, False)
        processing_times += [proc]
        recording_times += [rec]
    performance = list(zip(recording_times, processing_times))
    
    json.dump(performance, open("performance.json", 'w'))

if __name__ == "__main__":
    if len(sys.argv) == 1 or len(sys.argv) > 3:
        print("No tracking json file specified !!!")
    elif len(sys.argv) == 2:
        simulate(sys.argv[1], box_coords_2, True)
    elif len(sys.argv) == 3:
        process_all_in_path(sys.argv[1], box_coords_2)
