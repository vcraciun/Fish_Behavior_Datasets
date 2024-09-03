import pandas as pd
import numpy as np
import os
import json
import math
import cv2
import time
import matplotlib.pyplot as plt
import statistics
from scipy.interpolate import interp1d
import pandas.plotting
import seaborn as sns
from sklearn.cluster import KMeans
import pandas.plotting
import itertools

FPS = 60

cardinal_dir = [
    ((0, 22.5), 1),
    ((22.5, 67.5), 2),
    ((67.5, 112.5), 3),
    ((112.5, 157.5), 4),
    ((157.5, 202.5), 5),
    ((202.5, 247.5), 6),
    ((247.5, 292.5), 7),
    ((292.5, 337.5), 8),
    ((337.5, 360), 1),
]

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

box_coords_2 = [
    ((10,75,130,225), 'A'),
    ((130,75,166,225), 'B1'),     #30% B
    ((166,75,250,225), 'B2'),     #70% B
    ((250,75,386,225), 'C'),      #C
    ((386,75,572,225), 'D1'),     #D+ 50% E
    ((572,75,626,225), 'D2'),     #50% E        
    ((250,225,386,475), 'E1'),    #F+G
    ((250,475,386,600), 'E2')     #H
]

def check(word, label, pos):
    if pos == 1:
        if word.startswith(label):
            return True
    if pos == 2:
        if not word.startswith(label) and not word.endswith(label) and label in word:
            return True
    if pos == 3:
        if word.endswith(label):
            return True
    if pos == 0:
        if label not in word:
            return True
    return False

def process_recording(fisier):
    df = pd.DataFrame()
    data = json.load(open(fisier, 'r'))
    frames = [el[0] for el in data]
    px = [el[1] for el in data]
    py = [el[2] for el in data]
    df["frame"] = frames
    df["x"] = px
    df["y"] = py
    df["ts"] = round(df["frame"] / FPS, 2)
    df["dst_px"] = round(np.sqrt((df["x"].shift(-1) - df["x"]) ** 2 + (df["y"].shift(-1) - df["y"]) ** 2), 2)
    df["dst_cm"] = round(np.sqrt((df["x"].shift(-1) - df["x"]) ** 2 + (df["y"].shift(-1) - df["y"]) ** 2) / 125.8, 2)
    df["speed_cms"] = round(df["dst_cm"].rolling(FPS).sum(), 2)
    df["angle"] = round((np.arctan2(df["y"] - df["y"].shift(-1), df["x"] - df["x"].shift(-1)) * 180 / np.pi + 360) % 360, 2)
    dir_labels = []
    for el in list(df["angle"]):
        for pos in cardinal_dir:
            if el >= pos[0][0] and el < pos[0][1]:
                dir_labels += [pos[1]]
                break
        else:
            dir_labels += [1]
    df["dir"] = dir_labels    
    b2_labels = []
    for cx, cy in list(zip(list(df["x"]), list(df["y"]))):
        found = False
        for pos in box_coords_2:
            (xt,yt,xb,yb), lbl = pos
            if cx >= xt and cx < xb and cy >= yt and cy < yb:
                b2_labels += [lbl]
                found = True
                break
        if not found:
            for pos in box_coords_2:
                (xt,yt,xb,yb), lbl = pos
                if cx >= xt and cx <= xb and cy >= yt and cy <= yb:
                    b2_labels += [lbl]
                    found = True
                    break
    df["box"] = b2_labels

    wall_hit_B1B2 = []
    wall_hit_B2C = []
    wall_hit_CD1 = []
    wall_hit_D1D2 = []
    wall_hit_CE1 = []
    wall_hit_E1E2 = []
    prev_coords = (0,0)
    prev_lbl = ""
    for cx, cy in list(zip(list(df["x"]), list(df["y"]))):
        if prev_coords[0] == 0 and prev_coords[1] == 0:
            wall_hit_B1B2 += [0]
            wall_hit_B2C += [0]
            wall_hit_CD1 += [0]
            wall_hit_D1D2 += [0]
            wall_hit_CE1 += [0]
            wall_hit_E1E2 += [0]
            prev_coords = (cx, cy)
            for pos in box_coords_2:
                (xt,yt,xb,yb), lbl = pos
                if cx >= xt and cx < xb and cy >= yt and cy < yb:
                    prev_lbl = lbl
                    break
        else:
            wall_hit_B1B2 += [0]
            wall_hit_B2C += [0]
            wall_hit_CD1 += [0]
            wall_hit_D1D2 += [0]
            wall_hit_CE1 += [0]
            wall_hit_E1E2 += [0]
            for pos in box_coords_2:                
                (xt,yt,xb,yb), lbl = pos                    
                if cx >= xt and cx < xb and cy >= yt and cy < yb:
                    if lbl != prev_lbl:                        
                        if prev_lbl == "B1" and lbl == "B2" or prev_lbl == "B2" and lbl == "B1":
                            wall_hit_B1B2[-1] = 1
                        if prev_lbl == "B2" and lbl == "C" or prev_lbl == "C" and lbl == "B2":
                            wall_hit_B2C[-1] = 1
                        if prev_lbl == "C" and lbl == "D1" or prev_lbl == "D1" and lbl == "C":
                            wall_hit_CD1[-1] = 1
                        if prev_lbl == "D1" and lbl == "D2" or prev_lbl == "D2" and lbl == "D1":
                            wall_hit_D1D2[-1] = 1
                        if prev_lbl == "C" and lbl == "E1" or prev_lbl == "E1" and lbl == "C":
                            wall_hit_CE1[-1] = 1
                        if prev_lbl == "E1" and lbl == "E2" or prev_lbl == "E2" and lbl == "E1":
                            wall_hit_E1E2[-1] = 1
                        prev_lbl = lbl
                    break
    df["wall_hit_B1B2"] = wall_hit_B1B2
    df["wall_hit_B2C"] = wall_hit_B2C
    df["wall_hit_CD1"] = wall_hit_CD1
    df["wall_hit_D1D2"] = wall_hit_D1D2
    df["wall_hit_CE1"] = wall_hit_CE1
    df["wall_hit_E1E2"] = wall_hit_E1E2

    sharp_turns = []
    for angle in list(df["angle"]):
        if (angle >= 0 and angle < 90) or (angle > 270 and angle <= 360):
            sharp_turns += [1]
        else:
            sharp_turns += [0]
    df["sharp_turns"] = sharp_turns
    #---------------------------------------------------------------------------------
    box = b2_labels
    
    d1 = list(df[df["box"] == "B1"]["dir"])
    d2 = list(df[df["box"] == "B2"]["dir"])
    d3 = list(df[df["box"] == "C"]["dir"])
    d4 = list(df[df["box"] == "D1"]["dir"])
    d5 = list(df[df["box"] == "D2"]["dir"])
    d6 = list(df[df["box"] == "E1"]["dir"])
    d7 = list(df[df["box"] == "E2"]["dir"])

    summary = [
        round(df[df["box"] == "B1"]["dst_cm"].sum(), 2),
        round(df[df["box"] == "B2"]["dst_cm"].sum(), 2),
        round(df[df["box"] == "C"]["dst_cm"].sum(), 2),
        round(df[df["box"] == "D1"]["dst_cm"].sum(), 2),
        round(df[df["box"] == "D2"]["dst_cm"].sum(), 2),
        round(df[df["box"] == "E1"]["dst_cm"].sum(), 2),
        round(df[df["box"] == "E2"]["dst_cm"].sum(), 2),

        round(box.count("B1") / 59.9, 2),
        round(box.count("B2") / 59.9, 2),
        round(box.count("C")  / 59.9, 2),
        round(box.count("D1") / 59.9, 2),
        round(box.count("D2") / 59.9, 2),
        round(box.count("E1") / 59.9, 2),
        round(box.count("E2") / 59.9, 2),

        round(df[df["box"] == "B1"]["sharp_turns"].sum() / 100, 2),
        round(df[df["box"] == "B2"]["sharp_turns"].sum() / 100, 2),
        round(df[df["box"] == "C"]["sharp_turns"].sum() / 100, 2),
        round(df[df["box"] == "D1"]["sharp_turns"].sum() / 100, 2),
        round(df[df["box"] == "D2"]["sharp_turns"].sum() / 100, 2),
        round(df[df["box"] == "E1"]["sharp_turns"].sum() / 100, 2),
        round(df[df["box"] == "E2"]["sharp_turns"].sum() / 100, 2),

        round(df["speed_cms"].mean(), 2),
        round(df["speed_cms"].max(), 2),

        df["wall_hit_B1B2"].sum(),
        df["wall_hit_B2C"].sum(),
        df["wall_hit_CD1"].sum(), 
        df["wall_hit_D1D2"].sum(),
        df["wall_hit_CE1"].sum(), 
        df["wall_hit_E1E2"].sum(),

        round(d1.count(1) / 100, 2),
        round(d1.count(2) / 100, 2),
        round(d1.count(3) / 100, 2),
        round(d1.count(4) / 100, 2),
        round(d1.count(5) / 100, 2),
        round(d1.count(6) / 100, 2),
        round(d1.count(7) / 100, 2),
        round(d1.count(8) / 100, 2),
        round(d2.count(1) / 100, 2),
        round(d2.count(2) / 100, 2),
        round(d2.count(3) / 100, 2),
        round(d2.count(4) / 100, 2),
        round(d2.count(5) / 100, 2),
        round(d2.count(6) / 100, 2),
        round(d2.count(7) / 100, 2),
        round(d2.count(8) / 100, 2),
        round(d3.count(1) / 100, 2),
        round(d3.count(2) / 100, 2),
        round(d3.count(3) / 100, 2),
        round(d3.count(4) / 100, 2),
        round(d3.count(5) / 100, 2),
        round(d3.count(6) / 100, 2),
        round(d3.count(7) / 100, 2),
        round(d3.count(8) / 100, 2),
        round(d4.count(1) / 100, 2),
        round(d4.count(2) / 100, 2),
        round(d4.count(3) / 100, 2),
        round(d4.count(4) / 100, 2),
        round(d4.count(5) / 100, 2),
        round(d4.count(6) / 100, 2),
        round(d4.count(7) / 100, 2),
        round(d4.count(8) / 100, 2),
        round(d5.count(1) / 100, 2),
        round(d5.count(2) / 100, 2),
        round(d5.count(3) / 100, 2),
        round(d5.count(4) / 100, 2),
        round(d5.count(5) / 100, 2),
        round(d5.count(6) / 100, 2),
        round(d5.count(7) / 100, 2),
        round(d5.count(8) / 100, 2),
        round(d6.count(1) / 100, 2),
        round(d6.count(2) / 100, 2),
        round(d6.count(3) / 100, 2),
        round(d6.count(4) / 100, 2),
        round(d6.count(5) / 100, 2),
        round(d6.count(6) / 100, 2),
        round(d6.count(7) / 100, 2),
        round(d6.count(8) / 100, 2),
        round(d7.count(1) / 100, 2),
        round(d7.count(2) / 100, 2),
        round(d7.count(3) / 100, 2),
        round(d7.count(4) / 100, 2),
        round(d7.count(5) / 100, 2),
        round(d7.count(6) / 100, 2),
        round(d7.count(7) / 100, 2),
        round(d7.count(8) / 100, 2),
    ]

    box_trans_timig = pd.DataFrame()
    box_trans_timig["ttB1"] = [round(summary[0] + summary[7]  * summary[23], 2)]
    box_trans_timig["ttB2"] = [round(summary[1] + summary[8]  * (summary[23] + summary[24]), 2)]
    box_trans_timig["ttC"]  = [round(summary[2] + summary[9]  * (summary[24] + summary[25] + summary[27]), 2)]
    box_trans_timig["ttD1"] = [round(summary[3] + summary[10] * (summary[25] + summary[26]), 2)]
    box_trans_timig["ttD2"] = [round(summary[4] + summary[11] * summary[26], 2)]
    box_trans_timig["ttE1"] = [round(summary[5] + summary[12] * (summary[27] + summary[29]), 2)]
    box_trans_timig["ttE2"] = [round(summary[6] + summary[13] * summary[29], 2)]

    key_words = []
    words = list(box_trans_timig.iloc[1:,:])
    for i in range(len(box_trans_timig.index)):
        line = list(box_trans_timig.iloc[i,:])
        pairs = list(zip(line, words))
        pairs.sort(key=lambda x: x[0], reverse=True)
        word = "".join([el[1][2:] for el in pairs if el[0] > 0])
        key_words += [word]
    box_trans_timig["word"] = key_words   

    set2 = [(1, "H"), (2, "M"), (3, "L"), (0, "A")]
    set1 = [('B1', "S"), ('C', "E"), ('D2', "e"), ('E2', 'A')]
    cartezian = list(itertools.product(set1, set2))
    behaviors = []
    extract_words = list(box_trans_timig["word"])
    for word in extract_words:
        behavior = []
        for beh in cartezian:
            if check(word, beh[0][0], beh[1][0]):
                behavior += [beh[1][1]+beh[0][1]]
        behaviors += [",".join(behavior)]
    box_trans_timig["behavior"] = behaviors

    return df, summary, box_trans_timig

def simulate(fisier, box_template):
    df = pd.read_csv(fisier)
    img = np.zeros((720,635,3), np.uint8)
    i = 0
    limit = 5 if box_template[1][1] == 'B' else 6    
    for box in box_template:
        (x1,y1,x2,y2), lbl = box
        cv2.rectangle(img, (x1, y1), (x2, y2), (255,255,255), 1)                                
        if i < limit:
            cv2.putText(img, lbl, ((x1 + x2) // 2 - 20, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)        
        else:
            cv2.putText(img, lbl, (x1 - 50, (y1 + y2) // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)        
        i += 1
    for i in range(1, len(df.index)):
        cv2.line(img, (df['x'][i-1], df["y"][i-1]), (df['x'][i], df["y"][i]), (128, 255, 128), 1) 
        cv2.imshow("Tracking", img)
        time.sleep(1/59.9)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

def process_jsons(subjects):    
    for subject in subjects:
        rules_features = pd.DataFrame()
        print(f"Processing {subject}: ", end="")
        path_to_src = os.path.join("traces", subject)
        path_to_dst = os.path.join("traces_csv", subject)
        zebra_traces = [os.path.join(path_to_src, file) for file in os.listdir(path_to_src)]        
        features_header = [
            "name",             
            "dst_cm_B1",        "dst_cm_B2",    "dst_cm_C",     "dst_cm_D1",    "dst_cm_D2",    "dst_cm_E1",    "dst_cm_E2",
            "time_B1",          "time_B2",      "time_C",       "time_D1",      "time_D2",      "time_E1",      "time_E2",
            "sharps_B1",        "sharps_B2",    "sharps_C",     "sharps_D1",    "sharps_D2",    "sharps_E1",    "sharps_E2", 
            "speed_cms_mean",   "speed_cms_max",
            "wall_hit_B1B2",    "wall_hit_B2C", "wall_hit_CD1", "wall_hit_D1D2","wall_hit_CE1", "wall_hit_E1E2",            
            "dirs_1_B1",        "dirs_2_B1",    "dirs_3_B1",    "dirs_4_B1",    "dirs_5_B1",    "dirs_6_B1",    "dirs_7_B1", "dirs_8_B1",
            "dirs_1_B2",        "dirs_2_B2",    "dirs_3_B2",    "dirs_4_B2",    "dirs_5_B2",    "dirs_6_B2",    "dirs_7_B2", "dirs_8_B2",
            "dirs_1_C",         "dirs_2_C",     "dirs_3_C",     "dirs_4_C",     "dirs_5_C",     "dirs_6_C",     "dirs_7_C",  "dirs_8_C",
            "dirs_1_D1",        "dirs_2_D1",    "dirs_3_D1",    "dirs_4_D1",    "dirs_5_D1",    "dirs_6_D1",    "dirs_7_D1", "dirs_8_D1",
            "dirs_1_D2",        "dirs_2_D2",    "dirs_3_D2",    "dirs_4_D2",    "dirs_5_D2",    "dirs_6_D2",    "dirs_7_D2", "dirs_8_D2",
            "dirs_1_E1",        "dirs_2_E1",    "dirs_3_E1",    "dirs_4_E1",    "dirs_5_E1",    "dirs_6_E1",    "dirs_7_E1", "dirs_8_E1",
            "dirs_1_E2",        "dirs_2_E2",    "dirs_3_E2",    "dirs_4_E2",    "dirs_5_E2",    "dirs_6_E2",    "dirs_7_E2", "dirs_8_E2"
        ]
        features_table = []
        for fisier in zebra_traces:
            _, fname = os.path.split(fisier)
            name, ext = os.path.splitext(fname)
            df, summary, rules_summary = process_recording(fisier)
            features_table += [[name[:-6]] + summary]
            df.to_csv(os.path.join(path_to_dst, name + ".csv"))            
            rules_features = pd.concat([rules_features, rules_summary], ignore_index=True)            
            print(".", end="")
        print(" ")
        rules_features.to_csv(f"{subject}_rules_features.csv")
        stats = pd.DataFrame(features_table, columns = features_header) 
        stats.to_csv(f"{subject}_all_features.csv")

def _KMeans(table, clusters, name):
    y = []    
    x = []
    for i in range(len(table.index)):
        y += list(table.iloc[i,1:-3])
        x += [i] * len(table.iloc[i,1:-3])

    inertia = []
    for i in range(1,15):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(list(zip(x, y)))
        inertia += [kmeans.inertia_]

    fig, [ax1, ax2] = plt.subplots(2, 1, figsize =(15, 6)) 
    fig.tight_layout(pad=1.5)
    ax1.plot(range(1,15), inertia, marker='o')
    ax1.grid()
    ax1.set_title('kmeans inertia')

    kmeans = KMeans(n_clusters=clusters)
    kmeans.fit(list(zip(x, y)))
    ax2.scatter(x, y, c=kmeans.labels_)
    ax2.grid()   
    ax2.set_title('ttB1, ttB2, ttC, ttD1, ttD2, ttE1, ttE2')
    fig.savefig(f"{name}_KMeans_{clusters}.png")

def plot_csv(df, name):
    ttb1 = list(df["ttB1"])
    ttb2 = list(df["ttB2"])
    ttc = list(df["ttC"])
    ttd1 = list(df["ttD1"])
    ttd2 = list(df["ttD2"])
    tte1 = list(df["ttE1"])
    tte2 = list(df["ttE2"])
    s_ttb1 = sorted(ttb1, reverse=True)
    s_ttb2 = sorted(ttb2, reverse=True)
    s_ttc = sorted(ttc, reverse=True)
    s_ttd1 = sorted(ttd1, reverse=True)
    s_ttd2 = sorted(ttd2, reverse=True)
    s_tte1 = sorted(tte1, reverse=True)
    s_tte2 = sorted(tte2, reverse=True)

    fig, ax = plt.subplots(figsize =(12, 6)) 
    plt.plot(df.index, s_ttb1)
    plt.plot(df.index, s_ttb2)
    plt.plot(df.index, s_ttc)
    plt.plot(df.index, s_ttd1)
    plt.plot(df.index, s_ttd2)
    plt.plot(df.index, s_tte1)
    plt.plot(df.index, s_tte2)
    plt.xlabel('Recording', fontweight ='bold',fontsize = 15)
    plt.ylabel('box_time * box_entries', fontweight ='bold',fontsize = 15)
    plt.legend(["B1","B2","C","D1","D2","E1","E2"])
    plt.grid(axis='y')
    plt.savefig(f"{name}_individual_trends.png", bbox_inches='tight', dpi=300)
    #------------------------------------------------------------------

    sum_all = [(elem[0], elem[1], elem[2], elem[3], elem[4], elem[5], elem[6], sum(elem)) for elem in list(zip(ttb1, ttb2, ttc, ttd1, ttd2, tte1, tte2))]
    sum_all.sort(key = lambda x: x[7], reverse=True)
    fig, ax = plt.subplots(1, 2, figsize =(15, 6)) 

    re1 = [el[0] for el in sum_all]
    re2 = [el[1] for el in sum_all]
    re3 = [el[2] for el in sum_all]
    re4 = [el[3] for el in sum_all]
    re5 = [el[4] for el in sum_all]
    re6 = [el[5] for el in sum_all]
    re7 = [el[6] for el in sum_all]
    
    re8 = [el[7] for el in sum_all]

    ax[0].scatter(df.index, re1)
    ax[0].scatter(df.index, re2)
    ax[0].scatter(df.index, re3)
    ax[0].scatter(df.index, re4)
    ax[0].scatter(df.index, re5)
    ax[0].scatter(df.index, re6)
    ax[0].scatter(df.index, re7)

    ax[1].plot(df.index, re8)

    ax[1].set_xlabel('Recording', fontweight ='bold',fontsize = 15)
    ax[0].set_xlabel('Recording', fontweight ='bold',fontsize = 15)
    ax[0].legend(["B1","B2","C","D1","D2","E1","E2"])
    ax[1].legend(["sum(B1,B2,C,D1,D2,E1,E2)"])
    ax[0].grid(axis='y')
    ax[1].grid(axis='y')
    plt.savefig(f"{name}_by_sum_all.png", bbox_inches='tight', dpi=300)
    #----------------------------------------------------------------------
    for i in range(3, 9):
        _KMeans(df, i, name)
    #----------------------------------------------------------------------
    df1 = df.drop(df.columns[[0,8,9,10]], axis=1)
    plt.gcf().subplots_adjust(wspace=1, hspace=1)
    df1.plot(kind='box', subplots=False, sharex=True, sharey=True, figsize=[12,6])
    plt.grid(axis='y')
    plt.savefig(f'{name}_box-plot.png')
    #---------------------------------------------------------------------
    behavior = list(df["behavior"])
    b_unice = list(set(behavior))
    freq = [behavior.count(crt) for crt in b_unice]
    paired = list(zip(freq, b_unice))
    paired.sort(key = lambda x : x[0], reverse=True)
    b_unice = [be[1] for be in paired]
    freq = [be[0] for be in paired]
    ind1 = np.arange(len(b_unice)) 
    fig, ax = plt.subplots(figsize =(15, 6)) 
    ax.bar(ind1, freq, color = 'r', label = "behaviors", edgecolor ='grey')
    for bar in ax.patches:
        value = round(bar.get_height())
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + bar.get_y() + 0.3, value, ha = 'center', color = 'k', size = 10)
    ax.set_xticks(ind1, b_unice)
    ax.grid(axis='y')
    plt.xlabel('Behaviors', fontweight ='bold',fontsize = 15)
    plt.ylabel('Frequency', fontweight ='bold',fontsize = 15)
    plt.xticks(rotation=90)
    plt.savefig(f"{name}_behavior_freq.png", bbox_inches='tight', dpi=300)    

if __name__ == "__main__":
    process_jsons(["zebra", "caras"])
    #simulate("traces_csv\\zebra\\Trial01_trace.csv", box_coords_2)

    #df = pd.read_csv("D:\\GIT\\FARM\\traces_csv\\caras\\_final_stat.csv")
    #plot_csv(df, "caras")

