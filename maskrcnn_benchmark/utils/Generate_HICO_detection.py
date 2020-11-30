import pickle
import numpy as np
import scipy.io as sio
import os

# def save_HICO(HICO, HICO_dir, classid, begin, finish):


#     all_boxes = []
#     for i in range(finish - begin + 1):
#         total = []
#         score = []
#         for key, value in HICO.iteritems():
#             for element in value:
#                 if element[2] == classid:
#                     temp = []
#                     temp.append(element[0].tolist())  # Human box
#                     temp.append(element[1].tolist())  # Object box
#                     temp.append(int(key))             # image id
#                     temp.append(int(i))               # action id (0-599)
#                     temp.append(element[3][begin - 1 + i] * element[4] * element[5])
#                     total.append(temp)
#                     score.append(element[3][begin - 1 + i] * element[4] * element[5])

#         idx = np.argsort(score, axis=0)[::-1]
#         for i_idx in range(min(len(idx),19999)):
#             all_boxes.append(total[idx[i_idx][0]])
#     savefile = HICO_dir + 'detections_' + str(classid).zfill(2) + '.mat'
#     sio.savemat(savefile, {'all_boxes':all_boxes})


# def Generate_HICO_detection(output_dir, HICO_dir):


#     if not os.path.exists(HICO_dir):
#         os.makedirs(HICO_dir)

#     # Remove previous snapshots
#     filelist = [ f for f in os.listdir(HICO_dir)]
#     for f in filelist:
#         os.remove(os.path.join(HICO_dir, f))


#     HICO = pickle.load( open( output_dir, "rb" ) )

#     save_HICO(HICO, HICO_dir,  1 ,161, 170) # 1 person
#     save_HICO(HICO, HICO_dir,  2 ,11,  24 ) # 2 bicycle
#     save_HICO(HICO, HICO_dir,  3 ,66,  76 ) # 3 car
#     save_HICO(HICO, HICO_dir,  4 ,147, 160) # 4 motorcycle
#     save_HICO(HICO, HICO_dir,  5 ,1,   10 ) # 5 airplane
#     save_HICO(HICO, HICO_dir,  6 ,55,  65 ) # 6 bus
#     save_HICO(HICO, HICO_dir,  7 ,187, 194) # 7 train
#     save_HICO(HICO, HICO_dir,  8 ,568, 576) # 8 truck
#     save_HICO(HICO, HICO_dir,  9 ,32,  46 ) # 9 boat
#     save_HICO(HICO, HICO_dir,  10,563, 567) # 10 traffic light
#     save_HICO(HICO, HICO_dir,  11,326,330) # 11 fire_hydrant
#     save_HICO(HICO, HICO_dir,  12,503,506) # 12 stop_sign
#     save_HICO(HICO, HICO_dir,  13,415,418) # 13 parking_meter
#     save_HICO(HICO, HICO_dir,  14,244,247) # 14 bench
#     save_HICO(HICO, HICO_dir,  15,25,  31) # 15 bird
#     save_HICO(HICO, HICO_dir,  16,77,  86) # 16 cat
#     save_HICO(HICO, HICO_dir,  17,112,129) # 17 dog
#     save_HICO(HICO, HICO_dir,  18,130,146) # 18 horse
#     save_HICO(HICO, HICO_dir,  19,175,186) # 19 sheep
#     save_HICO(HICO, HICO_dir,  20,97,107)  # 20 cow
#     save_HICO(HICO, HICO_dir,  21,314,325) # 21 elephant
#     save_HICO(HICO, HICO_dir,  22,236,239) # 22 bear
#     save_HICO(HICO, HICO_dir,  23,596,600) # 23 zebra
#     save_HICO(HICO, HICO_dir,  24,343,348) # 24 giraffe
#     save_HICO(HICO, HICO_dir,  25,209,214) # 25 backpack
#     save_HICO(HICO, HICO_dir,  26,577,584) # 26 umbrella
#     save_HICO(HICO, HICO_dir,  27,353,356) # 27 handbag
#     save_HICO(HICO, HICO_dir,  28,539,546) # 28 tie
#     save_HICO(HICO, HICO_dir,  29,507,516) # 29 suitcase
#     save_HICO(HICO, HICO_dir,  30,337,342) # 30 Frisbee
#     save_HICO(HICO, HICO_dir,  31,464,474) # 31 skis
#     save_HICO(HICO, HICO_dir,  32,475,483) # 32 snowboard
#     save_HICO(HICO, HICO_dir,  33,489,502) # 33 sports_ball
#     save_HICO(HICO, HICO_dir,  34,369,376) # 34 kite
#     save_HICO(HICO, HICO_dir,  35,225,232) # 35 baseball_bat
#     save_HICO(HICO, HICO_dir,  36,233,235) # 36 baseball_glove
#     save_HICO(HICO, HICO_dir,  37,454,463) # 37 skateboard
#     save_HICO(HICO, HICO_dir,  38,517,528) # 38 surfboard
#     save_HICO(HICO, HICO_dir,  39,534,538) # 39 tennis_racket
#     save_HICO(HICO, HICO_dir,  40,47,54)   # 40 bottle
#     save_HICO(HICO, HICO_dir,  41,589,595) # 41 wine_glass
#     save_HICO(HICO, HICO_dir,  42,296,305) # 42 cup
#     save_HICO(HICO, HICO_dir,  43,331,336) # 43 fork
#     save_HICO(HICO, HICO_dir,  44,377,383) # 44 knife
#     save_HICO(HICO, HICO_dir,  45,484,488) # 45 spoon
#     save_HICO(HICO, HICO_dir,  46,253,257) # 46 bowl
#     save_HICO(HICO, HICO_dir,  47,215,224) # 47 banana
#     save_HICO(HICO, HICO_dir,  48,199,208) # 48 apple
#     save_HICO(HICO, HICO_dir,  49,439,445) # 49 sandwich
#     save_HICO(HICO, HICO_dir,  50,398,407) # 50 orange
#     save_HICO(HICO, HICO_dir,  51,258,264) # 51 broccoli
#     save_HICO(HICO, HICO_dir,  52,274,283) # 52 carrot
#     save_HICO(HICO, HICO_dir,  53,357,363) # 53 hot_dog
#     save_HICO(HICO, HICO_dir,  54,419,429) # 54 pizza
#     save_HICO(HICO, HICO_dir,  55,306,313) # 55 donut
#     save_HICO(HICO, HICO_dir,  56,265,273) # 56 cake
#     save_HICO(HICO, HICO_dir,  57,87,92)   # 57 chair
#     save_HICO(HICO, HICO_dir,  58,93,96)   # 58 couch
#     save_HICO(HICO, HICO_dir,  59,171,174) # 59 potted_plant
#     save_HICO(HICO, HICO_dir,  60,240,243) #60 bed
#     save_HICO(HICO, HICO_dir,  61,108,111) #61 dining_table
#     save_HICO(HICO, HICO_dir,  62,551,558) #62 toilet
#     save_HICO(HICO, HICO_dir,  63,195,198) #63 TV
#     save_HICO(HICO, HICO_dir,  64,384,389) #64 laptop
#     save_HICO(HICO, HICO_dir,  65,394,397) #65 mouse
#     save_HICO(HICO, HICO_dir,  66,435,438) #66 remote
#     save_HICO(HICO, HICO_dir,  67,364,368) #67 keyboard
#     save_HICO(HICO, HICO_dir,  68,284,290) #68 cell_phone
#     save_HICO(HICO, HICO_dir,  69,390,393) #69 microwave
#     save_HICO(HICO, HICO_dir,  70,408,414) #70 oven
#     save_HICO(HICO, HICO_dir,  71,547,550) #71 toaster
#     save_HICO(HICO, HICO_dir,  72,450,453) #72 sink
#     save_HICO(HICO, HICO_dir,  73,430,434) #73 refrigerator
#     save_HICO(HICO, HICO_dir,  74,248,252) #74 book
#     save_HICO(HICO, HICO_dir,  75,291,295) #75 clock
#     save_HICO(HICO, HICO_dir,  76,585,588) #76 vase
#     save_HICO(HICO, HICO_dir,  77,446,449) #77 scissors
#     save_HICO(HICO, HICO_dir,  78,529,533) #78 teddy_bear
#     save_HICO(HICO, HICO_dir,  79,349,352) #79 hair_drier
#     save_HICO(HICO, HICO_dir,  80,559,562) #80 toothbrush



def Generate_HICO_detection(detection, HICO_dir):

    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    DATA_DIR = os.path.abspath(os.path.join(ROOT_DIR, 'Data'))

    if not os.path.exists(HICO_dir):
        os.makedirs(HICO_dir)

    # Remove previous snapshots
    filelist = [ f for f in os.listdir(HICO_dir)]
    for f in filelist:
        os.remove(os.path.join(HICO_dir, f))

    HICO = pickle.load( open( detection, "rb" ) )

    COCO_obj = {}
    CLASSES = ('_background_', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus','train', 'truck', 'boat', 'traffic_light', 'fire_hydrant', 'stop_sign', 'parking_meter','bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack','umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports_ball', 'kite','baseball_bat', 'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket', 'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl','banana', 'apple', 'sandwich', 'orange', 'broccoli','carrot', 'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted_plant', 'bed', 'dining_table','toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave', 'oven','toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy_bear', 'hair_drier','toothbrush')

    for idx, value in enumerate(CLASSES):
        print(idx, value)
        COCO_obj[value] = idx

    HICO_vb = {}
    with open (os.path.join(DATA_DIR, 'hico_list_vb.txt'), 'rt') as in_file:
        for line in in_file:
            if line.split()[0].isdigit():

                print(int(line.split()[0]) - 1,line.split()[1] )
                HICO_vb[line.split()[1]] = int(line.split()[0]) - 1


    HICO_HOI = {}
    HICO_obj = {}
    with open (os.path.join(DATA_DIR, 'hico_list_hoi.txt'), 'rt') as in_file:
        for line in in_file:
            if line.split()[0].isdigit():
                HICO_HOI[int(line.split()[0]) - 1] = HICO_vb[line.split()[2]]
                HICO_obj[int(line.split()[0]) - 1] = COCO_obj[line.split()[1]]
                print(int(line.split()[0]) - 1,line.split()[1], line.split()[2])


    HICO_mask = {}

    for HOI_idx, Obj_idx in HICO_obj.items():
        vb_idx = HICO_HOI[HOI_idx]
        if Obj_idx in HICO_mask:
            HICO_mask[Obj_idx].append(vb_idx)
        else:
            HICO_mask[Obj_idx] = [vb_idx]

    HICO_map = {}

    for Obj_idx, vb_list in HICO_mask.items():

        vb_map = {}
        for vb in vb_list:
            for idx in range(600):
                if HICO_obj[idx] == Obj_idx and HICO_HOI[idx] == vb:
                    vb_map[vb] = idx
                    break
        HICO_map[Obj_idx] = vb_map

    # This is for the Journal
    for obj_idx in range(1, 81):
        print('~~~~~' + str(obj_idx) + '~~~~')
        all_boxes = []
        for verb_idx in HICO_map[obj_idx].keys():
            print(HICO_map[obj_idx][verb_idx])
            total = []
            score = []
            for image_id, image_detection in HICO.items():
                for inst_detection in image_detection:
                    # if inst_detection[2][0] == obj_idx:
                    if inst_detection[2] == obj_idx:
                        temp = []
                        temp.append(inst_detection[0].tolist())  # Human box
                        temp.append(inst_detection[1].tolist())  # Object box
                        temp.append(int(image_id))        # image id
                        temp.append(int(HICO_map[obj_idx][verb_idx]))     # action id (0-599)
                        temp.append(inst_detection[3][verb_idx] * inst_detection[4] * inst_detection[5])
                        total.append(temp)
                        score.append(inst_detection[3][verb_idx] * inst_detection[4] * inst_detection[5])
            idx = np.argsort(score, axis=0)[::-1]
            # for i_idx in range(min(len(idx), 19999)):
            for i_idx in range(len(idx)):
                all_boxes.append(total[idx[i_idx][0]])
        savefile = os.path.join(HICO_dir, 'detections_' + str(obj_idx).zfill(2) + '.mat')
        sio.savemat(savefile, {'all_boxes':all_boxes})



    # # This is for the Journal (VCOCO style)
    # for obj_idx in range(1, 81):
    #     print('~~~~~' + str(obj_idx) + '~~~~')
    #     all_boxes = []

    #     total = []
    #     score = []
    #     for inst_detection in HICO[obj_idx]:
    #         if inst_detection[4] in HICO_map[obj_idx].keys():
    #             temp = []
    #             temp.append(inst_detection[0].tolist())  # Human box
    #             temp.append(inst_detection[1].tolist())  # Object box
    #             temp.append(int(inst_detection[2]))      # image id
    #             temp.append(int(HICO_map[obj_idx][inst_detection[4]]))  # action id (0-599)
    #             temp.append(inst_detection[3])
    #             total.append(temp)
    #             score.append(inst_detection[3])

    #     idx = np.argsort(score, axis=0)[::-1]
    #     for i_idx in range(len(idx)):
    #         all_boxes.append(total[idx[i_idx]])
    #     savefile = HICO_dir + 'detections_' + str(obj_idx).zfill(2) + '.mat'
    #     sio.savemat(savefile, {'all_boxes':all_boxes})
