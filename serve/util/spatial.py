import numpy as np 

def center_offset(box1, box2, im_wh):
    c1 = [(box1[2]+box1[0])/2, (box1[3]+box1[1])/2]
    c2 = [(box2[2]+box2[0])/2, (box2[3]+box2[1])/2]
    offset = np.array(c1)-np.array(c2)/np.array(im_wh)
    return offset

def box_with_respect_to_img(box, im_wh):
    '''
        To get [x1/W, y1/H, x2/W, y2/H, A_box/A_img]
    '''
    feats = [box[0]/(im_wh[0]+ 1e-6), box[1]/(im_wh[1]+ 1e-6), box[2]/(im_wh[0]+ 1e-6), box[3]/(im_wh[1]+ 1e-6)]
    box_area = (box[2]-box[0])*(box[3]-box[1])
    img_area = im_wh[0]*im_wh[1]
    feats +=[ box_area/(img_area+ 1e-6) ]
    return feats

def box1_with_respect_to_box2(box1, box2):
    feats = [ (box1[0]-box2[0])/(box2[2]-box2[0]+1e-6),
              (box1[1]-box2[1])/(box2[3]-box2[1]+ 1e-6),
              np.log((box1[2]-box1[0])/(box2[2]-box2[0]+ 1e-6)),
              np.log((box1[3]-box1[1])/(box2[3]-box2[1]+ 1e-6))   
            ]
    return feats

def calculate_spatial_feats(det_boxes, im_wh):
    spatial_feats = []
    for i in range(det_boxes.shape[0]):
        for j in range(det_boxes.shape[0]):
            if j == i:
                continue
            else:
                single_feat = []
                box1_wrt_img = box_with_respect_to_img(det_boxes[i], im_wh)
                box2_wrt_img = box_with_respect_to_img(det_boxes[j], im_wh)
                box1_wrt_box2 = box1_with_respect_to_box2(det_boxes[i], det_boxes[j])
                offset = center_offset(det_boxes[i], det_boxes[j], im_wh)
                single_feat = single_feat + box1_wrt_img + box2_wrt_img + box1_wrt_box2 + offset.tolist()
                # ipdb.set_trace()
                spatial_feats.append(single_feat)
    spatial_feats = np.array(spatial_feats)
    return spatial_feats