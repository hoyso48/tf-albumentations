import tensorflow as tf
import numpy as np

_FILTER_BBOXES = True

def convert_to_yxhw(bboxes):
    box_yx = (bboxes[...,2:4] + bboxes[...,:2])/2
    box_hw = bboxes[...,2:4] - bboxes[...,:2]
    return tf.concat([box_yx,box_hw], -1)

def convert_to_corners(bboxes):
    box_tl = bboxes[...,:2] - bboxes[...,2:4]/2
    box_br = bboxes[...,:2] + bboxes[...,2:4]/2
    return tf.concat([box_tl,box_br], -1)

def swap_xy(bboxes):
    return tf.stack([bboxes[:, 1], bboxes[:, 0], bboxes[:, 3], bboxes[:, 2]], axis=-1)

# def clip_bboxes(bboxes, drop=_DROP_BBOXES, min_area=0):
#     bboxes = tf.clip_by_value(bboxes, 0., 1.)
#     if drop:
#         tf.debugging.assert_equal(tf.rank(bboxes), 2)
#         mask = box_area(bboxes)>min_area
#         bboxes = tf.boolean_mask(bboxes, mask, 0)
#     return bboxes

# def drop_bboxes_with_args(bboxes, *args, min_area=0):
#     bboxes = tf.clip_by_value(bboxes, 0., 1.)
#     tf.debugging.assert_equal(tf.rank(bboxes), 2)
#     mask = box_area(bboxes)>min_area
#     bboxes = tf.boolean_mask(bboxes, mask, 0)
#     for arg in args:
#       arg = tf.boolean_mask(arg, mask, 0)
#     return bboxes, args

def box_area(bboxes):
    wh = bboxes[...,2:]-bboxes[...,:2]
    return tf.reduce_prod(wh, -1)

def box_intersect(bboxes1, bboxes2):
    bboxes1=tf.cast(bboxes1, tf.float32)
    bboxes2=tf.cast(bboxes2, tf.float32)
    lt1 = bboxes1[...,:,None,:2]
    lt2 = bboxes2[...,None,:,:2]
    rb1 = bboxes1[...,:,None,2:]
    rb2 = bboxes2[...,None,:,2:]

    lt = tf.maximum(lt1, lt2)  # [...,N,M,2]
    rb = tf.minimum(rb1, rb2)  # [...,N,M,2]
    inter = tf.math.reduce_prod(tf.clip_by_value(rb - lt, 0, np.inf), -1) #[...,N,M]
    return inter


def coord2corners(bboxes, method='conservative'):
    '''
    input [(x1,x2,x3,x4),(y1,y2,y3,y4)]
    return: y1,x1,y2,x2 format bboxes
    (...N,4)
    '''
    if method == 'average':
        x1 = tf.reduce_mean(bboxes[...,0,:2],-1)
        x2 = tf.reduce_mean(bboxes[...,0,2:],-1)
        y1 = tf.reduce_mean(bboxes[...,1,::2],-1)
        y2 = tf.reduce_mean(bboxes[...,1,1::2],-1)
        bboxes = tf.stack([y1,x1,y2,x2],-1)
    elif method == 'conservative':
        x1y1 = tf.math.reduce_min(bboxes, -1) #(N,2)
        x2y2 = tf.math.reduce_max(bboxes, -1) #(N,2)
        bboxes = swap_xy(tf.concat([x1y1,x2y2], -1))
    else: raise ValueError
    return bboxes

def corner2coord(bboxes):
    '''
    input: y1,x1,y2,x2 format bboxes
    return [(x1,x1,x2,x2),(y1,y2,y1,y2)]
    (...N,2,4)
    '''
    y1,x1,y2,x2 = tf.unstack(bboxes,num=4,axis=-1)
    # return tf.stack([tf.stack([x1,y1],-1),tf.stack([x1,y2],-1),tf.stack([x2,y1],-1),tf.stack([x2,y2],-1)],-2)
    return tf.stack([tf.stack([x1,x1,x2,x2],-1),tf.stack([y1,y2,y1,y2],-1)],-2)


def get_rotation_matrix(radian):
    radian = -radian
    return tf.identity([[tf.math.cos(radian),-tf.math.sin(radian)],[tf.math.sin(radian),tf.math.cos(radian)]])

def get_shear_matrix(level_y, level_x):
    return tf.identity([[1.,-level_x],[-level_y,1.]])




def filter_objects_with_mask(objects, mask):
    for v in objects.keys():
        objects[v] = tf.boolean_mask(objects[v], mask, 0)
    return objects

def filter_objects_by_area(objects, min_area=0):
    objects['bbox'] = tf.clip_by_value(objects['bbox'], 0, 1)
    mask = box_area(objects['bbox'])>min_area
    return filter_objects_with_mask(objects, mask)

def flip_up_down_bboxes(objects):
    bboxes = objects.copy()
    y1,x1,y2,x2 = tf.unstack(bboxes['bbox'],num=4,axis=-1)
    bboxes['bbox'] = tf.stack([1-y2, x1, 1-y1, x2],axis=-1)
    return bboxes

def flip_left_right_bboxes(objects):
    bboxes = objects.copy()
    y1,x1,y2,x2 = tf.unstack(bboxes['bbox'],num=4,axis=-1)
    bboxes['bbox'] = tf.stack([y1, 1-x2, y2, 1-x1],axis=-1)
    return bboxes

# def scale_bboxes(objects, scale_y, scale_x, offset_y, offset_x, _filter=_FILTER_BBOXES):
#     bboxes = objects.copy()
#     bboxes['bbox'] = bboxes['bbox'] * [scale_y, scale_x, scale_y, scale_x] 
#     bboxes['bbox'] = bboxes['bbox'] + [offset_y, offset_x, offset_y, offset_x]
#     if _filter:
#         return filter_objects_by_area(bboxes, 0)
#     else: return bboxes
    
def scale_bboxes(objects, scale_y, scale_x, offset_y, offset_x, _filter=_FILTER_BBOXES):
    bboxes = objects.copy()
    bboxes['bbox'] = bboxes['bbox'] - [offset_y, offset_x, offset_y, offset_x]
    bboxes['bbox'] = bboxes['bbox'] / [scale_y, scale_x, scale_y, scale_x] 
    
    if _filter:
        return filter_objects_by_area(bboxes, 0)
    else: return bboxes
    
def translate_bboxes(objects, translate_y, translate_x, _filter=_FILTER_BBOXES):
    bboxes = objects.copy()
    bboxes['bbox'] = bboxes['bbox'] - [translate_y,translate_x,translate_y,translate_x]
    if _filter:
        return filter_objects_by_area(bboxes, 0)
    else: return bboxes

def rotate_bboxes(objects, radian, _filter=_FILTER_BBOXES):
    '''
    bbox = [y1,x1,y2,x2] format
    '''
    bboxes = objects.copy()
    bbox = bboxes['bbox'] - 0.5
    bbox = corner2coord(bbox) #(N,2,4)
    rot_mat = get_rotation_matrix(radian)
    rotated_box = tf.matmul(rot_mat, bbox)
    bboxes['bbox'] = coord2corners(rotated_box) + 0.5 #+ center
    if _filter:
        return filter_objects_by_area(bboxes, 0)
    else: return bboxes

def shear_bboxes(objects, level_y, level_x, _filter=_FILTER_BBOXES):
    '''
    bbox = [y1,x1,y2,x2] format
    '''
    bboxes = objects.copy()
    bbox = corner2coord(bboxes['bbox']) #(N,2,4)
    shear_mat = get_shear_matrix(level_y, level_x)
    rotated_box = tf.matmul(shear_mat, bbox)
    bboxes['bbox'] = coord2corners(rotated_box)
    if _filter:
        return filter_objects_by_area(bboxes, 0)
    else: return bboxes

def cutout_bboxes(objects, offset, size, area_filter=0.6, _filter=_FILTER_BBOXES):
    bboxes = objects.copy()
    if _filter:
        cutout_box = tf.identity([[offset[0], offset[1], offset[0]+size[0], offset[1]+size[1]]])
        inter = box_intersect(bboxes['bbox'], cutout_box)[...,0] #(N,1)
        mask = inter/box_area(bboxes['bbox']) < area_filter
        return filter_objects_with_mask(bboxes, mask)
    else:
        return bboxes
