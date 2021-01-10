import cv2
import maxflow
import numpy as np
import sys
import os
import json
from glob import glob
import argparse

# Initialise sift descriptor for feature detection
sift=cv2.xfeatures2d.SIFT_create()
def gauss_pyramid(image,level):

    # calculated the gaussian pyramid of size level passed as argument   
    img_copy = image.copy()  
    gauss_pyramid = [img_copy] 
    for i in range(level):
        img_copy = cv2.pyrDown(gauss_pyramid[i])
        gauss_pyramid.append(img_copy)           
    return gauss_pyramid

def laplacian_pyramid(gauss_prev,level): 

    # calculated the laplacian pyramid of size level passed as argument for the gauss pyramid 
    lap_pyramid = [gauss_prev[level-1]]
    for i in range(level-1,0,-1):
        size = (gauss_prev[i-1].shape[1], gauss_prev[i-1].shape[0]) 
        L_temp = cv2.pyrUp(gauss_prev[i], dstsize = size)           
        L_subtract = cv2.subtract(gauss_prev[i-1],L_temp)           
        lap_pyramid.append(L_subtract)                                
    return lap_pyramid

def combine_pyramid(gauss_mask, lap_src, lap_target,level):

    # combined the result according to the mask calulated at each level
    combined_result = []
    for i in range(level):
        x, y = gauss_mask[level-1-i].shape
        combined_img = lap_src[i]
        threshold = 210
        for count_x in range (1, x) :
            for count_y in range (1, y):
                if gauss_mask[level-1-i][count_x-1, count_y-1] > threshold:
                    combined_img[count_x, count_y] = lap_target[i][count_x, count_y]
        
        combined_result.append(combined_img)

    recon_img = combined_result[0] 
    for i in range(1,level):
        size = (combined_result[i].shape[1], combined_result[i].shape[0]) 
        recon_img = cv2.pyrUp(recon_img, dstsize = size)                  
        recon_img = cv2.add(recon_img, combined_result[i])
    return recon_img

def Pyramid_blend(target,src,mask, level):   

    # created the gaussian pyramids for each of source,target and mask.      
    gauss_src = gauss_pyramid(src, level) 
    gauss_target = gauss_pyramid(target, level)         
    gmask = gauss_pyramid(mask, level)           
    
    # created the laplacian pyramids for each of source,target and mask.  
    lap_src = laplacian_pyramid(gauss_src, level)
    lap_target = laplacian_pyramid(gauss_target, level)        
    lap_mask = laplacian_pyramid(gmask, level)
    
    # combined the result
    result = combine_pyramid(gmask, lap_src, lap_target, level)
    return result

def compute_weights(src, sink):
    
    # initialise the edge-weights
    edge_weights = np.zeros((src.shape[0], src.shape[1], 2))

    # shifted the matrix left or up for vectorised calculation of weights.
    src_left_shifted = np.roll(src, -1, axis=1)
    sink_left_shifted = np.roll(sink, -1, axis=1)
    src_up_shifted = np.roll(src, -1, axis=0)
    sink_up_shifted = np.roll(sink, -1, axis=0)
    eps = 1e-10

    # calculated the wait for both left and up direction and returned it.
    weight = np.sum(np.square(src - sink, dtype=np.float) +
                    np.square(src_left_shifted - sink_left_shifted, 
                    dtype=np.float),
                    axis=2)
    norm_factor = 1
    edge_weights[:, :, 0] = weight / (norm_factor + eps)
    weight = np.sum(np.square(src - sink, dtype=np.float) +
                    np.square(src_up_shifted - sink_up_shifted,
                    dtype=np.float),
                    axis=2)
    edge_weights[:, :, 1] = weight / (norm_factor + eps)
    return edge_weights

def find_cut(src, sink,t_val):

    height = src.shape[0]
    width = src.shape[1]

    # initialise the graph for maxflow computation
    graph = maxflow.Graph[float]()
    node_ids = graph.add_grid_nodes((height,width))

    # Computed the edge weights using the function implemented.
    edge_weights = compute_weights(src, sink)

    # added the edges in the graph of weights computed.
    for row_idx in range(height):
        for col_idx in range(width):
            if col_idx + 1 < width:
                weight = edge_weights[row_idx, col_idx, 0]
                graph.add_edge(node_ids[row_idx][col_idx],node_ids[row_idx][col_idx + 1],weight,weight)

            if row_idx + 1 < height:
                weight = edge_weights[row_idx, col_idx, 1]
                graph.add_edge(node_ids[row_idx][col_idx],node_ids[row_idx + 1][col_idx],weight,weight)

            # added the terminal edges connected to source and sink
            x = 0
            y = 0
            if t_val[1]!=0:
                x = 1
            if t_val[0]!=0:
                y = 1
            if row_idx==x*(height-1) or col_idx==y*(width-1):
                graph.add_tedge(node_ids[row_idx][col_idx], 0, np.inf)
                graph.add_tedge(node_ids[(height-1)-row_idx][(width-1)-col_idx], np.inf, 0)

    # find the flow using the inbuilt function and segments present in the graph is returned
    flow = graph.maxflow()
    segments = graph.get_grid_segments(node_ids)
    return segments

def registration(img1,img2):

    # detected the features in both images
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # created a matcher for matching the descriptors and found the homography according to best points matched.
    matcher = cv2.BFMatcher()
    raw_matches = matcher.knnMatch(des1, des2, k=2)
    good_points = []
    good_matches=[]
    for m1, m2 in raw_matches:
        if m1.distance < 0.85 * m2.distance:
            good_points.append((m1.trainIdx, m1.queryIdx))
            good_matches.append([m1])
    if len(good_points) > 10:
        image1_kp = np.float32(
            [kp1[i].pt for (_, i) in good_points])
        image2_kp = np.float32(
            [kp2[i].pt for (i, _) in good_points])
        H1, status = cv2.findHomography(image2_kp, image1_kp, cv2.RANSAC,5.0)
    return H1

def process_image(img1,img2):

    # calculated the width and height of image
    height1, width1 = img1.shape[:2]
    height2, width2 = img2.shape[:2]

    # Find the homography of one image with respect to other
    H1 = registration(img1,img2)

    # Applied the homography on 4 points to get the size and relative positioning of images.
    pts1 = np.array([[0, 0], [0, height1], [width1, height1], [width1, 0]],np.float32).reshape(-1, 1, 2)
    pts2 = np.array([[0, 0], [0, height2], [width2, height2], [width2, 0]],np.float32).reshape(-1, 1, 2)
    pts2_ = cv2.perspectiveTransform(pts2, H1)
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel()-0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel()+0.5)
    t_val = [-xmin, -ymin]

    #Calculated the translational coefficient
    h_translation = np.array([[1, 0, t_val[0]], [0, 1, t_val[1]], [0, 0, 1]])

    #Aligned one image2 with respect to the image1.
    panorama2 = cv2.warpPerspective(img2, h_translation.dot(H1), (xmax-xmin, ymax-ymin))

    #Translated the image1 with respect to the image2 so that it covers both the cases when image1 is left image or right image.
    indices1 = np.argwhere(np.sum(img1, axis=2)>=0)
    indices = indices1 + [t_val[1], t_val[0]]
    panorama1 = np.zeros_like(panorama2)
    panorama1[tuple(zip(*indices))] = img1[tuple(zip(*indices1))]

    # Converted both the image panorama into grayscale images for faster computation
    panorama1gray = cv2.cvtColor(panorama1, cv2.COLOR_RGB2GRAY).reshape((panorama1.shape[0],panorama1.shape[1],1))
    panorama2gray = cv2.cvtColor(panorama2, cv2.COLOR_RGB2GRAY).reshape((panorama2.shape[0],panorama2.shape[1],1))
    
    # found the intersection area of both the images
    final = np.where(np.logical_and(panorama1gray,panorama2gray),1,0)
    cnt = np.count_nonzero(final)
    p2 = np.multiply(panorama2,final)
    p1 = np.multiply(panorama1,final)
    diff = np.sum(np.dot(p2,np.array([0.114,0.587,0.299]))) - np.sum(np.dot(p1,np.array([0.114,0.587,0.299])))
    diff = diff/cnt
    panorama2 = np.clip(panorama2-diff,0,255)
    panorama2 = np.array(panorama2,np.uint8)
    panorama2gray = cv2.cvtColor(panorama2, cv2.COLOR_RGB2GRAY).reshape((panorama2.shape[0],panorama2.shape[1],1))
    rows, cols = np.where(final[:,:,0]!= 0)
    min_row, max_row = min(rows)-1, max(rows) + 2
    min_col, max_col = min(cols)-1, max(cols) + 2
    if min_row<0:
        min_row = 0
    if max_row>=panorama2.shape[0]:
        max_row = panorama2.shape[0] - 1
    if min_col<0:
        min_col = 0
    if max_col >= panorama2.shape[1]:
        max_col = panorama2.shape[1] - 1

    # equalise the brightness of image 2 with respect to image 1

    # found the min-cut within the intersection area  to blend.
    segments = find_cut(panorama1gray[min_row:max_row,min_col:max_col,:],panorama2gray[min_row:max_row,min_col:max_col,:],t_val)
   
    # created the mask for blending
    mask = np.zeros_like(panorama2)
    mask[tuple(zip(*indices))] = 255
    mask1 = np.zeros_like(panorama1[min_row:max_row,min_col:max_col,:])
    mask1[segments] = 255
    mask[min_row:max_row,min_col:max_col,:] = mask1
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
   
    # Blended the two image panaroma into one using pyramid blending and the mask obtained from cut
    result = Pyramid_blend(panorama1,panorama2,gray_mask,2)

    #Normalized the result and return it.
    norm_img = np.zeros_like(result)
    result = cv2.normalize(result,  norm_img, 0, 255, cv2.NORM_MINMAX)
    return result

if __name__ == "__main__":

    #The argument for the directory containing the images is taken.
    parser = argparse.ArgumentParser(description='Assignment1_mask_generator')
    parser.add_argument('-i', '--input_path', type=str, default='img', required=True, help="Path for the image folder")
    args = parser.parse_args()
    image_paths = sorted(os.listdir(args.input_path))

    # read the images.
    img1 = cv2.imread(os.path.join(args.input_path,image_paths[0]))
    img2 = cv2.imread(os.path.join(args.input_path,image_paths[1]))

    # Dynamically scaling the images.
    SCALING = int(((1500*2000)/(img1.shape[0]*img1.shape[1]))*100)
    if SCALING<40:
        SCALING = max(30,SCALING+10)
    elif SCALING>80:
        SCALING = min(SCALING-20,80)
    img1 = cv2.resize(img1, (((img1.shape[1])*SCALING)//100,(img1.shape[0]*SCALING)//100))
    img2 = cv2.resize(img2, (((img2.shape[1])*SCALING)//100,(img2.shape[0]*SCALING)//100))

    # The relative positioning of images is calculated.
    img1[np.where(img1==0)] = 1
    img2[np.where(img2==0)] = 1
    height1, width1 = img1.shape[:2]
    height2, width2 = img2.shape[:2]
    H1 = registration(img1,img2)
    pts1 = np.array([[0, 0], [0, height1], [width1, height1], [width1, 0]],np.float32).reshape(-1, 1, 2)
    pts2 = np.array([[0, 0], [0, height2], [width2, height2], [width2, 0]],np.float32).reshape(-1, 1, 2)
    pts2_ = cv2.perspectiveTransform(pts2, H1)
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel()-0.5)
    t_val = [-xmin, -ymin]

    # Calling the main function on the image according to the relative position calculated.
    if t_val[0]==0:
        final1 = process_image(img1,img2)
    else:
        final1 = process_image(img2,img1)

    # Writing the mosaic created
    cv2.imwrite("mosaic.jpg" , final1)
