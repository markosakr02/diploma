# Markos Akrivos AM5057

import os
import shutil

import cv2
import numpy as np

import matplotlib.pyplot as plt
import open3d as o3d
import random

import gsplat

sift = cv2.SIFT_create()

index_params = dict(algorithm=1, trees=5)
search_params = dict(checks=100)
flann = cv2.FlannBasedMatcher(index_params, search_params)

## TEST FUNCTIONS ##
def visualize1(img1, kp1, img2, kp2, matches):
    matched_img = cv2.drawMatches(img1, kp1, img2, kp2, random.sample(matches, 10), None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.figure(figsize=(12, 6))
    plt.imshow(matched_img)
    plt.axis('off')
    plt.title('Feature Matches')
    plt.show()


def visualize2(points, colors):
    print("Number of points: ", len(points))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    _, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd = pcd.select_by_index(ind)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

    render_option = vis.get_render_option()
    render_option.background_color = [0, 0, 0]

    vis.run()
    vis.destroy_window()


images = []
cameras = []
points = []
colors = []
img_names = os.listdir('images')

if os.path.exists("output") and os.path.isdir("output"):
    shutil.rmtree("output")
os.makedirs("output\\point_cloud")

img1 = cv2.imread('images\\'+img_names[0])
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
images.append(img1/255.0)

# assumed camera intrinsics
focal = 579.41
pp = (img1.shape[1]/2, img1.shape[0]/2)
K = np.array([[focal, 0, pp[0]],
              [0, focal, pp[1]],
              [0, 0, 1]])

R1 = np.eye(3)
t1 = np.zeros((3, 1))
cameras.append([R1, t1])

kp1, des1 = sift.detectAndCompute(cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY), None)

for i in range(1, len(img_names)):
    img2 = cv2.imread('images\\'+img_names[i])
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    kp2, des2 = sift.detectAndCompute(cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY), None)

    # find two nearest neighbors
    matches = flann.knnMatch(des1, des2, k=2)

    # apply Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7*n.distance:
            good_matches.append(m)

    if len(good_matches) < 30:
        continue
    images.append(img2/255.0)

    ## TEST 1 ##
    #visualize1(img1, kp1, img2, kp2, good_matches)

    pts1 = np.array([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.array([kp2[m.trainIdx].pt for m in good_matches])

    # structure from motion algorithm
    # ------------------------------------------
    E, mask = cv2.findEssentialMat(pts1, pts2, focal=focal, pp=pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    _, R12, t12, _ = cv2.recoverPose(E, pts1, pts2, focal=focal, pp=pp)

    # find R and t wrt the first image
    R2 = R12 @ R1
    t2 = R12 @ t1 + t12
    cameras.append([R2, t2])

    P1 = K @ np.hstack((R1, t1))
    P2 = K @ np.hstack((R2, t2))

    pts4D = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts3D = pts4D[:3]/pts4D[3]

    points += pts3D.T.tolist()
    # ------------------------------------------

    for pt in pts1:
        if pt[0] < 0:
            x = 0
        elif pt[0] >= img1.shape[1]:
            x = img1.shape[1]-1
        else:
            x = int(pt[0])

        if pt[1] < 0:
            y = 0
        elif pt[1] >= img1.shape[0]:
            y = img1.shape[0]-1
        else:
            y = int(pt[1])

        colors.append(img1[y, x]/255.0)

    img1 = img2
    kp1, des1 = kp2, des2
    R1, t1 = R2, t2

## TEST 2 ##
#visualize2(np.array(points), np.array(colors))

#gsplat.optimizer(images, K, cameras, np.array(points), np.array(colors))

