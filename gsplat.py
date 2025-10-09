# Markos Akrivos AM5057

import os
import shutil

import cv2
import pycolmap
import open3d as o3d

import random

import torch
from pytorch_msssim import ssim
from utils.rotation_conversions import quaternion_to_matrix

import torch.nn.functional as F
import kornia

import numpy as np
import matplotlib.pyplot as plt

from utils.orbitfun import random_cam_orbit

from tqdm import tqdm

# optimizer parameters
lr = 0.001
epochs = 1000

softp_off = 4.0

L = 0.2
l1_loss = torch.nn.L1Loss()

# density control parameters
scale_prune = 2.0
scale_split = 1.0

Ea = 0.005
Tpos = 0.0002
f = 1.6

# rasterizer parameters
near = 0.1
far = 10

TILE_WIDTH = 16

blend_stop = 0.0001

## HELPER FUNCTIONS ##
def compute_cov(scale, rot):
    S = torch.diag_embed(scale**2)
    R = quaternion_to_matrix(rot)
    
    cov = R @ S @ R.transpose(-1, -2)           # transpose the last two dimensions of the batch

    return 0.5*(cov+cov.transpose(-1, -2))      # symmetrization


def compute_sh_base(view_direction):
    Y00 = view_direction[:, 0]*0+0.282095
    Y1m1 = 0.488603*view_direction[:, 1]
    Y10 = 0.488603*view_direction[:, 2]
    Y11 = 0.488603*view_direction[:, 0]
    Y2m2 = 1.092548*view_direction[:, 0]*view_direction[:, 1]
    Y2m1 = 1.092548*view_direction[:, 1]*view_direction[:, 2]
    Y20 = 0.315392*(3*view_direction[:, 2]**2-1)
    Y21 = 1.092548*view_direction[:, 0]*view_direction[:, 2]
    Y22 = 0.546274*(view_direction[:, 0]**2-view_direction[:, 1]**2)

    return torch.stack([Y00, Y1m1, Y10, Y11, Y2m2, Y2m1, Y20, Y21, Y22], dim=0).T


def antialiasing(img, img_width, img_height, scale=0.25, kernel=5, detail_boost=0.5):
    img = img.permute(2, 0, 1).unsqueeze(0)

    img_down = kornia.geometry.transform.resize(img, (int(scale*img_height), int(scale*img_width)), interpolation='bilinear')
    img_blur = kornia.filters.median_blur(img_down, (kernel, kernel))

    low_freq = kornia.geometry.transform.resize(img_blur, (img_height, img_width), interpolation='bilinear')
    high_freq = img-kornia.filters.median_blur(img, (kernel, kernel))

    img = low_freq+detail_boost*high_freq

    return img.squeeze(0).permute(1, 2, 0)


def visualize3(image, epoch, loss):
    if epoch%100 == 0 or epoch == 1:
        plt.imshow(image)
        plt.axis('off')
        plt.title(f'epoch: {epoch}, loss: {loss}')

        plt.savefig(f"render_out\\iter_{epoch}.png")
        plt.close()

## MAIN PROGRAM FUNCTIONS ##
#@profile
def init(mean, cinfo):
    Npts = len(mean)

    scale = torch.zeros((Npts, 3), dtype=torch.float32)
    rot = torch.zeros((Npts, 4), dtype=torch.float32)
    color = torch.zeros((Npts, 9, 3), dtype=torch.float32)
    alpha = torch.zeros(Npts, dtype=torch.float32)

    for i in range(Npts):
        # isotropic Gaussian with axes equal to the mean
        # of the distance to the closest 3 points
        dist = torch.linalg.norm(mean-mean[i], axis=1)
        _, idx = torch.topk(dist, k=4, largest=False)

        base = torch.mean(dist[idx[1:]])
        scale[i] = base+base*0.05*torch.randn(3)     # 5% relative noise

        rot[i, 0] = 1.0

        color[i, 0] = cinfo[i]
        color[i, 1:] = 0.01*torch.randn((8, 3))

        alpha[i] = 0.8

    return scale, rot, color, alpha

#@profile
def cull(mean, cam, img_width, img_height):
    cam_mean = (cam[1] @ mean.T + cam[2]).T

    x_left = -cam[0][0, 2]/cam[0][0, 0]
    x_right = (img_width-cam[0][0, 2])/cam[0][0, 0]
    y_bottom = -cam[0][1, 2]/cam[0][1, 1]
    y_top = (img_height-cam[0][1, 2])/cam[0][1, 1]

    inside_frustum = (
        (cam_mean[:, 2] >= near) & (cam_mean[:, 2] <= far) &                                          # Near/Far
        (cam_mean[:, 0] >= x_left*cam_mean[:, 2]) & (cam_mean[:, 0] <= x_right*cam_mean[:, 2]) &      # Left/Right
        (cam_mean[:, 1] >= y_bottom*cam_mean[:, 2]) & (cam_mean[:, 1] <= y_top*cam_mean[:, 2])        # Bottom/Top
    )

    return cam_mean[inside_frustum], inside_frustum

#@profile
def project(cam_mean, cov, cam):
    sp_mean = (cam[0] @ cam_mean.T).T
    sp_mean_proj = sp_mean[:, :2]/sp_mean[:, 2].unsqueeze(1)

    cam_cov = cam[1] @ cov @ cam[1].T

    J00 = cam[0][0, 0]/cam_mean[:, 2]
    J11 = cam[0][1, 1]/cam_mean[:, 2]
    J02 = -(cam[0][0, 0]*cam_mean[:, 0])/cam_mean[:, 2]**2
    J12 = -(cam[0][1, 1]*cam_mean[:, 1])/cam_mean[:, 2]**2

    J = torch.stack(
        [torch.stack([J00, torch.zeros_like(J00), J02], dim=1),
                torch.stack([torch.zeros_like(J11), J11, J12], dim=1)],
        dim=1)

    sp_cov = J @ cam_cov @ J.transpose(-1, -2)

    return sp_mean_proj, sp_cov+1e-4*torch.eye(2)

#@profile
def sortByTile(sp_mean, depths, sp_cov, Tcol, Trow):
    # globally sort Gaussians by depth
    sorted_indices = torch.argsort(depths)

    # find which tiles each 3D Gaussian overlaps with
    bound = 3*torch.sqrt(torch.linalg.eigvalsh(sp_cov)[:, -1])
    min_tile = torch.clamp((sp_mean[sorted_indices]-bound[sorted_indices].unsqueeze(1))//TILE_WIDTH, min=torch.tensor([0, 0]), max=torch.tensor([Tcol-1, Trow-1]))
    max_tile = torch.clamp((sp_mean[sorted_indices]+bound[sorted_indices].unsqueeze(1))//TILE_WIDTH, min=torch.tensor([0, 0]), max=torch.tensor([Tcol-1, Trow-1]))

    tile_list = [[] for i in range(Tcol*Trow)]
    for i in range(len(sorted_indices)):
        tile_coords = torch.cartesian_prod(torch.arange(min_tile[i, 0].item(), max_tile[i, 0].item()+1),
                                           torch.arange(min_tile[i, 1].item(), max_tile[i, 1].item()+1))

        for tile in tile_coords:
            tile_list[int(Tcol*tile[1].item()+tile[0].item())].append(sorted_indices[i])

    return tile_list
    
#@profile
def render(mean, sp_mean, sp_cov, color, alpha, tile_list, cam_center, img_width, img_height):
    num_tiles = len(tile_list)
    Tcol = img_width//TILE_WIDTH
    res = torch.zeros((img_height, img_width, 3), dtype=torch.float32)

    # precompute color for all Gaussians
    viewdir = mean-cam_center
    viewdir = viewdir/(torch.linalg.norm(viewdir, dim=1).unsqueeze(1)+1e-6)
    rgbval = torch.bmm(compute_sh_base(viewdir).unsqueeze(1), color).squeeze(1)      # batched matrix multiplication

    max_len = max(len(t) for t in tile_list)
    tile_list = torch.tensor([t+[-1]*(max_len-len(t)) for t in tile_list])

    tile_ids = torch.arange(num_tiles)
    tx = tile_ids%Tcol
    ty = tile_ids//Tcol

    x0 = tx*TILE_WIDTH
    y0 = ty*TILE_WIDTH
    x1 = torch.clamp(x0+TILE_WIDTH, max=img_width)
    y1 = torch.clamp(y0+TILE_WIDTH, max=img_height)

    pixel_coords = torch.zeros((num_tiles, TILE_WIDTH**2, 2), dtype=torch.float32)                              # [B, P, 2]
    for i in range(num_tiles):
        coords = torch.cartesian_prod(torch.arange(x0[i], x0[i]+TILE_WIDTH, dtype=torch.float32),
                                      torch.arange(y0[i], y0[i]+TILE_WIDTH, dtype=torch.float32))
        pixel_coords[i] = coords

    tiles = torch.zeros((pixel_coords.shape[0], pixel_coords.shape[1], 3), dtype=torch.float32)                 # [B, P, 3]
    trans = torch.ones((pixel_coords.shape[0], pixel_coords.shape[1]), dtype=torch.float32)                     # [B, P]

    active_tiles = torch.ones_like(tile_ids, dtype=torch.bool)
    pixels_mask = torch.ones_like(trans, dtype=torch.bool)
    for i in range(max_len):
        col = tile_list[:, i]                                                                                   # [B]

        valid_mask = (col != -1) & active_tiles                                                                 # [B]
        if not valid_mask.any():
            break
        col = col[valid_mask]                                                                                   # [B']

        offsets_T = (pixel_coords[valid_mask]-sp_mean[col].unsqueeze(1)).transpose(1, 2)             # [B', 2, P]

        power = -0.5*torch.clamp(torch.sum(offsets_T*torch.linalg.solve(sp_cov[col], offsets_T), dim=1), min=0.0)       # Mahalanobis squared
        weights = torch.exp(power)                                                                              # [B', P]

        alpha_w = weights*alpha[col].unsqueeze(1)                                                               # [B', P]
        blend_rgb = (trans[valid_mask]*alpha_w).unsqueeze(2)*rgbval[col].unsqueeze(1)                           # [B', P, 3]

        tiles[valid_mask] = tiles[valid_mask]+blend_rgb*pixels_mask[valid_mask].unsqueeze(2)
        trans[valid_mask] = trans[valid_mask]*(1-alpha_w)*pixels_mask[valid_mask]

        # check for tiles that have all their pixels saturated
        if i < max_len-1:
            pixels_mask[valid_mask] = trans[valid_mask] > blend_stop
            tiles_mask = ~pixels_mask[valid_mask].any(dim=1)

            active_tiles[tile_ids[valid_mask][tiles_mask]] = False

            if not active_tiles.any():
                break

    tiles = tiles.reshape(tiles.shape[0], TILE_WIDTH, TILE_WIDTH, 3)
    for i, tile in enumerate(tiles):
        res[y0[i]:y1[i], x0[i]:x1[i]] = tile[:y1[i]-y0[i], :x1[i]-x0[i]]

    res = antialiasing(res, img_width, img_height)

    return torch.clamp(res, min=0.0, max=1.0)


def optimizer(images, K, cameras, mean, cinfo):
    if os.path.exists("render_out"):
        shutil.rmtree("render_out")
    os.makedirs("render_out")

    print("Initializing 3D spheres...")
    mean = torch.from_numpy(mean).float()
    cinfo = torch.from_numpy(cinfo).float()
    scale, rot, color, alpha = init(mean, cinfo)

    print("Now training...")
    mean = torch.nn.Parameter(mean); scale = torch.nn.Parameter(scale); rot = torch.nn.Parameter(rot)
    color = torch.nn.Parameter(color); alpha = torch.nn.Parameter(alpha)
    stohasticgd = torch.optim.Adam([mean, scale, rot, color, alpha], lr=lr)

    for t in tqdm(range(epochs)):
        stohasticgd.zero_grad()

        # random target sample from input images
        idx = random.randint(0, len(images)-1)
        img = torch.from_numpy(images[idx]).float()

        cam = [torch.from_numpy(K).float(), torch.from_numpy(cameras[idx][0]).float(), torch.from_numpy(cameras[idx][1]).float()]

        # activation
        alpha_act = torch.sigmoid(alpha)
        scale_act = F.softplus(scale-softp_off)

        # rasterize
        cov = compute_cov(scale_act, rot)

        cam_mean, mask = cull(mean, cam, img.shape[1], img.shape[0])
        sp_mean, sp_cov = project(cam_mean, cov[mask], cam)

        with torch.no_grad():
            tile_list = sortByTile(sp_mean, cam_mean[:, 2], sp_cov, img.shape[1]//TILE_WIDTH, img.shape[0]//TILE_WIDTH)

        cam_center = (-cam[1].T @ cam[2]).squeeze()
        img_est = render(mean[mask], sp_mean, sp_cov, color[mask], alpha_act[mask], tile_list, cam_center, img.shape[1], img.shape[0])

        # compute loss and backpropagate
        loss = (1-L)*l1_loss(img_est, img)+L*(1-ssim(img_est.permute(2, 0, 1).unsqueeze(0), img.permute(2, 0, 1).unsqueeze(0), data_range=1.0, size_average=True))
        loss.backward()

        stohasticgd.step()

        ## TEST 3 ##
        visualize3(img_est.detach().numpy(), t+1, loss.item())

        # refinement every 100 epochs
        if (t+1)%100 == 0:
            gpos = mean.grad

            with torch.no_grad():
                # pruning
                idx = (alpha > Ea) & (torch.linalg.norm(scale, axis=1) < scale_prune)

                mean = mean[idx]
                gpos = gpos[idx]
                scale = scale[idx]
                rot = rot[idx]
                color = color[idx]
                alpha = alpha[idx]

                # densification
                idx = torch.nonzero(torch.mean(gpos, dim=1) > Tpos, as_tuple=True)[0]

                if len(idx) != 0:
                    cond = torch.linalg.norm(scale[idx], axis=1) > scale_split

                    if len(idx[cond]) != 0:
                        # over-reconstruction case
                        dist = torch.distributions.MultivariateNormal(mean[idx[cond]], compute_cov(scale[idx[cond]], rot[idx[cond]]))

                        mean[idx[cond]] = dist.sample()
                        mean = torch.cat((mean, dist.sample()))

                        scale[idx[cond]] = (1/f)*scale[idx[cond]]
                        scale = torch.cat((scale, scale[idx[cond]]))

                        rot = torch.cat((rot, rot[idx[cond]]))
                        color = torch.cat((color, color[idx[cond]]))
                        alpha = torch.cat((alpha, alpha[idx[cond]]))

                    if len(idx[~cond]) != 0:
                        # under-reconstruction case
                        mean = torch.cat((mean, mean[idx[~cond]]+gpos[idx[~cond]]))
                        scale = torch.cat((scale, scale[idx[~cond]]))
                        rot = torch.cat((rot, rot[idx[~cond]]))
                        color = torch.cat((color, color[idx[~cond]]))
                        alpha = torch.cat((alpha, alpha[idx[~cond]]))

                # opacity reset every 3000 epochs
                if (t+1)%3000 == 0:
                    alpha[alpha > 0.01] = 0.01

                # optimizer reset
                mean = torch.nn.Parameter(mean.clone()); scale = torch.nn.Parameter(scale.clone()); rot = torch.nn.Parameter(rot.clone())
                color = torch.nn.Parameter(color.clone()); alpha = torch.nn.Parameter(alpha.clone())
                stohasticgd = torch.optim.Adam([mean, scale, rot, color, alpha], lr=lr)

    print("Training complete.")
    random_cam_orbit(K, cameras, images[0].shape[1], images[0].shape[0],
                     mean.detach(), scale.detach(), rot.detach(), color.detach(), alpha.detach())

## COLMAP VERSION OF MAIN ##
## SAVE COLMAP RESULT IN COLMAP FOLDER ##
if __name__ == "__main__":
    images = []
    cameras = []

    # manual adjust, check K file in scene folder
    K = np.array([[1110.54, 0, 400.0],
                  [0, 1110.54, 400.0],
                  [0, 0, 1]])

    # colmap\\name_of_scene\\...
    recon = pycolmap.Reconstruction('colmap\\hotdog\\sparse\\0')
       
    # use images folder of the selected scene in the same folder as the program
    for image in recon.images.values():
        img = cv2.imread('images\\'+image.name)
        images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255.0)

        T = image.cam_from_world()

        R = T.rotation.matrix()
        t = T.translation.reshape(3, 1)
        cameras.append([R, t])

    pcd = o3d.io.read_point_cloud('colmap\\hotdog\\sparse\\0\\points3D.ply')    # must export points3D.bin to ply before using
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    optimizer(images, K, cameras, points, colors)
