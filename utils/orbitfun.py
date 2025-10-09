import torch
import torch.nn.functional as F
import numpy as np
import random
import imageio


def render_frame(cam, img_width, img_height, mean, scale, rot, color, alpha, TILE_WIDTH=16, softp_off=4.0):
    from gsplat import compute_cov, cull, project, sortByTile, render

    alpha_act = torch.sigmoid(alpha)
    scale_act = F.softplus(scale-softp_off)

    cov = compute_cov(scale_act, rot)

    cam_mean, mask = cull(mean, cam, img_width, img_height)
    sp_mean, sp_cov = project(cam_mean, cov[mask], cam)

    tile_list = sortByTile(sp_mean, cam_mean[:, 2], sp_cov, img_width//TILE_WIDTH, img_height//TILE_WIDTH)

    cam_center = (-cam[1].T @ cam[2]).squeeze()
    frame = render(mean[mask], sp_mean, sp_cov, color[mask], alpha_act[mask], tile_list, cam_center, img_width, img_height)

    return frame


def look_at(cam_pos, center):
    up = torch.tensor([0.0, 1.0, 0.0])

    forward = center-cam_pos
    forward /= torch.linalg.norm(forward)

    right = torch.linalg.cross(forward, up)
    right /= torch.linalg.norm(right)

    true_up = torch.linalg.cross(right, forward)
    true_up /= torch.linalg.norm(true_up)

    return torch.stack([right, true_up, forward], dim=1).T


def random_cam_orbit(K, cameras, img_width, img_height, mean, scale, rot, color, alpha, steps=120, fps=30):
    idx = random.randint(0, len(cameras)-1)
    cam = [torch.from_numpy(K).float(), torch.from_numpy(cameras[idx][0]).float(), torch.from_numpy(cameras[idx][1]).float()]
    cam_pos = -cam[1].T @ cam[2].squeeze()

    center = mean.mean(dim=0)
    radius = torch.linalg.norm(cam_pos-center)

    frames = []
    angles = torch.linspace(0, 2*torch.pi, steps=steps)
    for a in angles:
        pos = torch.tensor([torch.cos(a), 0.0, torch.sin(a)])*radius+center
        cam[1] = look_at(pos, center)
        cam[2] = -cam[1] @ pos.unsqueeze(1)

        frame = render_frame(cam, img_width, img_height, mean, scale, rot, color, alpha)
        frames.append((frame*255).numpy().astype(np.uint8))

    imageio.mimsave("testset.gif", frames, fps=fps)

