#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import torch
from einops import repeat

import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel

def generate_neural_gaussians(viewpoint_camera, pc : GaussianModel, visible_mask=None, is_training=False):
    ## view frustum filtering for acceleration    
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)
    
    feat = pc._anchor_feat[visible_mask]
    anchor = pc.get_anchor[visible_mask]
    grid_offsets = pc._offset[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]

    ## get view properties for anchor
    ob_view = anchor - viewpoint_camera.camera_center
    # dist
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    # view
    ob_view = ob_view / ob_dist

    ## view-adaptive feature
    if pc.use_feat_bank:
        cat_view = torch.cat([ob_view, ob_dist], dim=1)
        
        bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1) # [n, 1, 3]

        ## multi-resolution feat
        feat = feat.unsqueeze(dim=-1)
        feat = feat[:,::4, :1].repeat([1,4,1])*bank_weight[:,:,:1] + \
            feat[:,::2, :1].repeat([1,2,1])*bank_weight[:,:,1:2] + \
            feat[:,::1, :1]*bank_weight[:,:,2:]
        feat = feat.squeeze(dim=-1) # [n, c]


    # ========== Optimization: Conditional concatenation to reduce memory ==========
    # Check what we actually need
    need_dist = pc.add_opacity_dist or pc.add_color_dist or pc.add_cov_dist
    need_no_dist = not pc.add_opacity_dist or not pc.add_color_dist or not pc.add_cov_dist

    # Create base feature (always needed)
    cat_local_view_wodist = torch.cat([feat, ob_view], dim=1)  # [N, c+3]

    # Only create with-dist version if any MLP needs it
    if need_dist:
        cat_local_view = torch.cat([cat_local_view_wodist, ob_dist], dim=1)  # [N, c+3+1]
    else:
        cat_local_view = None  # Save memory

    if pc.appearance_dim > 0:
        camera_indicies = torch.ones_like(cat_local_view_wodist[:,0], dtype=torch.long, device=ob_dist.device) * viewpoint_camera.uid
        appearance = pc.get_appearance(camera_indicies)

    # ========== FP16 Mixed Precision for MLP Inference ==========
    # Use autocast for MLP forward passes to reduce memory and increase speed
    # Activations computed in FP16, weights can stay in FP32
    with torch.amp.autocast('cuda', enabled=True):
        # get offset's opacity
        if pc.add_opacity_dist:
            neural_opacity = pc.get_opacity_mlp(cat_local_view) # [N, k]
        else:
            neural_opacity = pc.get_opacity_mlp(cat_local_view_wodist)

        # opacity mask generation
        neural_opacity = neural_opacity.reshape([-1, 1])
        mask = (neural_opacity>0.0)
        mask = mask.view(-1)

        # select opacity
        opacity = neural_opacity[mask]

        # get offset's color
        if pc.appearance_dim > 0:
            if pc.add_color_dist:
                color = pc.get_color_mlp(torch.cat([cat_local_view, appearance], dim=1))
            else:
                color = pc.get_color_mlp(torch.cat([cat_local_view_wodist, appearance], dim=1))
        else:
            if pc.add_color_dist:
                color = pc.get_color_mlp(cat_local_view)
            else:
                color = pc.get_color_mlp(cat_local_view_wodist)
        color = color.reshape([anchor.shape[0]*pc.n_offsets, 3])[mask]  # Early filter: [G,3]->[S,3]

        # get offset's cov
        if pc.add_cov_dist:
            scale_rot = pc.get_cov_mlp(cat_local_view)
        else:
            scale_rot = pc.get_cov_mlp(cat_local_view_wodist)
        scale_rot = scale_rot.reshape([anchor.shape[0]*pc.n_offsets, 7])[mask]  # Early filter: [G,7]->[S,7]

    # Convert back to FP32 for rasterizer (expects Float, not Half)
    opacity = opacity.float()
    color = color.float()
    scale_rot = scale_rot.float()
    # ========== End of FP16 Mixed Precision ==========

    # offsets
    offsets = grid_offsets.view([-1, 3])[mask]  # Early filter: [G,3]->[S,3]

    # ========== Optimization: Use indexing instead of repeat to avoid memory copy ==========
    # Instead of repeat [V,9] -> [G,9] -> [S,9], directly index [V] -> [S]
    # Create anchor indices for survived gaussians
    num_visible = anchor.shape[0]
    anchor_indices = torch.arange(num_visible, device=anchor.device).repeat_interleave(pc.n_offsets)[mask]  # [S]

    # Index into original arrays - no intermediate [G] tensor
    scaling_repeat = grid_scaling[anchor_indices]  # [S, 6] - direct indexing
    repeat_anchor = anchor[anchor_indices]  # [S, 3] - direct indexing

    # post-process cov
    scaling = scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3])
    rot = pc.rotation_activation(scale_rot[:,3:7])

    # ========== Optimization: In-place operations to reduce temporary tensors ==========
    # post-process offsets to get centers for gaussians
    offsets.mul_(scaling_repeat[:,:3])  # in-place multiplication
    xyz = repeat_anchor.add(offsets)  # add instead of creating new tensor with +

    if is_training:
        return xyz, color, opacity, scaling, rot, neural_opacity, mask
    else:
        return xyz, color, opacity, scaling, rot

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, visible_mask=None, retain_grad=False):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    is_training = pc.get_color_mlp.training

    if is_training:
        xyz, color, opacity, scaling, rot, neural_opacity, mask = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training)
    else:
        xyz, color, opacity, scaling, rot = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training)


    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass


    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii = rasterizer(
        means3D = xyz,
        means2D = screenspace_points,
        shs = None,
        colors_precomp = color,
        opacities = opacity,
        scales = scaling,
        rotations = rot,
        cov3D_precomp = None)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    if is_training:
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "selection_mask": mask,
                "neural_opacity": neural_opacity,
                "scaling": scaling,
                }
    else:
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                }

def prefilter_voxel(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_anchor, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_anchor


    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    radii_pure = rasterizer.visible_filter(means3D = means3D,
        scales = scales[:,:3],
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    return radii_pure > 0
