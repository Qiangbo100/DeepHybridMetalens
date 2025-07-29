"""
This script optimizes a hybrid achromatic lens combining a geometric metasurface and an aspheric lens for multiple wavelengths.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.io as sio
import sys
# Add parent directory to path for importing diffoptics
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import diffoptics as do
from diffoptics.util import helper
from diffoptics.util.wavelength_to_rgb import wavelength_to_rgb

torch.set_default_dtype(torch.float64)
helper.setup_seed(20)

save_picture_folder = 'results/Optimize_Geometric_hybrid_achromatic'
helper.mkdir(save_picture_folder)

# Initialize a lens
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
lens = do.Lensgroup(device=device)

R = 4.0
surfaces = [
    do.Aspheric(R, 1.0, c=0., device=device),
    do.Meta_Geometric_polofit_autograd(R, 1.5, c=0., device=device),
    do.Aspheric(R, 3.0, c=1 / 5.944973623339388E+000, k=-5.725578800111354E-001,
                ai=[1.004787933698957E-005, -4.920867551892389E-007], device=device),
    do.Aspheric(R, 6, c=0., device=device),
]
materials = [
    do.Material('air'),
    do.Material('silica'),
    do.Material('air'),
    do.Material('pmma'),
    do.Material('air'),
]
lens.load(surfaces, materials)
lens.d_sensor = 16
lens.r_last = 2  # Radius of the sensor
coefficient_rms_chr_z = 1  # Penalty coefficient for chromatic aberration

wavelengths = torch.linspace(420, 680, 14).to(device)  # [nm]
colors_list = []
# Generate RGB colors for each wavelength
for wavelength in wavelengths:
    rgb_color = wavelength_to_rgb(wavelength.item())
    hex_color = '#{:02x}{:02x}{:02x}'.format(rgb_color[0], rgb_color[1], rgb_color[2])
    colors_list.append(hex_color)

R = R - 0.01  # [mm]

def render2image(wavelength, random=False):
    if wavelength.dim() == 0:
        wavelength = wavelength.unsqueeze(0)
    ray_init = lens.sample_ray(wavelength, M=201, R=R, sampling='grid', valid=True)
    ps = lens.trace_to_sensor(ray_init, ignore_invalid=True)  # Intersection points with the sensor
    return ps[..., :2]

def render2ChiefRay(wavelength):
    if wavelength.dim() == 0:
        wavelength = wavelength.unsqueeze(0)
    x = torch.arange(0.01, 1, 0.01, device=device) * R
    y = torch.zeros_like(x, device=device)

    o = torch.stack((x, y, torch.zeros_like(x, device=device)), axis=-1)
    d = torch.zeros_like(o)
    d[..., 2] = torch.ones_like(x)
    ray_init = do.Ray(o, d, wavelength, device=device)
    ps = lens.trace_to_sensor(ray_init, intersect2axis='x')  # Intersection points with the optical axis
    return ps[..., 2]

def render_all():
    ps = []
    ps_z = []
    for wavelength in wavelengths:
        ps.append(render2image(wavelength))
        ps_z.append(torch.abs(render2ChiefRay(wavelength)-lens.d_sensor))
    return [ps, ps_z]

def loss_fn(ps_both):
    def spherical_rms(arr):
        arr = torch.cat(arr, axis=0)
        return torch.sqrt(torch.mean(torch.sum(arr ** 2, axis=-1)))

    def chromatic_z_rms(arr):
        chromatic_rms_tensor = [torch.sqrt(torch.mean(arr_wavelength ** 2)) for arr_wavelength in arr]
        chromatic_rms_tensor = torch.stack(chromatic_rms_tensor)
        return torch.sqrt(torch.mean(chromatic_rms_tensor ** 2))
    
    ps, ps_z = ps_both
    rms_sph = spherical_rms(ps)
    rms_chr_z = chromatic_z_rms(ps_z)
    loss = rms_sph + coefficient_rms_chr_z * rms_chr_z
    return loss

def trace_all(wavelength):
    if wavelength.dim() == 0:
        wavelength = wavelength.unsqueeze(0)
    ray_init = lens.sample_ray_2D(R, wavelength, M=31)  # Sample rays for 2D visualization
    ps, oss = lens.trace_to_sensor_r(ray_init)
    return ps[..., :2], oss  # Intersection points and ray paths

# ----------Initial setup visualization----------
for i in range(len(wavelengths)):
    wavelength = wavelengths[i]
    ps, oss = trace_all(wavelength)
    ax, fig = lens.plot_raytraces(oss, show=False, color=colors_list[i])
    fig.savefig(os.path.join(save_picture_folder, 'setup_initial_{:.0f}.png'.format(wavelength)), bbox_inches='tight',
                dpi=300)

# ----------Optimization----------
diff_names = [
    'surfaces[1].ai',
    'surfaces[2].c',
    'surfaces[2].k',
    'surfaces[2].ai',
    'surfaces[3].c',
]

out = do.Adam(lens, diff_names, lr=1e-4, lrs=[1, 1, 1, 0.1, 1], gamma_rate=0.95) \
    .optimize(loss_func=loss_fn, render=render_all, maxit=50, sub_iteration=1000,
              record=False)

# ----------Final results visualization----------
# Plot loss
plt.figure()
plt.plot(out['ls'])
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.savefig(os.path.join(save_picture_folder, 'loss.png'), bbox_inches='tight', dpi=300)

for i in range(len(wavelengths)):
    wavelength = wavelengths[i]
    ps, oss = trace_all(wavelength)
    ax, fig = lens.plot_raytraces(oss, show=False, color=colors_list[i])
    fig.savefig(os.path.join(save_picture_folder, 'setup_final_{:.0f}.png'.format(wavelength)), bbox_inches='tight',
                dpi=300)

# Print optimized lens parameters
print(lens.surfaces[1])
print(lens.surfaces[2])
print(lens.surfaces[3])

# Save optimized lens parameters to .mat file
save_mat_path = os.path.join(save_picture_folder, 'optimized_lens.mat')
sio.savemat(save_mat_path,
            {'meta_period': lens.surfaces[1].period.cpu().numpy(),
             'meta_radius': lens.surfaces[1].r,
             'meta_ai_bound': lens.surfaces[1].ai_bound,
             'meta_ai': lens.surfaces[1].ai.detach().cpu().numpy(),
             'meta_d': lens.surfaces[1].d.detach().cpu().numpy(),
             'lens_1_c': lens.surfaces[2].c.detach().cpu().numpy(),
             'lens_1_k': lens.surfaces[2].k.detach().cpu().numpy(),
             'lens_1_ai': lens.surfaces[2].ai.detach().cpu().numpy(),
             'lens_1_d': lens.surfaces[2].d.detach().cpu().numpy(),
             'lens_2_c': lens.surfaces[3].c.detach().cpu().numpy(),
             })
