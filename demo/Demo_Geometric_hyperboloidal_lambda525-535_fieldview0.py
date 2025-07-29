"""
This script uses a hyperbolic phase distribution metalens.
Wavelength range: 525-535nm, field view: 0.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
# Add parent directory to path for importing diffoptics
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import diffoptics as do
from diffoptics.util.wavelength_to_rgb import wavelength_to_rgb

# # # Set to double precision
# torch.set_default_dtype(torch.float64)

# Folder to save results
save_picture_folder = 'results/Demo_Geometric_hyperboloidal_lambda525-535_fieldview0'
    
# Create folder if it doesn't exist
if not os.path.exists(save_picture_folder):
    os.makedirs(save_picture_folder)

# Initialize lens group
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
lens = do.Lensgroup(device=device)

# Lens parameters
R = 2.0  # Lens radius
surfaces = [
    do.Aspheric(R, 1.0, c=0., device=device),
    do.Meta_Geometric_hyperboloidal_phase_autograd(R, 1.5, focal_length=4.0, wavelength=530*1e-6, c=0., device=device),
]
materials = [
    do.Material('air'),
    do.Material('silica'),
    do.Material('air'),
]
lens.load(surfaces, materials)
lens.d_sensor = 5.5
lens.r_last = 2  # Sensor radius

wavelengths = torch.linspace(525, 535, 11).to(device)  # [nm]

R = R - 0.01  # [mm]

# Render rays to image plane
def render2image(wavelength, random=False):
    if wavelength.dim() == 0:  # If wavelength is a scalar
        wavelength = wavelength.unsqueeze(0)
    ray_init = lens.sample_ray(wavelength, M=201, R=R, sampling='grid', valid=True)
    ps = lens.trace_to_sensor(ray_init, ignore_invalid=True)  # Ray intersections with sensor
    return ps[..., :2]

# Calculate and save RMS data
RMS = []
for i in range(len(wavelengths)):
    wavelength = wavelengths[i]
    ps = render2image(wavelength)
    
    # Compute RMS (convert to microns)
    L = torch.sqrt(torch.mean(torch.sum(ps ** 2, axis=-1))) * 1000  # mm to μm
    print('final {:.0f}nm RMS loss: {:.3f} μm'.format(wavelength, L))
    RMS.append(L.item())

# Ray tracing visualization
wavelengths = torch.linspace(525, 535, 11).to(device)  # [nm]

# Generate colors for each wavelength
colors_list = []
for wavelength in wavelengths:
    rgb_color = wavelength_to_rgb(wavelength.item())
    hex_color = '#{:02x}{:02x}{:02x}'.format(rgb_color[0], rgb_color[1], rgb_color[2])
    colors_list.append(hex_color)

def trace_all(wavelength):
    if wavelength.dim() == 0:  
        wavelength = wavelength.unsqueeze(0)
    ray_init = lens.sample_ray_2D(R, wavelength, M=31)  # 2D ray sampling for visualization
    ps, oss = lens.trace_to_sensor_r(ray_init)
    return ps[..., :2], oss  

# Save ray tracing plots for each wavelength
for i in range(len(wavelengths)):
    wavelength = wavelengths[i]
    ps, oss = trace_all(wavelength)
    ax, fig = lens.plot_raytraces(oss, show=False, color=colors_list[i])
    fig.savefig(
        os.path.join(save_picture_folder, 'raytrace_{:.0f}nm.png'.format(wavelength)), 
        bbox_inches='tight',
        dpi=300
    )
    plt.close(fig)  # Free memory

# Plot RMS vs wavelength
plt.figure(figsize=(10, 7), facecolor='white')
plt.plot(wavelengths.cpu().numpy(), RMS, '-o', 
         linewidth=3.0, markersize=10, markeredgewidth=2.0, 
         color='#146eb4')

# Plot styling
plt.xlabel('Wavelength (nm)', fontweight='bold', fontname='Arial', fontsize=12)
plt.ylabel('RMS (μm)', fontweight='bold', fontname='Arial', fontsize=12)
plt.xlim(524, 536)
plt.ylim(-0.5, 17)
plt.xticks(np.arange(524, 538, 2))
plt.grid(False)
plt.box(True)

# Axis and font settings
ax = plt.gca()
ax.spines['top'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.tick_params(width=2)

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.weight'] = 'bold'
for tick in ax.get_xticklabels() + ax.get_yticklabels():
    tick.set_fontname('Arial')
    tick.set_fontweight('bold')
    tick.set_fontsize(12)

# Save RMS plot
plt.savefig(os.path.join(save_picture_folder, 'RMS_Plot.png'), 
            dpi=300, bbox_inches='tight')
plt.close()



