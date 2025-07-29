"""
This script compares the RMS performance of an optimized hybrid metalens and a single aspheric lens.
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

# Paths for loading and saving data
load_picture_folder = 'results/Optimize_Geometric_hybrid_achromatic'
save_picture_folder = 'results/Load_optimized_Geometric_hybrid_vs_SingleLens'

if not os.path.exists(save_picture_folder):
    os.makedirs(save_picture_folder)

# Device setup
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Wavelength range
wavelengths = torch.linspace(420, 680, 27).to(device)  # [nm]

# Lens radius
R = 4.0  # [mm]
R_sampling = R - 0.01  # [mm] Sampling radius for rays

def render2image(lens_system, wavelength):
    # Trace rays and return intersection points with the sensor
    if wavelength.dim() == 0:
        wavelength = wavelength.unsqueeze(0)
    ray_init = lens_system.sample_ray(wavelength, M=201, R=R_sampling, sampling='grid', valid=True)
    ps = lens_system.trace_to_sensor(ray_init, ignore_invalid=True)
    return ps[..., :2]

def calculate_rms_for_system(lens_system, wavelengths):
    # Calculate RMS for a lens system across wavelengths
    RMS = []
    for i in range(len(wavelengths)):
        wavelength = wavelengths[i]
        ps = render2image(lens_system, wavelength)
        
        # Convert RMS to micrometers
        L = torch.sqrt(torch.mean(torch.sum(ps ** 2, axis=-1))) * 1000  # mm to μm
        RMS.append(L.item())
    return RMS

# ================ 1. Hybrid Lens System ================
print("Calculating RMS for hybrid lens...")

# Load optimized lens parameters
mat_path = os.path.join(load_picture_folder, 'optimized_lens.mat')
mat_data = sio.loadmat(mat_path)
meta_radius = mat_data['meta_radius'][0]
meta_ai = mat_data['meta_ai'][0]
lens_1_c = mat_data['lens_1_c'][0]
lens_1_k = mat_data['lens_1_k'][0]
lens_1_ai = mat_data['lens_1_ai'][0]
lens_2_c = mat_data['lens_2_c'][0]

# Create hybrid lens system
hybrid_lens = do.Lensgroup(device=device)
hybrid_surfaces = [
    do.Aspheric(R, 1.0, c=0., device=device),
    do.Meta_Geometric_polofit_autograd(R, 1.5, c=0., ai=meta_ai, device=device),
    do.Aspheric(R, 3.0, c=lens_1_c, k=lens_1_k, ai=lens_1_ai, device=device),
    do.Aspheric(R, 6, c=lens_2_c, device=device),
]
hybrid_materials = [
    do.Material('air'),
    do.Material('silica'),
    do.Material('air'),
    do.Material('pmma'),
    do.Material('air'),
]
hybrid_lens.load(hybrid_surfaces, hybrid_materials)
hybrid_lens.d_sensor = 16
hybrid_lens.r_last = 2

# Calculate RMS for hybrid lens
hybrid_RMS = calculate_rms_for_system(hybrid_lens, wavelengths)

# ================ 2. Single Lens System ================
print("Calculating RMS for single lens...")

# Create single lens system
single_lens = do.Lensgroup(device=device)
single_surfaces = [
    do.Aspheric(R, 3.0, c=1 / 5.944973623339388E+000, k=-5.725578800111354E-001,
                ai=[1.004787933698957E-005, -4.920867551892389E-007], device=device),
    do.Aspheric(R, 6, c=0., device=device),
]
single_materials = [
    do.Material('air'),
    do.Material('pmma'),
    do.Material('air'),
]
single_lens.load(single_surfaces, single_materials)
single_lens.d_sensor = 16
single_lens.r_last = 2

# Calculate RMS for single lens
single_RMS = calculate_rms_for_system(single_lens, wavelengths)

# ================ 3. Plot Comparison ================
# Create figure
wavelengths_np = wavelengths.cpu().numpy()
plt.figure(figsize=(10, 7), facecolor='white')

# Plot curves with custom styling
plt.plot(wavelengths_np, single_RMS, '-d', linewidth=2.0, markersize=10, 
         markeredgewidth=2.0, color='#fdb913', label='Single lens (ours)')
plt.plot(wavelengths_np, hybrid_RMS, '-o', linewidth=2.0, markersize=10, 
         markeredgewidth=2.0, color='#228ae6', label='Hybrid metalens (ours)')

# Style the plot
plt.xlabel('Wavelength (nm)', fontweight='bold', fontname='Arial', fontsize=16)
plt.ylabel('RMS (μm)', fontweight='bold', fontname='Arial', fontsize=16)

# Set axis limits and ticks
plt.xlim(390, 710)
plt.ylim(-2, 58)
plt.xticks(np.arange(400, 701, 50))
plt.yticks(np.arange(0, 56, 10))

# Customize grid and borders
plt.grid(False)
plt.box(True)

# Adjust border and tick widths
ax = plt.gca()
ax.spines['top'].set_linewidth(3)
ax.spines['right'].set_linewidth(3)
ax.spines['bottom'].set_linewidth(3)
ax.spines['left'].set_linewidth(3)
ax.tick_params(width=3)

# Add legend
plt.legend(loc='upper right', frameon=False, prop={'family': 'Arial', 'weight': 'bold', 'size': 16})

# Set font properties
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.weight'] = 'bold'
for tick in ax.get_xticklabels() + ax.get_yticklabels():
    tick.set_fontname('Arial')
    tick.set_fontweight('bold')
    tick.set_fontsize(16)

# Save and display the plot
plt.tight_layout()
plt.savefig(os.path.join(save_picture_folder, 'RMS_Comparison_Hybrid_vs_Single.png'), 
            dpi=300, bbox_inches='tight')
plt.show()




