"""
This script uses a hyperbolic phase distribution metalens.
Wavelength: 530nm, field view: 0-15.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
# Add parent directory to path for importing diffoptics
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import diffoptics as do

# # # Set to double precision
# torch.set_default_dtype(torch.float64)

# Folder to save results
save_picture_folder = 'results/Demo_Geometric_hyperboloidal_lambda530_fieldview0-15'
    
# Create folder if it doesn't exist
if not os.path.exists(save_picture_folder):
    os.makedirs(save_picture_folder)

# Initialize lens group
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
lens = do.Lensgroup(device=device)

# Lens parameters
R = 0.4  # Lens radius
surfaces = [
    do.Aspheric(R, 0.0, c=0., device=device),
    do.Meta_Geometric_hyperboloidal_phase_autograd(R, 0.5, focal_length=4.0, wavelength=530*1e-6, c=0., device=device),
]
materials = [
    do.Material('air'),
    do.Material('silica'),
    do.Material('air'),
]
lens.load(surfaces, materials)
lens.d_sensor = 4.5
lens.r_last = 1.5  # Sensor radius

wavelengths = torch.Tensor([530]).to(device)

R = R - 0.01  # [mm]

# Plot 2D setup with field views
colors_list = 'bgry'
views = np.linspace(0, 15, 4, endpoint=True)
ax, fig = lens.plot_setup2D_with_trace(views, wavelengths, M=4)
ax.axis('off')
ax.set_title('Field View Angle Setup 2D')
fig.savefig(os.path.join(save_picture_folder, 'field_view_angle_setup.pdf'))

# Generate spot diagrams for each field view
spot_rmss = []
valid_maps = []
for i, view in enumerate(views):
    ray = lens.sample_ray(wavelengths, view=view, M=31, sampling='grid', entrance_pupil=True)
    ps = lens.trace_to_sensor(ray, ignore_invalid=True)
    lim = 100e-3
    lens.spot_diagram(
        ps[...,:2], show=True, xlims=[-lim, lim], ylims=[-lim, lim], color=colors_list[i]+'.',
        savepath= os.path.join(save_picture_folder, 'sanity_check_field_view_{}.png'.format(int(view))),
        show_ticks=False
    )

    spot_rmss.append(lens.rms(ps))
    print('RMS for view {:.0f}: {:.3e}'.format(view, spot_rmss[-1][0]))

plt.show()

