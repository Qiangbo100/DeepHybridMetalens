import torch

from .shapes import *
import matplotlib.pyplot as plt
import pathlib


def tex(img_2d, size_2d, x, y, bmode=BoundaryMode.replicate):  # texture indexing function
    if bmode is BoundaryMode.zero:
        raise NotImplementedError()
    elif bmode is BoundaryMode.replicate:
        x = torch.clamp(x, min=0, max=size_2d[0] - 1)
        y = torch.clamp(y, min=0, max=size_2d[1] - 1)
    elif bmode is BoundaryMode.symmetric:
        raise NotImplementedError()
    elif bmode is BoundaryMode.periodic:
        raise NotImplementedError()
    img = img_2d[x.flatten(), y.flatten()]
    return img.reshape(x.shape)


def tex4(img_2d, size_2d, x0, y0, bmode=BoundaryMode.replicate):  # texture indexing four pixels
    _tex = lambda x, y: tex(img_2d, size_2d, x, y, bmode)
    s00 = _tex(x0, y0)
    s01 = _tex(x0, 1 + y0)
    s10 = _tex(1 + x0, y0)
    s11 = _tex(1 + x0, 1 + y0)
    return s00, s01, s10, s11


class Lensgroup(Endpoint):
    """
    The origin of the Lensgroup, which is a collection of multiple optical surfaces, is located at "origin".
    The Lensgroup can rotate freely around the x/y axes, and the rotation angles are defined as "theta_x", "theta_y", and "theta_z" (in degrees).
    
    In the Lensgroup's coordinate system, which is the object frame coordinate system, surfaces are arranged starting from "z = 0".
    There is a small 3D origin shift, called "shift", between the center of the surface (0,0,0) and the mount's origin.
    The sum of the shift and the origin is equal to the Lensgroup's origin.
    
    There are two configurations for ray tracing: forward and backward.
    - In the forward mode, rays begin at the surface with "d = 0" and propagate along the +z axis, e.g. from scene to image plane.
    - In the backward mode, rays begin at the surface with "d = d_max" and propagate along the -z axis, e.g. from image plane to scene.
    """

    def __init__(self, origin=np.zeros(3), shift=np.zeros(3), theta_x=0., theta_y=0., theta_z=0.,
                 device=torch.device('cpu')):
        self.origin = torch.Tensor(origin).to(device)
        self.shift = torch.Tensor(shift).to(device)
        self.theta_x = torch.Tensor(np.asarray(theta_x)).to(device)
        self.theta_y = torch.Tensor(np.asarray(theta_y)).to(device)
        self.theta_z = torch.Tensor(np.asarray(theta_z)).to(device)
        self.device = device

        # Sequentials properties
        self.surfaces = []
        self.materials = []

        # Sensor properties
        self.pixel_size = 6.45  # [um]
        self.film_size = [640, 480]  # [pixel]

        # Endpoint.__init__(self, self._compute_transformation(), device)

        # TODO: in case you would like to render something in Mitsuba2 ...
        self.mts_prepared = False

    def load_file(self, filename: pathlib.Path):
        self.surfaces, self.materials, self.r_last, d_last = self.read_lensfile(str(filename))
        self.d_sensor = d_last + self.surfaces[-1].d
        self._sync()

    def load(self, surfaces: list, materials: list):
        self.surfaces = surfaces
        self.materials = materials
        self._sync()

    def _sync(self):
        for i in range(len(self.surfaces)):
            self.surfaces[i].to(self.device)
        self.aperture_ind = self._find_aperture()

    def _find_aperture(self):
        for i in range(len(self.surfaces) - 1):
            if self.materials[i].A < 1.0003 and self.materials[i + 1].A < 1.0003:  # both are AIR
                return i

    @staticmethod
    def read_lensfile(filename):
        surfaces = []
        materials = []
        ds = []  # no use for now
        with open(filename) as file:
            line_no = 0
            d_total = 0.
            for line in file:
                if line_no < 2:  # first two lines are comments; ignore them
                    line_no += 1
                else:
                    ls = line.split()
                    surface_type, d, r = ls[0], float(ls[1]), float(ls[3]) / 2
                    roc = float(ls[2])
                    if roc != 0: roc = 1 / roc
                    materials.append(Material(ls[4]))

                    d_total += d
                    ds.append(d)

                    if surface_type == 'O':  # object
                        d_total = 0.
                        ds.pop()
                    # elif surface_type == 'X': # XY-polynomial
                    #     del roc
                    #     ai = []
                    #     for ac in range(5, len(ls)):
                    #         if ac == 5:
                    #             b = float(ls[5])
                    #         else:
                    #             ai.append(float(ls[ac]))
                    #     surfaces.append(XYPolynomial(r, d_total, J=3, ai=ai, b=b))
                    # elif surface_type == 'B': # B-spline
                    #     del roc
                    #     ai = []
                    #     for ac in range(5, len(ls)):
                    #         if ac == 5:
                    #             nx = int(ls[5])
                    #         elif ac == 6:
                    #             ny = int(ls[6])
                    #         else:
                    #             ai.append(float(ls[ac]))
                    #     tx = ai[:nx+8]
                    #     ai = ai[nx+8:]
                    #     ty = ai[:ny+8]
                    #     ai = ai[ny+8:]
                    #     c  = ai
                    #     surfaces.append(BSpline(r, d, size=[nx, ny], tx=tx, ty=ty, c=c))
                    elif surface_type == 'M':  # mixed-type of X and B
                        raise NotImplementedError()
                    # elif surface_type == 'META':  # metasurface
                    #     surfaces.append(
                    #         Metasurface(r, d_total, roc))
                    elif surface_type == 'S':  # aspheric surface
                        if len(ls) <= 5:
                            surfaces.append(
                                Aspheric(r, d_total, roc))
                        else:
                            ai = []
                            for ac in range(5, len(ls)):
                                if ac == 5:
                                    conic = float(ls[5])
                                else:
                                    ai.append(float(ls[ac]))
                            surfaces.append(Aspheric(r, d_total, roc, conic, ai))
                    elif surface_type == 'A':  # aperture
                        surfaces.append(Aspheric(r, d_total, roc))
                    elif surface_type == 'I':  # sensor
                        d_total -= d
                        ds.pop()
                        materials.pop()
                        r_last = r
                        d_last = d
        return surfaces, materials, r_last, d_last

    def reverse(self):
        # reverse surfaces
        d_total = self.surfaces[-1].d
        for i in range(len(self.surfaces)):
            self.surfaces[i].d = d_total - self.surfaces[i].d
            self.surfaces[i].reverse()  
        self.surfaces.reverse()

        # reverse materials
        self.materials.reverse()

    # ------------------------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------------------------
    def rms(self, ps, units=1, option='centroid', squared=False):
        """
        Compute RMS of the spot diagram.
        """
        ps = ps[..., :2] * units
        if option == 'centroid':
            ps_mean = torch.mean(ps, axis=0)
        ps = ps - ps_mean[None, ...]  # we now use normalized ps
        if squared:
            return torch.mean(torch.sum(ps ** 2, axis=-1)), ps / units
        else:
            return torch.sqrt(torch.mean(torch.sum(ps ** 2, axis=-1))), ps / units

    def spot_diagram(self, ps, show=True, xlims=None, ylims=None, color='b.', savepath=None, show_ticks=True):
        """
        Plot spot diagram.
        """
        units = 1
        spot_rms = float(self.rms(ps, units)[0])
        ps = ps.cpu().detach().numpy()[..., :2]
        ps_mean = np.mean(ps, axis=0)  # centroid
        ps = ps - ps_mean[None, ...]  # we now use normalized ps

        fig = plt.figure()
        ax = plt.axes()
        ax.plot(ps[..., 1], ps[..., 0], color)
        # ax.scatter(ps[..., 1], ps[..., 0], c=color, s=1, edgecolors='none')
        plt.gca().set_aspect('equal', adjustable='box')

        if xlims is not None:
            plt.xlim(*xlims)
        if ylims is not None:
            plt.ylim(*ylims)
        ax.set_aspect(1. / ax.get_data_ratio())
        units_str = '[mm]'
        
        if show_ticks:
            plt.xlabel('x ' + units_str)
            plt.ylabel('y ' + units_str)
            plt.xticks(np.linspace(xlims[0], xlims[1], 11))
            plt.yticks(np.linspace(ylims[0], ylims[1], 11))
        else:
            plt.xlabel('')
            plt.ylabel('')
            plt.xticks([])
            plt.yticks([])
        # plt.grid(True)

        if savepath is not None:
            fig.savefig(savepath, bbox_inches='tight', dpi=300)
        if show:
            plt.show()
        else:
            plt.close()

        return spot_rms

    def get_lines_from_plot_setup2D(self, with_sensor=True):
        """
        Generate and return line data for 2D plotting of optical surfaces in lens group.

        Args:
            with_sensor (bool): Whether to include sensor plane in plot.

        Returns:
            lines (list): List of line data dicts for each surface contour. Each dict contains:
                - z: z-coordinates
                - x: x-coordinates  
                - id: surface ID

        Notes:
            - Visualizes 2D layout of lens group including optical surfaces and aperture
            - Computes points on each surface and transforms to world coordinates
            - Optionally includes sensor plane representation
            - Useful for analyzing lens group structure
        """
        lines = []

        # to world coordinate
        def plot(lines: list, surface_id, z, x):
            # p = self.to_world.transform_point(
            #     torch.stack(
            #         (x, torch.zeros_like(x, device=self.device), z), axis=-1
            #     )
            # ).cpu().detach().numpy()
            p = torch.stack(
                (x, torch.zeros_like(x, device=self.device), z), axis=-1
            ).cpu().detach().numpy()
            lines.append({'z': p[..., 2], 'x': p[..., 0], 'id': surface_id})

        def draw_aperture(lines: list, surface, surface_id):
            """
            Draw aperture.
            """
            N = 3
            d = surface.d.cpu()
            R = surface.r
            APERTURE_WEDGE_LENGTH = 0.05 * R  # [mm]
            APERTURE_WEDGE_HEIGHT = 0.15 * R  # [mm]

            # wedge length
            z = torch.linspace(d - APERTURE_WEDGE_LENGTH, d + APERTURE_WEDGE_LENGTH, N, device=self.device)
            x = -R * torch.ones(N, device=self.device)
            plot(lines, surface_id, z, x)
            x = R * torch.ones(N, device=self.device)
            plot(lines, surface_id, z, x)

            # wedge height
            z = d * torch.ones(N, device=self.device)
            x = torch.linspace(R, R + APERTURE_WEDGE_HEIGHT, N, device=self.device)
            plot(lines, surface_id, z, x)
            x = torch.linspace(-R - APERTURE_WEDGE_HEIGHT, -R, N, device=self.device)
            plot(lines, surface_id, z, x)

        if len(self.surfaces) == 1:  # if there is only one surface, then it has to be the aperture
            draw_aperture(lines, self.surfaces[0], 0)
        else:
            # draw sensor plane
            if with_sensor == True:
                try:
                    tmpr, tmpdd = self.r_last, self.d_sensor
                except AttributeError:
                    with_sensor = False

            if with_sensor:  # if with_sensor is True, then add a virtual sensor surface
                self.surfaces.append(Aspheric(self.r_last, self.d_sensor, 0.0))

            # draw surface
            for i, s in enumerate(self.surfaces):
                # find aperture
                if i < len(self.surfaces) - 1:
                    if self.materials[i].A < 1.0003 and self.materials[i + 1].A < 1.0003:  # both are AIR
                        draw_aperture(lines, s, i)
                        continue
                r = torch.linspace(-s.r, s.r, s.APERTURE_SAMPLING, device=self.device)
                z = s.surface_with_offset(r, torch.zeros(len(r), device=self.device))
                plot(lines, i, z, r)

            # draw boundary
            s_prev = []
            for i, s in enumerate(self.surfaces):
                if self.materials[i].A < 1.0003:  # AIR
                    s_prev = s
                else:
                    r_prev = s_prev.r
                    r = s.r
                    sag_prev = s_prev.surface_with_offset(r_prev, 0.0)
                    sag = s.surface_with_offset(r, 0.0)
                    z = torch.stack((sag_prev, sag))
                    x = torch.Tensor(np.array([r_prev, r])).to(self.device)
                    plot(lines, i, z, x)
                    plot(lines, i, z, -x)
                    s_prev = s

            # remove sensor plane
            if with_sensor:
                self.surfaces.pop()

        return lines

    def plot_setup2D(self, ax=None, fig=None, show=True, color='k', with_sensor=True):
        """
        Plot the layout of the lens group in 2D.
        """
        if ax is None and fig is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            show = False

        # to world coordinate
        def plot(ax, z, x, color):
            # p = self.to_world.transform_point(
            #     torch.stack(
            #         (x, torch.zeros_like(x, device=self.device), z), axis=-1
            #     )
            # ).cpu().detach().numpy()
            p = torch.stack(
                (x, torch.zeros_like(x, device=self.device), z), axis=-1
            ).cpu().detach().numpy()
            ax.plot(p[..., 2], p[..., 0], color)

        def draw_aperture(ax, surface, color):
            N = 3
            d = surface.d.cpu()
            R = surface.r
            APERTURE_WEDGE_LENGTH = 0.05 * R  # [mm]
            APERTURE_WEDGE_HEIGHT = 0.15 * R  # [mm]

            # wedge length
            z = torch.linspace(d - APERTURE_WEDGE_LENGTH, d + APERTURE_WEDGE_LENGTH, N, device=self.device)
            x = -R * torch.ones(N, device=self.device)
            plot(ax, z, x, color)
            x = R * torch.ones(N, device=self.device)
            plot(ax, z, x, color)

            # wedge height
            z = d * torch.ones(N, device=self.device)
            x = torch.linspace(R, R + APERTURE_WEDGE_HEIGHT, N, device=self.device)
            plot(ax, z, x, color)
            x = torch.linspace(-R - APERTURE_WEDGE_HEIGHT, -R, N, device=self.device)
            plot(ax, z, x, color)

        if len(self.surfaces) == 1:  # if there is only one surface, then it has to be the aperture
            draw_aperture(ax, self.surfaces[0], color)
        else:
            # draw sensor plane
            if with_sensor:
                try:
                    # if with_sensor is True, then add a virtual sensor surface
                    self.surfaces.append(Aspheric(self.r_last, self.d_sensor, 0.0))
                except AttributeError:
                    with_sensor = False

            # draw surface
            for i, s in enumerate(self.surfaces):
                # find aperture
                if i < len(self.surfaces) - 1:
                    if self.materials[i].A < 1.0003 and self.materials[i + 1].A < 1.0003:  # both are AIR
                        draw_aperture(ax, s, color)
                        continue
                r = torch.linspace(-s.r, s.r, s.APERTURE_SAMPLING, device=self.device)  # aperture sampling
                z = s.surface_with_offset(r, torch.zeros(len(r), device=self.device))
                plot(ax, z, r, color)

            # draw boundary
            s_prev = []
            for i, s in enumerate(self.surfaces):
                if self.materials[i].A < 1.0003:  # AIR
                    s_prev = s
                else:
                    r_prev = s_prev.r
                    r = s.r
                    sag_prev = s_prev.surface_with_offset(r_prev, 0.0)
                    sag = s.surface_with_offset(r, 0.0)
                    z = torch.stack((sag_prev, sag))
                    x = torch.Tensor(np.array([r_prev, r])).to(self.device)
                    plot(ax, z, x, color)
                    plot(ax, z, -x, color)
                    s_prev = s

            # remove sensor plane
            if with_sensor:
                self.surfaces.pop()

        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel('z [mm]')
        plt.ylabel('r [mm]')
        plt.title("Layout 2D")
        if show: plt.show()
        return ax, fig

    # TODO: modify the tracing part to include oss
    def plot_raytraces(self, oss, ax=None, fig=None, color='b-', show=True, p=None, valid_p=None):
        """
        Plot all ray traces (oss).
        """
        if ax is None and fig is None:
            ax, fig = self.plot_setup2D(show=False)
        else:
            show = False
        for i, os in enumerate(oss):
            o = torch.Tensor(np.array(os)).to(self.device)
            x = o[..., 0]
            z = o[..., 2]

            # to world coordinate
            # o = self.to_world.transform_point(
            #     torch.stack(
            #         (x, torch.zeros_like(x, device=self.device), z), axis=-1
            #     )
            # )
            o = o.cpu().detach().numpy()
            z = o[..., 2].flatten()
            x = o[..., 0].flatten()

            if p is not None and valid_p is not None:
                if valid_p[i]:
                    x = np.append(x, p[i, 0])
                    z = np.append(z, p[i, 2])

            ax.plot(z, x, color, linewidth=1.0)  # ax.plot can connect independent points

        if show:
            plt.show()
        else:
            plt.close()
        return ax, fig

    def plot_setup2D_with_trace(self, views, wavelength, M=2, R=None, entrance_pupil=True):
        if R is None:
            R = self.surfaces[0].r
        colors_list = 'bgrymck'
        ax, fig = self.plot_setup2D(show=False)

        for i, view in enumerate(views):
            ray = self.sample_ray_2D(R, wavelength, view=view, M=M, entrance_pupil=entrance_pupil)
            ps, oss = self.trace_to_sensor_r(ray)
            ax, fig = self.plot_raytraces(oss, ax=ax, fig=fig, color=colors_list[i])

        # fig.show()
        return ax, fig

    # ------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------------------------
    def calc_entrance_pupil(self, view=0.0, R=None):
        """
        Calculate and determine the position and size of entrance pupil.

        Args:
        - view (float): Field angle in degrees to determine beam direction
        - R (float): Radius of entrance pupil. If None, calculated from first lens surface

        Returns:
        - valid_map (torch.Tensor): Boolean array indicating valid rays
        - xs, ys (torch.Tensor): x,y coordinates of valid rays for determining pupil boundary

        Notes:
        - Computes entrance pupil size and position, critical for optical design
        - Uses grid sampling and ray tracing to determine which rays pass through pupil
        """
        angle = np.radians(np.asarray(view))

        # maximum radius input
        if R is None:
            with torch.no_grad():
                sag = self.surfaces[0].surface(self.surfaces[0].r, 0.0)  # calculate the thickness of the first surface
                R = np.tan(angle) * sag + self.surfaces[0].r  # [mm]
                # if the entrance pupil is not set, then calculate the entrance pupil based on the thickness of the first surface
                R = R.item()

        APERTURE_SAMPLING = 101
        x, y = torch.meshgrid(
            torch.linspace(-R, R, APERTURE_SAMPLING, device=self.device),
            torch.linspace(-R, R, APERTURE_SAMPLING, device=self.device),
            indexing='ij'
        )

        # generate rays and find valid map
        ones = torch.ones_like(x)
        zeros = torch.zeros_like(x)
        o = torch.stack((x, y, zeros), axis=2)
        d = torch.stack((
            np.sin(angle) * ones,
            zeros,
            np.cos(angle) * ones), axis=-1
        )
        ray = Ray(o, d, torch.Tensor([580.0]).to(self.device), device=self.device)
        valid_map = self.trace_valid(ray)

        # find bounding box
        xs, ys = x[valid_map], y[valid_map]

        return valid_map, xs, ys

    def sample_ray(self, wavelength, view=0.0, M=15, R=None, shift_x=0., shift_y=0., sampling='grid', random=None,
                   entrance_pupil=False, valid=False):
        """
        Generate a set of sample rays in the optical system.
        """
        angle = np.radians(np.asarray(view))

        # maximum radius input
        if R is None:
            with torch.no_grad():
                sag = self.surfaces[0].surface(self.surfaces[0].r, 0.0)
                R = np.tan(angle) * sag + self.surfaces[0].r  # [mm]
                R = R.item()

        if entrance_pupil:
            xs, ys = self.calc_entrance_pupil(view, R)[1:]
            if sampling == 'grid':
                x, y = torch.meshgrid(
                    torch.linspace(xs.min(), xs.max(), M, device=self.device),
                    torch.linspace(ys.min(), ys.max(), M, device=self.device),
                    indexing='ij'
                )
            elif sampling == 'radial':
                R = np.minimum((xs.max() - xs.min()).cpu().numpy(), (ys.max() - ys.min()).cpu().numpy())
                r = torch.linspace(0, R, M, device=self.device)
                theta = torch.linspace(0, 2 * np.pi, M + 1, device=self.device)[0:M]
                x = xs.mean() + r[None, ...] * torch.cos(theta[..., None])
                y = ys.mean() + r[None, ...] * torch.sin(theta[..., None])
        else:
            if sampling == 'grid':
                x, y = torch.meshgrid(
                    torch.linspace(-R, R, M, device=self.device),
                    torch.linspace(-R, R, M, device=self.device),
                    indexing='ij'
                )
            elif sampling == 'radial':
                r = torch.linspace(0, R, M, device=self.device)
                theta = torch.linspace(0, 2 * np.pi, M + 1, device=self.device)[0:M]
                x = r[None, ...] * torch.cos(theta[..., None])
                y = r[None, ...] * torch.sin(theta[..., None])

        p = 2 * R / M
        x = x + p * shift_x
        y = y + p * shift_y

        if random:
            x = x + p * (torch.rand(M, M, device=x.device) - 0.5)
            y = y + p * (torch.rand(M, M, device=x.device) - 0.5)

        o = torch.stack((x, y, torch.zeros_like(x, device=self.device)), axis=2)
        d = torch.stack((
            np.sin(angle) * torch.ones_like(x),
            torch.zeros_like(x),
            np.cos(angle) * torch.ones_like(x)), axis=-1
        )

        # if valid is True, then filter out the rays that satisfy the condition
        if valid:
            valid_indices = x.pow(2) + y.pow(2) < R ** 2
            o = o[valid_indices]
            d = d[valid_indices]

        return Ray(o, d, wavelength, device=self.device)

    def sample_ray_2D(self, R, wavelength, view=0.0, M=15, shift_x=0., entrance_pupil=False):
        """
        Generate a set of sample rays in 2D.
        This is a 2D version of sample_ray.
        """
        if entrance_pupil:
            # x_up, x_down, x_center = self.find_ray_2D(view=view)
            xs = self.calc_entrance_pupil(view=view)[1]
            x_up = xs.min()
            x_down = xs.max()
            x_center = xs.mean()

            x = torch.hstack((
                torch.linspace(x_down, x_center, M + 1, device=self.device)[:M],
                torch.linspace(x_center, x_up, M + 1, device=self.device),
            ))
        else:
            x = torch.linspace(-R, R, M, device=self.device)
        p = 2 * R / M
        x = x + p * shift_x

        ones = torch.ones_like(x)
        zeros = torch.zeros_like(x)

        o = torch.stack((x, zeros, zeros), axis=1)
        angle = torch.Tensor(np.asarray(np.radians(view))).to(self.device)
        d = torch.stack((
            torch.sin(angle) * ones,
            zeros,
            torch.cos(angle) * ones), axis=-1
        )
        return Ray(o, d, wavelength, device=self.device)

    # def find_ray_2D(self, view=0.0, y=0.0):
    #     wavelength = torch.Tensor([589.3]).to(self.device)
    #     R_aperture = self.surfaces[self.aperture_ind].r
    #     angle = np.radians(view)
    #     d = torch.Tensor(np.stack((
    #         np.sin(angle),
    #         y,
    #         np.cos(angle)), axis=-1
    #     )).to(self.device)
    #
    #     def find_x(alpha=1.0): # TODO: does not work for wide-angle lenses!
    #         x = - np.tan(angle) * self.surfaces[self.aperture_ind].d.cpu().detach().numpy()
    #         is_converge = False
    #         for k in range(30):
    #             o = torch.Tensor([x, y, 0.0])
    #             ray = Ray(o, d, wavelength, device=self.device)
    #             ray_final, valid = self.trace(ray, stop_ind=self.aperture_ind)[:2]
    #             x_aperture = ray_final.o[0].cpu().detach().numpy()
    #             diff = 0.0 - x_aperture
    #             if np.abs(diff) < 0.001:
    #                 print('`find_x` converges!')
    #                 is_converge = True
    #                 break
    #             if valid:
    #                 x_last = x
    #                 if diff > 0.0:
    #                     x += alpha * diff
    #                 else:
    #                     x -= alpha * diff
    #             else:
    #                 x = (x + x_last)/2
    #         return x, is_converge
    #
    #     def find_bx(x_center, R_aperture, alpha=1.0):
    #         x = x_center
    #         x_last = 0.0 # temp
    #         for k in range(100):
    #             o = torch.Tensor([x, y, 0.0])
    #             ray = Ray(o, d, wavelength, device=self.device)
    #             ray_final, valid = self.trace(ray, stop_ind=self.aperture_ind)[:2]
    #             x_aperture = ray_final.o[0].cpu().detach().numpy()
    #             diff = R_aperture - x_aperture
    #             if np.abs(diff) < 0.01:
    #                 print('`find_x` converges!')
    #                 break
    #             if valid:
    #                 x_last = x
    #                 if diff > 0.0:
    #                     x += alpha * diff
    #                 else:
    #                     x -= alpha * diff
    #             else:
    #                 x = (x + x_last)/2
    #         return x_last
    #
    #     x_center, is_converge = find_x(alpha=-np.sign(view)*1.0)
    #     if not is_converge:
    #         x_center, is_converge = find_x(alpha=np.sign(view)*1.0)
    #
    #     x_up = find_bx(x_center, R_aperture, alpha=1)
    #     x_down = find_bx(x_center, -R_aperture, alpha=-1)
    #     return x_up, x_down, x_center

    # ------------------------------------------------------------------------------------

    def render(self, ray, irr=1.0):
        """
        Forward rendering.
        """
        # TODO: remind users to prepare filmsize and pixelsize before using this function.

        # trace rays
        ray_final, valid = self.trace(ray)

        # intersecting sensor plane
        t = (self.d_sensor - ray_final.o[..., 2]) / ray_final.d[..., 2]
        p = ray_final(t)

        R_sensor = [self.film_size[i] * self.pixel_size / 2 for i in range(2)]  # R_sensor 是带长度单位的
        valid = valid & (
                (-R_sensor[0] <= p[..., 0]) & (p[..., 0] <= R_sensor[0]) &
                (-R_sensor[1] <= p[..., 1]) & (p[..., 1] <= R_sensor[1])
        )

        # intensity
        J = irr
        p = p[valid]

        # compute shift and find nearest pixel index
        u = (p[..., 0] + R_sensor[0]) / self.pixel_size
        v = (p[..., 1] + R_sensor[1]) / self.pixel_size

        index_l = torch.stack(
            (torch.clamp(torch.floor(u).long(), min=0, max=self.film_size[0] - 1),
             torch.clamp(torch.floor(v).long(), min=0, max=self.film_size[1] - 1)),
            axis=-1
        )
        index_r = torch.stack(
            (torch.clamp(index_l[..., 0] + 1, min=0, max=self.film_size[0] - 1),
             torch.clamp(index_l[..., 1] + 1, min=0, max=self.film_size[1] - 1)),
            axis=-1
        )
        w_r = torch.clamp(torch.stack((u, v), axis=-1) - index_l, min=0, max=1)
        w_l = 1.0 - w_r
        del u, v

        # compute image
        I = torch.zeros(*self.film_size, device=self.device)
        I = torch.index_put(I, (index_l[..., 0], index_l[..., 1]), w_l[..., 0] * w_l[..., 1] * J, accumulate=True)
        I = torch.index_put(I, (index_r[..., 0], index_l[..., 1]), w_r[..., 0] * w_l[..., 1] * J, accumulate=True)
        I = torch.index_put(I, (index_l[..., 0], index_r[..., 1]), w_l[..., 0] * w_r[..., 1] * J, accumulate=True)
        I = torch.index_put(I, (index_r[..., 0], index_r[..., 1]), w_r[..., 0] * w_r[..., 1] * J, accumulate=True)
        return I

    def trace_valid(self, ray):
        """
        Trace rays to see if they intersect the sensor plane or not.
        """
        valid = self.trace(ray)[1]
        return valid

    def trace_to_sensor(self, ray, ignore_invalid=False, intersect2axis=False):
        """
        Trace rays towards intersecting onto the sensor plane.
        返回光线与传感器平面的交点。
        """
        # trace rays
        ray_final, valid = self.trace(ray)

        if not intersect2axis:
            # intersecting sensor plane
            t = (self.d_sensor - ray_final.o[..., 2]) / ray_final.d[..., 2]
            p = ray_final(t)  # p represents the intersection point, which is the intersection point between the ray and the sensor
            if ignore_invalid:
                p = p[valid]
            else:
                if len(p.shape) < 2:
                    return p
                p = torch.reshape(p, (np.prod(p.shape[:-1]), 3))
        else:
            if intersect2axis == 'x':
                t = (0 - ray_final.o[..., 0]) / ray_final.d[..., 0]
                p = ray_final(t)
            elif intersect2axis == 'y':
                t = (0 - ray_final.o[..., 1]) / ray_final.d[..., 1]
                p = ray_final(t)
            else:
                raise ValueError("intersect2axis should be 'x' or 'y'.")
        return p

    def trace_to_sensor_r(self, ray, ignore_invalid=False):
        """
        Trace rays towards intersecting onto the sensor plane, with records.
        Compared to the trace_to_sensor function, it not only records the intersection point between the ray and the sensor plane, but also records all the points through which the ray passes during the propagation process.
        """
        # trace rays
        ray_final, valid, oss = self.trace_r(ray)

        # intersecting sensor plane
        t = (self.d_sensor - ray_final.o[..., 2]) / ray_final.d[..., 2]
        p = ray_final(t)
        if ignore_invalid:
            p = p[valid]
        else:
            p = torch.reshape(p, (np.prod(p.shape[:-1]), 3))

        for v, os, pp in zip(valid, oss, p):
            if v:
                os.append(pp.cpu().detach().numpy())

        return p, oss

    def trace(self, ray, stop_ind=None):
        # update transformation when doing pose estimation
        if (
                self.origin.requires_grad
                or
                self.shift.requires_grad
                or
                self.theta_x.requires_grad
                or
                self.theta_y.requires_grad
                or
                self.theta_z.requires_grad
        ):
            self.update()

        ray_in = ray

        valid, ray_out = self._trace(ray_in, stop_ind=stop_ind, record=False)

        ray_final = ray_out

        # # in world
        # ray_final = self.to_world.transform_ray(ray_out)

        return ray_final, valid

    def trace_r(self, ray, stop_ind=None):
        """
        Compared to the trace function, it returns an additional oss parameter, which is used to record all the points through which the ray passes during the propagation process.
        """
        # update transformation when doing pose estimation
        if (
                self.origin.requires_grad
                or
                self.shift.requires_grad
                or
                self.theta_x.requires_grad
                or
                self.theta_y.requires_grad
                or
                self.theta_z.requires_grad
        ):
            self.update()

        # in local
        # ray_in = self.to_object.transform_ray(ray)

        ray_in = ray

        valid, ray_out, oss = self._trace(ray_in, stop_ind=stop_ind, record=True)
        
        ray_final = ray_out

        # in world
        # ray_final = self.to_world.transform_ray(ray_out)
        # for os in oss:
        #     for o in os:
        #         os = self.to_world.transform_point(torch.Tensor(np.asarray(os)).to(self.device)).cpu().detach().numpy()

        return ray_final, valid, oss

    # ------------------------------------------------------------------------------------
    # Rendering  backward rendering
    # ------------------------------------------------------------------------------------
    def prepare_mts(self, pixel_size, film_size, R=np.eye(3), t=np.zeros(3)):
        # TODO: this is actually prepare_backward tracing ...
        """
        Revert surfaces for Mitsuba2 rendering.
        """
        if self.mts_prepared:
            print('MTS already prepared for this lensgroup.')
            return

        # sensor parameters
        self.pixel_size = pixel_size  # [mm]
        self.film_size = film_size  # [pixel]

        # rendering parameters
        self.mts_Rt = Transformation(R, t)  # transformation of the lensgroup
        self.mts_Rt.to(self.device)

        # for visualization
        self.r_last = self.pixel_size * max(self.film_size) / 2

        # TODO: could be further optimized:
        # treat the lenspart as a camera; append one more surface to it
        self.surfaces.append(Aspheric(self.r_last, self.d_sensor, 0.0))

        # reverse surfaces
        d_total = self.surfaces[-1].d
        for i in range(len(self.surfaces)):
            self.surfaces[i].d = d_total - self.surfaces[i].d
            self.surfaces[i].reverse()
        self.surfaces.reverse()
        self.surfaces.pop(0)  # remove sensor plane

        # reverse materials
        self.materials.reverse()

        # aperture plane (TODO: could be optimized further to trace pupil positions)
        self.aperture_radius = self.surfaces[0].r
        self.aperture_distance = self.surfaces[0].d
        self.mts_prepared = True
        self.d_sensor = 0

    def _generate_sensor_samples(self):
        sX, sY = np.meshgrid(
            np.linspace(0, 1, self.film_size[0]),
            np.linspace(0, 1, self.film_size[1])
        )
        return np.stack((sX.flatten(), sY.flatten()), axis=1)

    def _generate_aperture_samples(self):
        Dx = np.random.rand(*self.film_size)
        Dy = np.random.rand(*self.film_size)
        [px, py] = Sampler().concentric_sample_disk(Dx, Dy)
        return np.stack((px.flatten(), py.flatten()), axis=1)

    def sample_ray_sensor_pinhole(self, wavelength, focal_length):
        """
        Sample ray on the sensor plane, assuming a pinhole camera model, given a focal length.
        """
        if not self.mts_prepared:
            raise Exception('MTS unprepared; please call `prepare_mts()` first!')

        N = np.prod(self.film_size)

        # sensor and aperture plane samplings
        sample2 = self._generate_sensor_samples()

        # wavelength [nm]
        wavelength = torch.Tensor(wavelength * np.ones(N))

        # normalized to [-0,5, 0.5]
        sample2 = sample2 - 0.5

        # sample sensor and aperture planes
        p_sensor = sample2 * np.array([
            self.pixel_size * self.film_size[0], self.pixel_size * self.film_size[1]
        ])[None, :]

        # aperture samples (last surface plane)
        p_aperture = 0
        d_xy = p_aperture - p_sensor

        # construct ray
        o = torch.Tensor(np.hstack((p_sensor, np.zeros((N, 1)))).reshape((N, 3)))
        d = torch.Tensor(np.hstack((d_xy, focal_length * np.ones((N, 1)))).reshape((N, 3)))
        d = normalize(d)

        ray = Ray(o, d, wavelength, device=self.device)
        valid = torch.ones(ray.o[..., 2].shape, device=self.device).bool()
        return valid, ray

    def sample_ray_sensor(self, wavelength, offset=np.zeros(2)):
        """
        Sample rays on the sensor plane.
        """
        if not self.mts_prepared:
            raise Exception('MTS unprepared; please call `prepare_mts()` first!')

        N = np.prod(self.film_size)

        # sensor and aperture plane samplings
        sample2 = self._generate_sensor_samples()
        sample3 = self._generate_aperture_samples()

        # wavelength [nm]
        wav = wavelength * np.ones(N)

        # sample ray
        valid, ray = self._sample_ray_render(N, wav, sample2, sample3, offset)
        ray_new = self.mts_Rt.transform_ray(ray)
        return valid, ray_new

    def _sample_ray_render(self, N, wav, sample2, sample3, offset):
        """
        `offset`: sensor position offsets [mm].
        """

        # sample2 \in [ 0, 1]^2
        # sample3 \in [-1, 1]^2
        if not self.mts_prepared:
            raise Exception('MTS unprepared; please call `prepare_mts()` first!')

        # normalized to [-0,5, 0.5]
        sample2 = sample2 - 0.5

        # sample sensor and aperture planes
        p_sensor = sample2 * np.array([
            self.pixel_size * self.film_size[0], self.pixel_size * self.film_size[1]
        ])[None, :]

        # perturb sensor position by half pixel size
        p_sensor = p_sensor + (np.random.rand(*p_sensor.shape) - 0.5) * self.pixel_size

        # offset sensor positions
        p_sensor = p_sensor + offset

        # aperture samples (last surface plane)
        p_aperture = sample3 * self.aperture_radius
        d_xy = p_aperture - p_sensor

        # construct ray
        o = torch.Tensor(np.hstack((p_sensor, np.zeros((N, 1)))).reshape((N, 3)))
        d = torch.Tensor(np.hstack((d_xy, self.aperture_distance.item() * np.ones((N, 1)))).reshape((N, 3)))
        d = normalize(d)
        wavelength = torch.Tensor(wav)

        # trace
        valid, ray = self._trace(Ray(o, d, wavelength, device=self.device))
        return valid, ray

    # ------------------------------------------------------------------------------------

    def _refract(self, wi, n, eta, approx=False, General_Snell=False, surface=None, wavelength=None, ni=None, nt=None, p=None):
        if type(eta) is float:
            eta_ = eta
        else:
            if np.prod(eta.shape) > 1:
                eta_ = eta[..., None]
            else:
                eta_ = eta

        cosi = torch.sum(wi * n, axis=-1)

        if General_Snell:
            """
            Implementation of generalized Snell's law
            """
            if surface == None:
                raise Exception('when General_Snell, surface can not be None!')

            # calculate the phase derivative dφ/dr
            phase_derivatives = surface.define_phase_spatial_derivative(p, wavelength)

            x = p[..., 0]
            y = p[..., 1]
            r = torch.sqrt(x**2 + y**2 + 1e-8)
            
            # handle the center point
            mask = (x == 0) & (y == 0)
            
            # phase gradient vector ∇φ = (dφ/dr) * (x/r, y/r, 0) - in the local coordinate system of the surface
            grad_phi = torch.stack([
                phase_derivatives * x / r,
                phase_derivatives * y / r,
                torch.zeros_like(x)
            ], dim=-1)
            
            # handle the center point
            grad_phi[mask, :] = torch.tensor([0., 0., 0.], device=self.device)

            wavelength_mm = wavelength * 1e-6
        
            # according to the generalized Snell's law: n_t * sin(θ_t) * t̂ = n_i * sin(θ_i) * î + (λ/2π) * ∇φ
            # where t̂ and î are the tangential unit vectors
            
            # calculate the tangential component of the incident light (remove the normal component)
            wi_tangential = wi - cosi[..., None] * n
            
            # the tangential component equation of the generalized Snell's law
            wt_tangential = (ni / nt)[..., None] * wi_tangential + wavelength_mm / (2 * np.pi * nt)[..., None] * grad_phi
            
            # calculate the normal component (ensure unit length)
            wt_tangential_norm_sq = torch.sum(wt_tangential**2, dim=-1)
            wt_normal_norm_sq = 1 - wt_tangential_norm_sq
            
            # check total reflection
            valid = wt_normal_norm_sq > 0
            wt_normal_norm_sq = torch.clamp(wt_normal_norm_sq, min=1e-8)
            wt_normal_norm = torch.sqrt(wt_normal_norm_sq)
            
            # construct the complete outgoing direction (positive sign, assuming transmission)
            wt = wt_tangential + wt_normal_norm[..., None] * n
            
            # handle special cases
            if mask.any():
                # use the traditional Snell's law at the center point
                cost2 = 1. - (1. - cosi[mask] ** 2) * eta ** 2
                valid_center = cost2 > 0
                cost2 = torch.clamp(cost2, min=1e-8)
                tmp = torch.sqrt(cost2)
                wt_center = tmp[..., None] * n[mask] + eta * (wi[mask] - cosi[mask][..., None] * n[mask])
                wt[mask] = wt_center
                valid[mask] = valid_center
            
        else:
            # traditional Snell's law
            if approx:
                tmp = 1. - eta ** 2 * (1. - cosi)
                valid = tmp > 0.
                wt = tmp[..., None] * n + eta_ * (wi - cosi[..., None] * n)
            else:
                cost2 = 1. - (1. - cosi ** 2) * eta ** 2
                valid = cost2 > 0.
                cost2 = torch.clamp(cost2, min=1e-8)
                tmp = torch.sqrt(cost2)
                wt = tmp[..., None] * n + eta_ * (wi - cosi[..., None] * n)
        
        return valid, wt

    def _trace(self, ray, stop_ind=None, record=False):
        if stop_ind is None:
            stop_ind = len(self.surfaces) - 1  # last index to stop
        is_forward = (ray.d[..., 2] > 0).all()

        # TODO: Check ray origins to ensure valid ray intersections onto the surfaces
        if is_forward:
            return self._forward_tracing(ray, stop_ind, record)
        else:
            return self._backward_tracing(ray, stop_ind, record)

    def _forward_tracing(self, ray, stop_ind, record):
        wavelength = ray.wavelength
        dim = ray.o[..., 2].shape

        if record:
            oss = []
            for i in range(dim[0]):
                oss.append([ray.o[i, :].cpu().detach().numpy()])  # oss records the o point of each ray

        valid = torch.ones(dim, device=self.device).bool()
        for i in range(stop_ind + 1):
            # through the loop to traverse the intersection coordinates of each optical surface, finally, the path of each ray is in an os list, and finally, the path of all rays is merged into the oss list
            # oss {list}: ray1 os1 {list}, ray2 os2 {list} ...
            eta = self.materials[i].ior(wavelength) / self.materials[i + 1].ior(wavelength)
            ni = self.materials[i].ior(wavelength)
            nt = self.materials[i + 1].ior(wavelength)

            # ray intersecting surface
            valid_o, p = self.surfaces[i].ray_surface_intersection(ray, valid)

            # get surface normal and refract 
            n = self.surfaces[i].normal(p[..., 0], p[..., 1])
            if self.surfaces[i].General_Snell:
                valid_d, d = self._refract(ray.d, -n, eta, General_Snell=True, surface=self.surfaces[i],
                                           wavelength=wavelength, ni=ni, nt=nt, p=p)
            else:
                valid_d, d = self._refract(ray.d, -n, eta)

            # check validity
            valid = valid & valid_o & valid_d
            if not valid.any():
                break

            # update ray {o,d}
            if record:  # TODO: make it pythonic ...
                for os, v, pp in zip(oss, valid.cpu().detach().numpy(), p.cpu().detach().numpy()):
                    if v:
                        os.append(pp)  # os updates the o point of each ray, and oss also updates with os
            ray.o = p
            ray.d = d

        if record:
            return valid, ray, oss
        else:
            return valid, ray

    def _backward_tracing(self, ray, stop_ind, record):
        wavelength = ray.wavelength
        dim = ray.o[..., 2].shape

        if record:
            oss = []
            for i in range(dim[0]):
                oss.append([ray.o[i, :].cpu().detach().numpy()])

        valid = torch.ones(dim, device=ray.o.device).bool()
        for i in np.flip(range(stop_ind + 1)):
            surface = self.surfaces[i]
            eta = self.materials[i + 1].ior(wavelength) / self.materials[i].ior(
                wavelength)  # the only difference between _backward_tracing and _forward_tracing is this line

            # ray intersecting surface
            valid_o, p = surface.ray_surface_intersection(ray, valid)

            # get surface normal and refract 
            n = surface.normal(p[..., 0], p[..., 1])
            valid_d, d = self._refract(ray.d, n, eta)  # backward: no need to revert the normal

            # check validity
            valid = valid & valid_o & valid_d
            if not valid.any():
                break

            # update ray {o,d}
            if record:  # TODO: make it pythonic ...
                for os, v, pp in zip(oss, valid.numpy(), p.cpu().detach().numpy()):
                    if v:
                        os.append(pp)
            ray.o = p
            ray.d = d

        if record:
            return valid, ray, oss
        else:
            return valid, ray


class Surface(PrettyPrinter):
    """
    This is the base class for optical surfaces.

    The surface is parameterized as an implicit function f(x,y,z) = 0.
    For simplicity, we assume the surface function f(x,y,z) can be decomposed as:
    
    f(x,y,z) = g(x,y) + h(z),

    where g(x,y) and h(z) are explicit functions to be defined in sub-classes.

    Args:
        r: Radius of the aperture (default to be circular, unless specified as square).
        d: Distance of z-direction in global coordinate
        is_square: is the aperture square
        device: Torch device
    """

    def __init__(self, r, d, is_square=False, device=torch.device('cpu')):
        if torch.is_tensor(d):
            self.d = d
        else:
            self.d = torch.Tensor(np.asarray(float(d))).to(device)
        self.is_square = is_square
        self.r = float(r)
        self.device = device

        # There are the parameters controlling the accuracy of ray tracing.
        self.NEWTONS_MAXITER = 10
        self.NEWTONS_TOLERANCE_TIGHT = 50e-6  # in [mm], i.e. 50 [nm] here (up to <10 [nm])
        self.NEWTONS_TOLERANCE_LOOSE = 300e-6  # in [mm], i.e. 300 [nm] here (up to <10 [nm])
        self.APERTURE_SAMPLING = 257

        self.General_Snell = False  # whether to use the generalized Snell's law to calculate the refraction of this surface, default is False

    # === Common methods (must not be overridden)
    def surface_with_offset(self, x, y):
        """
        Returns the z coordinate plus the surface's own distance; useful when drawing the surfaces.
        """
        return self.surface(x, y) + self.d

    def normal(self, x, y):
        """
        Returns the 3D normal vector of the surface at 2D coordinate (x,y), in local coordinate.
        """
        ds_dxyz = self.surface_derivatives(x, y)
        return normalize(torch.stack(ds_dxyz, axis=-1))

    def surface_area(self):
        """
        Computes the surface's area.
        """
        if self.is_square:
            return self.r ** 2
        else:  # is round
            return np.pi * self.r ** 2

    def mesh(self):
        """
        Generates a 2D meshgrid for the current surface.
        """
        x, y = torch.meshgrid(
            torch.linspace(-self.r, self.r, self.APERTURE_SAMPLING, device=self.device),
            torch.linspace(-self.r, self.r, self.APERTURE_SAMPLING, device=self.device),
            indexing='ij'
        )
        valid_map = self.is_valid(torch.stack((x, y), axis=-1))
        return self.surface(x, y) * valid_map

    def sdf_approx(self, p):
        """
        (Approximated) Signed Distance Function (SDF) of a 2D point p to the surface's aperture boundary.
        If:
        - Returns < 0: p is within the surface's aperture.
        - Returns = 0: p is at the surface's aperture boundary.
        - Returns > 0: p is outside of the surface's aperture.

        Args:
            p: Local 2D point.
        
        Returns:
            A SDF mask.
        """
        if self.is_square:
            return torch.max(torch.abs(p) - self.r, axis=-1)[0]
        else:  # is round
            return length2(p) - self.r ** 2

    def is_valid(self, p):
        """
        If a 2D point p is valid, i.e. if p is within the surface's aperture.
        """
        return (self.sdf_approx(p) < 0.0).bool()

    def ray_surface_intersection(self, ray, active=None):
        """
        Computes ray-surface intersection, one of the most crucial functions in this class.
        Given ray(s) and an activity mask, the function computes the intersection point(s),
        and determines if the intersection is valid, and update the active mask accordingly. 
        
        Args:
            ray: Rays.
            active: The initial active mask.

        Returns:
            valid_o: The updated active mask (if the current ray is physically active in tracing).
            local: The computed intersection point(s).
        """
        solution_found, local = self.newtons_method(ray.maxt, ray.o, ray.d)

        valid_o = solution_found & self.is_valid(local[..., 0:2])
        if active is not None:
            valid_o = active & valid_o
        return valid_o, local

    def newtons_method(self, maxt, o, D, option='implicit'):
        """
        Newton's method to find the root of the ray-surface intersection point.
        
        Two modes are supported here:

        1. 'explicit": This implements the loop using autodiff, and gradients will be
        accurate for o, D, and self.parameters. Slow and memory-consuming.
        
        2. 'implicit": This implements the loop using implicit-layer theory, find the 
        solution without autodiff, then hook up the gradient. Less memory-consuming.

        Args:
            maxt: The maximum travel distance of a single ray.
            o: The origins of the rays.
            D: The directional vector of the rays.
            option: The computing modes.

        Returns:
            valid: The updated active mask (if the current ray is physically active in tracing).
            p: The computed intersection point(s).
        """

        # pre-compute constants
        ox, oy, oz = (o[..., i].clone() for i in range(3))
        dx, dy, dz = (D[..., i].clone() for i in range(3))
        A = dx ** 2 + dy ** 2
        B = 2 * (dx * ox + dy * oy)
        C = ox ** 2 + oy ** 2

        # initial guess of t
        t0 = (self.d - oz) / dz

        if option == 'explicit':
            t, t_delta, valid = self.newtons_method_impl(
                maxt, t0, dx, dy, dz, ox, oy, oz, A, B, C
            )
        elif option == 'implicit':
            with torch.no_grad():
                t, t_delta, valid = self.newtons_method_impl(
                    maxt, t0, dx, dy, dz, ox, oy, oz, A, B, C
                )
                s_derivatives_dot_D = self.surface_and_derivatives_dot_D(
                    t, dx, dy, dz, ox, oy, t_delta * dz, A, B, C
                )[1]
            t = t0 + t_delta
            t = t - (self.g(ox + t * dx, oy + t * dy) + self.h(oz + t * dz) + self.d) / s_derivatives_dot_D
        else:
            raise Exception('option={} is not available!'.format(option))

        p = o + t[..., None] * D

        return valid, p

    def newtons_method_impl(self, maxt, t0, dx, dy, dz, ox, oy, oz, A, B, C):
        """
        The actual implementation of Newton's method.

        Args:
            dx,dy,dx,ox,oy,oz,A,B,C: Variables to a quadratic problem.
        
        Returns:
            t: The travel distance of the ray.
            t_delta: The incremental change of t at each iteration.
            valid: The updated active mask (if the current ray is physically active in tracing).
        """
        if oz.numel() < 2:
            oz = torch.Tensor([oz.item()]).to(self.device)
        t_delta = torch.zeros_like(oz)

        # iterate until the intersection error is small
        t = maxt * torch.ones_like(oz)
        residual = maxt * torch.ones_like(oz)
        it = 0
        while (torch.abs(residual) > self.NEWTONS_TOLERANCE_TIGHT).any() and (it < self.NEWTONS_MAXITER):
            it += 1
            t = t0 + t_delta
            residual, s_derivatives_dot_D = self.surface_and_derivatives_dot_D(
                t, dx, dy, dz, ox, oy, t_delta * dz, A, B, C  # here z = t_delta * dz
            )
            t_delta = t_delta - residual / s_derivatives_dot_D
        t = t0 + t_delta
        valid = (torch.abs(residual) < self.NEWTONS_TOLERANCE_LOOSE) & (t <= maxt)
        return t, t_delta, valid

    # === Virtual methods (must be overridden)
    def g(self, x, y):
        """
        Function g(x,y).

        Args:
            x: The x local coordinate.
            y: The y local coordinate.

        Returns:
            g(x,y): Function g(x,y) at (x,y).
        """
        raise NotImplementedError()

    def dgd(self, x, y):
        """
        Derivatives of g: (dg/dx, dg/dy).

        Args:
            x: The x local coordinate.
            y: The y local coordinate.

        Returns:
            dg/dx: dg/dx of function g(x,y) at (x,y).
            dg/dy: dg/dy of function g(x,y) at (x,y).
        """
        raise NotImplementedError()

    def h(self, z):
        """
        Function h(z).

        Args:
            z: The z local coordinate.

            
        Returns:
            h(z): Function h(z) at z.
        """
        raise NotImplementedError()

    def dhd(self, z):
        """
        Derivatives of h: dh/dz.

        Args:
            z: The z local coordinate.

        Returns:
            dh/dz: dh/dz of function h(z) at z.
        """
        raise NotImplementedError()

    def surface(self, x, y):
        """
        Solve z from h(z) = -g(x,y).
        
        Args:
            x: The x local coordinate.
            y: The y local coordinate.

        Returns:
            z: Surface's z coordinate.
        """
        raise NotImplementedError()

    def reverse(self):
        raise NotImplementedError()

    # === Default methods (better be overridden)
    def surface_derivatives(self, x, y):
        """
        Computes the surface's spatial derivatives:
        
        Assume the surface height function f(x,y,z) = g(x,y) + h(z). The spatial derivatives are:
        
        \nabla f = \nabla (g(x,y) + h(z)) = (dg/dx, dg/dy, dh/dz).
        
        (Note: this default implementation is not efficient)
        
        Args:
            x: The x local coordinate.
            y: The y local coordinate.

        Returns:
            gx: dg/dx.
            gy: dg/dy.
            hz: dh/dz.
        """
        gx, gy = self.dgd(x, y)
        z = self.surface(x, y)
        return gx, gy, self.dhd(z)

    def surface_and_derivatives_dot_D(self, t, dx, dy, dz, ox, oy, z, A, B, C):
        """
        Computes the surface and the dot product of its spatial derivatives and ray direction.

        Assume the surface height function f(x,y,z) = g(x,y) + h(z). The outputs are:
        
        g(x,y) + h(z)  and  (dg/dx, dg/dy, dh/dz) \cdot (dx,dy,dz).

        (Note: this default implementation is not efficient)

        Args:
            t: The travel distance of the considered ray(s).
            dx,dy,dx,ox,oy,oz,A,B,C: Variables to a quadratic problem.

        Returns:
            s: Value of f(x,y,z). The intersection is at the surface if s equals zero.
            sx*dx + sy*dy + sz*dz: The dot product between the surface's spatial derivatives and ray direction d.
        """
        x = ox + t * dx
        y = oy + t * dy
        s = self.g(x, y) + self.h(z)
        sx, sy = self.dgd(x, y)
        sz = self.dhd(z)
        return s, sx * dx + sy * dy + sz * dz


class MetaSurface(Surface):
    """
    MetaSurface基类，用于定义metasurface的基本属性和方法。
    """

    def __init__(self, r, d, c=0., is_square=False, device=torch.device('cpu')):
        Surface.__init__(self, r, d, is_square, device)
        self.c = torch.tensor(c, device=device)
        self.period = torch.tensor(350e-6, device=device)
        self.General_Snell = True  # Metasurface needs to use the generalized Snell's law

        num_pillars = int(self.r / self.period)  # calculate the number of nanopillars
        self.number_pillars = num_pillars

        # determine the center position of the nanopillar
        r_positions = (torch.arange(num_pillars, device=device) + 0.5) * self.period
        self.r_positions = r_positions

    # === Common methods
    def g(self, x, y):
        return self._g(x ** 2 + y ** 2)

    def dgd(self, x, y):
        dsdr2 = 2 * self._dgd(x ** 2 + y ** 2)
        return dsdr2 * x, dsdr2 * y

    def h(self, z):
        return -z

    def dhd(self, z):
        return -torch.ones_like(z)

    def surface(self, x, y):
        return self._g(x ** 2 + y ** 2)

    def reverse(self):
        self.c = -self.c

    def surface_derivatives(self, x, y):
        dsdr2 = 2 * self._dgd(x ** 2 + y ** 2)
        return dsdr2 * x, dsdr2 * y, -torch.ones_like(x)

    def surface_and_derivatives_dot_D(self, t, dx, dy, dz, ox, oy, z, A, B, C):
        # pylint: disable=unused-argument
        # TODO: could be further optimized
        r2 = A * t ** 2 + B * t + C
        return self._g(r2) - z, self._dgd(r2) * (2 * A * t + B) - dz

    def calculate_complex_amplitude_derivatives(self, r, wavelength):
        """
        Calculate the phase derivative along the radial direction of the metasurface.
        This method needs to be implemented in subclasses.

        Args:
            wavelength: wavelength of the light wave.

        Returns:
            phase, phase derivative, etc.
        """
        raise NotImplementedError()

    def define_phase_spatial_derivative(self, p, wavelength):
        """
        Calculate the phase spatial gradient at the current metasurface position.

        Args:
            p: the coordinates of the current ray.
            wavelength: wavelength of the light wave.

        Returns:
            the derivative of the phase.
        """
        raise NotImplementedError()

    def unwrap_phase(self, phase):
        """
        Manually implement the function of phase unwrapping.

        Args:
            phase: the input phase tensor.

        Returns:
            the unwrapped phase tensor.
        """
        device = phase.device
        pi = torch.tensor(torch.pi, device=device)

        # calculate the phase difference between adjacent elements
        phase_diff = torch.diff(phase)

        # detect phase jumps and calculate the adjustment value
        # when the phase difference is greater than π, subtract 2π; when the phase difference is less than -π, add 2π
        jumps = torch.where(phase_diff > pi, -2 * pi,
                            torch.where(phase_diff < -pi, 2 * pi, torch.tensor(0.0, device=device)))

        # accumulate phase adjustment
        phase_adjust = torch.cumsum(jumps, dim=0)

        # apply phase adjustment
        unwrapped_phase = phase + torch.cat([torch.zeros(1, device=device), phase_adjust])

        return unwrapped_phase

    # === Private methods
    def _g(self, r2):
        if not isinstance(r2, torch.Tensor):
            r2 = torch.tensor(r2)
        return torch.zeros_like(r2)  # metasurface is a plane

    def _dgd(self, r2):
        if not isinstance(r2, torch.Tensor):
            r2 = torch.tensor(r2)
        return torch.zeros_like(r2)  # metasurface is a plane


class Meta_Geometric_polofit_autograd(MetaSurface):
    """
    This class is designed for geometric phase metasurface. Use polynomial to represent the phase distribution.
    Then use torch.autograd to calculate the phase gradient.

    Args:
        ai: Polynomial coefficients
    """

    def __init__(self, r, d, c=0., ai=None, initial=None, is_square=False, device=torch.device('cpu')):
        MetaSurface.__init__(self, r, d, c, is_square, device)

        if ai is not None:
            self.ai = torch.Tensor(np.array(ai))
        else:
            self.ai = torch.zeros(8)
            # self.ai = torch.tensor([0.0, -0.001, 0, 0, 0, 0, 0, 0, 0, 0])
        self.ai_bound = 1000  # limit the range of ai

    def calculate_complex_amplitude_derivatives(self, r, wavelength):
        R = self.r  # radius of the metasurface

        # check the dimension of r, if it is 2, then save the original shape and flatten it
        original_shape = r.shape
        if r.dim() == 2:
            r = r.view(-1)  # flatten it

        r_positions_clone = r.clone().detach().requires_grad_(True)
        r_normalized = r_positions_clone / R  # normalized r position

        # use broadcasting to calculate the exponents of all terms
        exponents = torch.arange(0, 2 * len(self.ai), 2, device=self.device)
        terms = torch.pow(r_normalized[:, None], exponents)

        # calculate the value of each nanopillar
        ai = self.ai  # polynomial coefficients
        ai.data.clamp_(-1.0, 1.0)
        ai = ai * self.ai_bound  # <-- assuming this should be 1000

        phase = torch.sum(ai * terms, dim=1)

        # calculate the spatial derivative of the phase along the radial direction
        phase_spatial_derivative = torch.autograd.grad(outputs=phase, inputs=r_positions_clone, grad_outputs=torch.ones_like(phase),
                                                       retain_graph=True, create_graph=True)[0]
        

        # if the original r is 2D, restore the result to the original shape
        if original_shape != r.shape:
            phase = phase.view(original_shape)
            phase_spatial_derivative = phase_spatial_derivative.view(original_shape)

        return phase, phase_spatial_derivative

    def define_phase_spatial_derivative(self, p, wavelength):
        """
        Calculate the phase spatial gradient at the current metasurface position.

        Args:
            p: the coordinates of the current ray.
            wavelength: wavelength of the light wave.

        Returns:
            the derivative of the phase.
        """
        x = p[..., 0]
        y = p[..., 1]
        r2 = x ** 2 + y ** 2
        r = torch.sqrt(torch.clamp(r2, min=1e-8))

        phase, phase_derivatives = self.calculate_complex_amplitude_derivatives(r, wavelength)

        return phase_derivatives
    

class Meta_Geometric_hyperboloidal_phase_autograd(MetaSurface):
    """
    This class is designed for geometric phase metasurface. Use a fixed hyperbolic phase distribution to represent the phase distribution.
    Then use torch.autograd to calculate the phase gradient.

    Args:
        wavelength: wavelength of the light wave. Default is 530*1e-6.
        focal_length: focal length.
    """

    def __init__(self, r, d, focal_length, wavelength=530*1e-6, c=0., is_square=False, device=torch.device('cpu')):
        MetaSurface.__init__(self, r, d, c, is_square, device)
        self.wavelength = torch.tensor(wavelength, device=device)  # wavelength
        self.focal_length = torch.tensor(focal_length, device=device)  # focal length

    def calculate_complex_amplitude_derivatives(self, r, wavelength):
        R = self.r  # radius of the metasurface

        # check the dimension of r, if it is 2, then save the original shape and flatten it
        original_shape = r.shape
        if r.dim() == 2:
            r = r.view(-1)  # flatten it

        r_positions_clone = r.clone().detach().requires_grad_(True)

        # calculate the hyperbolic phase distribution
        f = self.focal_length
        phase = (2 * torch.pi / self.wavelength) * (f - torch.sqrt(r_positions_clone**2 + f**2))

        # # # calculate the spatial derivative of the phase along the radial direction
        phase_spatial_derivative = torch.autograd.grad(outputs=phase, inputs=r_positions_clone, grad_outputs=torch.ones_like(phase),
                                                       retain_graph=True, create_graph=True)[0]
        # calculate the spatial derivative of the phase along the radial direction
        # for the hyperbolic phase distribution phase = (2π/λ)(f - √(r² + f²))
        # the derivative is dphase/dr = -(2π/λ) * r/√(r² + f²)
        # phase_spatial_derivative = -(2 * torch.pi / self.wavelength) * r_positions_clone / torch.sqrt(r_positions_clone**2 + f**2)

        # if the original r is 2D, restore the result to the original shape
        if original_shape != r.shape:
            phase = phase.view(original_shape)
            phase_spatial_derivative = phase_spatial_derivative.view(original_shape)

        return phase, phase_spatial_derivative

    def define_phase_spatial_derivative(self, p, wavelength):
        """
        Calculate the phase spatial gradient at the current metasurface position.

        Args:
            p: the coordinates of the current ray.
            wavelength: wavelength of the light wave.

        Returns:
            the derivative of the phase.
        """
        x = p[..., 0]
        y = p[..., 1]
        r2 = x ** 2 + y ** 2
        r = torch.sqrt(torch.clamp(r2, min=1e-8))

        phase, phase_derivatives = self.calculate_complex_amplitude_derivatives(r, wavelength)

        return phase_derivatives


class Aspheric(Surface):
    """
    This is the aspheric surface class, implementation follows: https://en.wikipedia.org/wiki/Aspheric_lens.

    The surface is parameterized as an implicit function f(x,y,z) = 0.
    For simplicity, we assume the surface function f(x,y,z) can be decomposed as:
    
    f(x,y,z) = g(x,y) + h(z),

    where g(x,y) and h(z) are explicit functions:
    
    g(x,y) = c * r**2 / (1 + sqrt( 1 - (1+k) * r**2/R**2 )) + ai[0] * r**4 + ai[1] * r**6 + \cdots.
    h(z) = -z.
    
    Args (new attributes):
        c: Surface curvature, or one over radius of curvature.
        k: Conic coefficient.
        ai: Aspheric parameters, could be a vector. When None, the surface is spherical.
    """

    def __init__(self, r, d, c=0., k=0., ai=None, is_square=False, device=torch.device('cpu')):
        Surface.__init__(self, r, d, is_square, device)
        self.c, self.k = (torch.Tensor(np.array(v)) for v in [c, k])
        self.ai = None
        if ai is not None:
            self.ai = torch.Tensor(np.array(ai))

    # === Common methods
    def g(self, x, y):
        return self._g(x ** 2 + y ** 2)

    def dgd(self, x, y):
        dsdr2 = 2 * self._dgd(x ** 2 + y ** 2)
        return dsdr2 * x, dsdr2 * y

    def h(self, z):
        return -z

    def dhd(self, z):
        return -torch.ones_like(z)

    def surface(self, x, y):
        return self._g(x ** 2 + y ** 2)

    def reverse(self):
        self.c = -self.c
        if self.ai is not None:
            self.ai = -self.ai

    def surface_derivatives(self, x, y):
        dsdr2 = 2 * self._dgd(x ** 2 + y ** 2)
        return dsdr2 * x, dsdr2 * y, -torch.ones_like(x)

    def surface_and_derivatives_dot_D(self, t, dx, dy, dz, ox, oy, z, A, B, C):
        # pylint: disable=unused-argument
        # TODO: could be further optimized
        r2 = A * t ** 2 + B * t + C
        return self._g(r2) - z, self._dgd(r2) * (2 * A * t + B) - dz

    # === Private methods
    def _g(self, r2):
        tmp = r2 * self.c
        total_surface = tmp / (1 + torch.sqrt(1 - (1 + self.k) * tmp * self.c))
        higher_surface = 0
        if self.ai is not None:
            for i in np.flip(range(len(self.ai))):
                higher_surface = r2 * higher_surface + self.ai[i]
            higher_surface = higher_surface * r2 ** 2
        return total_surface + higher_surface

    def _dgd(self, r2):
        alpha_r2 = (1 + self.k) * self.c ** 2 * r2
        tmp = torch.sqrt(1 - alpha_r2)  # TODO: potential NaN grad
        total_derivative = self.c * (1 + tmp - 0.5 * alpha_r2) / (tmp * (1 + tmp) ** 2)

        higher_derivative = 0
        if self.ai is not None:
            for i in np.flip(range(len(self.ai))):
                higher_derivative = r2 * higher_derivative + (i + 2) * self.ai[i]
        return total_derivative + higher_derivative * r2
