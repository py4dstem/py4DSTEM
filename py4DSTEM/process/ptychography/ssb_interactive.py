import time

import cupy as cp
import cupyx.scipy.fft as fft

import ipywidgets as widgets

import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle
from matplotlib_scalebar.scalebar import ScaleBar

import numpy as np
from ipywidgets import FloatSlider, GridspecLayout, VBox, HBox

import torch as th

from numpy.fft import fftshift
from skimage.filters import gaussian

import py4DSTEM
from py4DSTEM.process.ptychography import ZernikeProbeSingle, find_rotation_angle_with_double_disk_overlap, \
    disk_overlap_function, single_sideband_reconstruction
from py4DSTEM.process.ptychography.utils import fourier_coordinates_2D
from py4DSTEM.process.ptychography.visualize import imsave
from py4DSTEM.process.utils import Param, sector_mask, get_qx_qy_1d
from py4DSTEM.io.datastructure import Metadata

plt.ioff()
out0 = widgets.Output(layout={'border': '1px solid black'})


class InteractiveSSB:
    """
    A class for interactive single-sideband reconstruction.

    A InteractiveSSB instance is intended for use in jupyter notebooks with ipywidgets enabled, where it displays a
    graphical user interface that allows to select a region of interest and the STEM rotation, and reconstructs an image
    using the Single-sideband reconstruction method. The aberrations used for the reconstruction can be tuned interactively.

    Initialization methods:

        __init__:
            Initializes the class variables and the graphical user interface.

        show: shows the ipywidgets GUI

    Args:
        data (ndarray): 4D data cube, cropped around the bright-field disk
        slice_image (ndarray of floats): an image that is displayed to select a ROI to reconstruct
        bright_field_radius (float): radisu of the bright-field disk in pixels
        metadata (Metadata): a py4Dstem metadata object with populated metadata
        ssb_size (int): size of the diffraction patten to use in the SSB reconstruction, mostly used for saving memory. default: 16
    """
    def __init__(self, data: np.array, slice_image: np.array, bright_field_radius: float, metadata: Metadata,
                 ssb_size=16):

        meta = Param()
        meta.alpha_rad = metadata.microscope['convergence_semiangle_mrad'] * 1e-3
        meta.wavelength = py4DSTEM.process.utils.electron_wavelength_angstrom(metadata.microscope['beam_energy'])
        meta.scan_step = metadata.calibration['R_pixel_size']
        meta.E_ev = metadata.microscope['beam_energy']

        self.out = widgets.Output(layout={'border': '1px solid black'})
        self.meta = meta
        self.data = data
        self.scan_dimensions = np.array(self.data.shape[:2])
        self.frame_dimensions = np.array(self.data.shape[2:])
        self.ssb_size = ssb_size
        self.slic = np.s_[0:self.scan_dimensions[0], 0:self.scan_dimensions[1]]
        self.rmax = self.frame_dimensions[-1] // 2
        self.alpha_max = self.rmax / bright_field_radius * meta.alpha_rad
        self.slice_image = slice_image
        self.window_center = self.scan_dimensions // 2
        self.window_size = self.scan_dimensions[0]

        r_min = meta.wavelength / (2 * self.alpha_max)
        r_min = np.array([r_min, r_min])
        self.k_max = [self.alpha_max / meta.wavelength, self.alpha_max / meta.wavelength]
        self.r_min = np.array(r_min)
        self.dxy = np.array(meta.scan_step)

        self.margin = self.frame_dimensions // 2
        self.rotation_deg = 0.0

        self.out.append_stdout(f"alpha_max       = {self.alpha_max * 1e3:2.2f} mrad\n")
        self.out.append_stdout(f"E               = {meta.E_ev / 1e3} keV\n")
        self.out.append_stdout(f"λ               = {meta.wavelength * 1e2:2.2}   pm\n")
        self.out.append_stdout(f"dR              = {self.dxy[0]:2.2f}             Å\n")
        self.out.append_stdout(f"dK              = {self.k_max[0]:2.2f}          Å^-1\n")
        self.out.append_stdout(f"scan       size = {self.scan_dimensions}\n")
        self.out.append_stdout(f"detector   size = {self.frame_dimensions}\n")

        self.rotation_slider = widgets.FloatSlider(
            value=0,
            min=-180,
            max=180,
            step=0.1,
            description='STEM rotation angle:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.1f',
        )
        self.rotation_slider.observe(self.rotation_changed, 'value')

        self.window_size_slider = widgets.IntSlider(
            value=self.scan_dimensions[0],
            min=2,
            max=np.max(self.scan_dimensions),
            step=2,
            description='Window size',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True
        )
        self.window_size_slider.observe(self.window_size_slider_changes, 'value')

        self.aberration_text = widgets.HTML(
            value="1",
            placeholder='',
            description='',
        )

        self.gs = GridspecLayout(4, 9)
        self.Cslider_box = VBox(width=50)
        self.scale_slider_box = VBox()
        children = []
        sliders = []

        self.probe_output = widgets.Output()
        self.overlaps_output = widgets.Output()

        qn = fourier_coordinates_2D([128, 128], r_min / 1.6)
        q = th.as_tensor(qn).cuda()
        self.probe_gen = ZernikeProbeSingle(q, meta.wavelength, fft_shifted=False)
        self.A = np.linalg.norm(qn, axis=0) * meta.wavelength < meta.alpha_rad
        self.A = gaussian(self.A, 1)

        self.C = cp.zeros((12,))
        self.C_names = ['C1', 'C12a', 'C12b', 'C21a', 'C21b', 'C23a', 'C23b', 'C3', 'C32a', 'C32b', 'C34a', 'C34b']
        self.C_min = [-120, -20, -20, -50, -50, -50, -50, -20, -20, -20, -20, -20]
        self.C_max = [120, 20, 20, 50, 50, 50, 50, 20, 20, 20, 20, 20]
        self.C_multiplier = [1e1, 1e1, 1e1, 1e1, 1e1, 1e1, 1e4, 1e4, 1e4, 1e4, 1e4]

        Psi = self.probe_gen(th.tensor(self.C.get()).cuda(), th.tensor(self.A).cuda())
        self.phases = th.angle(
            th.fft.fftshift(self.probe_gen(th.tensor(self.C.get()).cuda(), th.tensor(np.ones_like(self.A)).cuda())))
        self.Psi_shifted = th.fft.fftshift(Psi)
        self.psi = th.fft.fftshift(th.fft.ifft2(Psi))

        with self.probe_output:
            self.probe_figure = plt.figure(constrained_layout=True, figsize=(9, 3))
            gs1 = self.probe_figure.add_gridspec(1, 3, wspace=0.05, hspace=0.05)
            self.f1ax0 = self.probe_figure.add_subplot(gs1[0])
            self.f1ax1 = self.probe_figure.add_subplot(gs1[1])
            self.f1ax2 = self.probe_figure.add_subplot(gs1[2])

            self.f1ax0.set_title('Probe  (real space)')
            self.f1ax1.set_title('Probe  (Fourier space)')
            self.f1ax2.set_title('Phase profile (Fourier space)')

            self.probe_realspace_imax = self.f1ax0.imshow(imsave(self.psi.cpu().numpy()))
            self.probe_fourier_imax = self.f1ax1.imshow(imsave(self.Psi_shifted.cpu().numpy()))
            self.probe_phases_imax = self.f1ax2.imshow(self.phases.cpu().numpy())

            self.f1ax0.set_xticks([])
            self.f1ax0.set_yticks([])
            self.f1ax1.set_xticks([])
            self.f1ax1.set_yticks([])
            self.f1ax2.set_xticks([])
            self.f1ax2.set_yticks([])

        self.overlap_figure_axes = []
        with self.overlaps_output:
            self.overlap_figure = plt.figure(constrained_layout=True, figsize=(9, 9))
            gs1 = self.overlap_figure.add_gridspec(3, 3, wspace=0.05, hspace=0.05)
            for ggs in gs1:
                f3_ax1 = self.overlap_figure.add_subplot(ggs)
                imax2 = f3_ax1.imshow(np.zeros((40, 40)))
                f3_ax1.set_xticks([])
                f3_ax1.set_yticks([])
                self.overlap_figure_axes.append(imax2)

        #         self.plot_box = VBox(children=[])
        self.plot_box = VBox(children=[self.probe_figure.canvas, self.overlap_figure.canvas])
        self.recon_fig, self.recon_axes = plt.subplots(figsize=(9, 9))
        self.recon_img = self.recon_axes.imshow(np.zeros(self.scan_dimensions), cmap=plt.get_cmap('bone'))
        self.recon_axes.set_xticks([])
        self.recon_axes.set_yticks([])
        scalebar = ScaleBar(meta.scan_step[0] / 10, 'nm')  # 1 pixel = 0.2 meter
        self.recon_axes.add_artist(scalebar)

        for i, (name, mins, maxs, multiplier) in enumerate(
                zip(self.C_names, self.C_min, self.C_max, self.C_multiplier)):
            s = FloatSlider(description=name,
                            min=mins, max=maxs)
            s.observe(self.create_function(f'slider_changed_{i}', i, multiplier), names='value')
            sliders.append(s)
            children.append(s)

        self.Cslider_box.children = children + [self.aberration_text]

        self.gs[:2, 0] = self.Cslider_box
        self.gs[2:, 0] = self.scale_slider_box
        self.gs[:, 1:5] = self.plot_box
        self.gs[:, 5:9] = self.recon_fig.canvas

        self.first_time_calc = True

    def __del__(self):
        del self.recon_fig
        del self.overlap_figure
        del self.probe_figure

    def rotation_changed(self, change):
        self.rotation_deg = change['new']

    @out0.capture()
    def update_slice(self):
        y0 = self.window_center[0] - self.window_size[0] // 2
        y1 = self.window_center[0] + self.window_size[0] // 2
        x0 = self.window_center[1] - self.window_size[1] // 2
        x1 = self.window_center[1] + self.window_size[1] // 2

        y0 = int(y0 if y0 > 0 else 0)
        y1 = int(y1 if y1 <= self.scan_dimensions[0] else self.scan_dimensions[0])
        x0 = int(x0 if x0 > 0 else 0)
        x1 = int(x1 if x1 <= self.scan_dimensions[1] else self.scan_dimensions[1])

        if y0 == 0:
            y1 = (y1 // 2) * 2
            self.window_size[0] = y1 - y0

        if x0 == 0:
            x1 = (x1 // 2) * 2
            self.window_size[1] = x1 - x0

        if y1 == self.scan_dimensions[0] and y0 % 2 > 0:
            y0 -= 1
            self.window_size[0] = self.window_size[0] + 1

        if x1 == self.scan_dimensions[0] and x0 % 2 > 0:
            x0 -= 1
            self.window_size[1] = self.window_size[1] + 1

        self.slic = np.s_[y0:y1, x0:x1]
        self.out.append_stdout(f"slice {y0},{y1},{x0},{x1}\n")

        self.slice_rect.set_xy((x0, y0))
        self.slice_rect.set_height(y1 - y0)
        self.slice_rect.set_width(x1 - x0)

        self.G = self._get_G(self.ssb_size)
        self.Gabs = cp.sum(cp.abs(self.G), (2, 3))

        sh = np.array(self.Gabs.shape)
        mask = ~np.array(fftshift(sector_mask(sh, sh // 2, sh[0] / 30, (0, 360))))
        mask[:, -1] = 0
        mask[:, 0] = 0
        mask[:, 1] = 0

        gg = np.log10(self.Gabs.get() + 1)
        gg[~mask] = gg.mean()
        gg = fftshift(gg)
        gg1 = gg  # [self.margin[0]:-self.margin[0],self.margin[1]:-self.margin[1]]
        self.imax_power_spectrum.set_data(gg1)
        self.imax_power_spectrum.set_clim(gg1.min(), gg1.max())

        self.fig_select_slice.canvas.draw()
        self.fig_select_slice.canvas.flush_events()
        self.fig_power_spectrum.canvas.draw()
        self.fig_power_spectrum.canvas.flush_events()

    @out0.capture()
    def window_size_slider_changes(self, change):
        self.window_size = [change['new'], change['new']]
        self.update_slice()

    @out0.capture()
    def fig_select_slice_onclick(self, event):
        ix, iy = event.xdata, event.ydata
        #         self.out.append_stdout(f"clicked ix:{ix},iy:{iy}\n")
        self.window_center = [iy, ix]
        self.update_slice()

    @out0.capture()
    def _get_G(self, size):
        dc = self.dc[self.slic]
        M = cp.array(dc, dtype=cp.complex64)
        G = fft.fft2(M, axes=(0, 1), overwrite_x=True)
        G /= cp.sqrt(np.prod(G.shape[:2]))
        return G

    @out0.capture()
    def _get_G_full(self, size):
        start = time.perf_counter()
        self.dc = self.data[self.slic]
        self.out.append_stdout(f"Data shape is {self.dc.shape}\n")

        self.Qy1d, self.Qx1d = get_qx_qy_1d(self.scan_dimensions, self.dxy, fft_shifted=False)
        self.Ky, self.Kx = get_qx_qy_1d(self.dc.shape[-2:], self.r_min, fft_shifted=True)

        self.Kx = cp.array(self.Kx, dtype=cp.float32)
        self.Ky = cp.array(self.Ky, dtype=cp.float32)
        self.Qy1d = cp.array(self.Qy1d, dtype=cp.float32)
        self.Qx1d = cp.array(self.Qx1d, dtype=cp.float32)

        self.Psi_Qp = cp.zeros(self.scan_dimensions, dtype=np.complex64)
        self.Psi_Qp_left_sb = cp.zeros(self.scan_dimensions, dtype=np.complex64)
        self.Psi_Qp_right_sb = cp.zeros(self.scan_dimensions, dtype=np.complex64)
        self.Psi_Rp = cp.zeros(self.scan_dimensions, dtype=np.complex64)
        self.Psi_Rp_left_sb = cp.zeros(self.scan_dimensions, dtype=np.complex64)
        self.Psi_Rp_right_sb = cp.zeros(self.scan_dimensions, dtype=np.complex64)

        M = cp.array(self.dc, dtype=cp.complex64)
        start = time.perf_counter()
        G = fft.fft2(M, axes=(0, 1), overwrite_x=True)
        G /= cp.sqrt(np.prod(G.shape[:2]))

        self.out.append_stdout(f"FFT along scan coordinate took {time.perf_counter() - start:2.2g}s\n")
        return G

    @out0.capture()
    def update_variables(self):
        self.Psi_Qp[:] = 0
        self.Psi_Qp_left_sb[:] = 0
        self.Psi_Qp_right_sb[:] = 0

        eps = 1e-3
        single_sideband_reconstruction(
            self.G,
            self.Qx1d,
            self.Qy1d,
            self.Kx,
            self.Ky,
            self.C,
            np.deg2rad(self.rotation_deg),
            self.meta.alpha_rad,
            self.Psi_Qp,
            self.Psi_Qp_left_sb,
            self.Psi_Qp_right_sb,
            eps,
            self.meta.wavelength,
        )

        self.Psi_Rp[:] = fft.ifft2(self.Psi_Qp, norm="ortho")
        self.Psi_Rp_left_sb[:] = fft.ifft2(self.Psi_Qp_left_sb, norm="ortho")
        self.Psi_Rp_right_sb[:] = fft.ifft2(self.Psi_Qp_right_sb, norm="ortho")

        self.Gamma = disk_overlap_function(self.Qx_max1d, self.Qy_max1d, self.Kx, self.Ky, self.C,
                                           np.deg2rad(self.rotation_deg), self.meta.alpha_rad, self.meta.wavelength)

        Psi = self.probe_gen(th.tensor(self.C.get()).cuda(), th.tensor(self.A).cuda())
        self.phases = th.angle(
            th.fft.fftshift(self.probe_gen(th.tensor(self.C.get()).cuda(), th.tensor(self.A).cuda())))
        self.Psi_shifted = th.fft.fftshift(Psi)
        self.psi = th.fft.fftshift(th.fft.ifft2(Psi))

    @out0.capture()
    def update_gui(self):
        gg = self.Gamma * self.G_max

        m = 10
        img = np.angle(self.Psi_Rp_left_sb.get()[m:-m, m:-m])
        self.recon_img.set_data(img)
        self.recon_img.set_clim(img.min(), img.max())

        for ax, g in zip(self.overlap_figure_axes, gg):
            ax.set_data(imsave(g.get()))

        self.probe_realspace_imax.set_data(imsave(self.psi.cpu().numpy()))
        self.probe_fourier_imax.set_data(imsave(self.Psi_shifted.cpu().numpy()))
        self.probe_phases_imax.set_data(self.phases.cpu().numpy())
        self.probe_phases_imax.set_clim(self.phases.min(), self.phases.max())

        self.recon_fig.canvas.draw()
        self.overlap_figure.canvas.draw()
        self.probe_figure.canvas.draw()

        self.recon_fig.canvas.flush_events()
        self.overlap_figure.canvas.flush_events()
        self.probe_figure.canvas.flush_events()

    @out0.capture()
    def create_function(self, name, i, multiplier):
        def func1(change):
            self.C[i] = change['new'] * multiplier
            w = change['new'] * multiplier
            self.update_variables()
            self.update_gui()

        func1.__name__ = name
        return func1

    @out0.capture()
    def selected_index_changed(self, change):
        w = change['new']
        if w == 1:
            n_fit = 9
            self.G = self._get_G_full(self.ssb_size)
            self.Gabs = cp.sum(cp.abs(self.G), (2, 3))
            self.out.append_stdout(f"self.G.shape {self.G.shape}\n")

            sh = np.array(self.Gabs.shape)
            mask = ~np.array(fftshift(sector_mask(sh, sh // 2, 15, (0, 360))))
            mask[:, -1] = 0
            mask[:, 0] = 0
            mask[:, 1] = 0

            gg = self.Gabs.get()
            gg[~mask] = gg.mean()

            inds = np.argsort((gg).ravel())
            strongest_object_frequencies = np.unravel_index(inds[-1 - n_fit:-1], self.G.shape[:2])

            self.out.append_stdout(f"strongest_object_frequencies: {strongest_object_frequencies}\n")

            self.G_max = self.G[strongest_object_frequencies]
            self.Qy_max1d = self.Qy1d[strongest_object_frequencies[0]]
            self.Qx_max1d = self.Qx1d[strongest_object_frequencies[1]]

            self.Gamma = disk_overlap_function(self.Qx_max1d, self.Qy_max1d, self.Kx, self.Ky, self.C,
                                               np.deg2rad(self.rotation_deg), self.meta.alpha_rad, self.meta.wavelength)

            self.update_variables()
            self.update_gui()

    def show(self):
        """
        Shows the intarctive GUI.

        Args:

        Returns:
            an ipywidgets GridspecLayout containing the GUI
        """

        n_fit = 49
        ranges = [360, 30]
        partitions = [144, 120]
        ranges = [360]
        partitions = [360]
        manual_frequencies = None
        manual_frequencies = None

        bin_factor = int(np.min(np.floor(self.frame_dimensions / 12)))
        self.dc = self.data[self.slic]

        M = cp.array(self.dc, dtype=cp.complex64)
        self.G = fft.fft2(M, axes=(0, 1), overwrite_x=True)
        self.G /= cp.sqrt(np.prod(self.G.shape[:2]))
        self.Gabs = cp.sum(cp.abs(self.G), (2, 3))

        sh = np.array(self.Gabs.shape)
        mask = ~np.array(fftshift(sector_mask(sh, sh // 2, 15, (0, 360))))
        mask[:, -1] = 0
        mask[:, 0] = 0
        mask[:, 1] = 0

        gg = np.log10(self.Gabs.get() + 1)
        gg[~mask] = gg.mean()
        gg = fftshift(gg)
        gg1 = gg  # [self.margin[0]:-self.margin[0],self.margin[1]:-self.margin[1]]

        self.fig_select_slice, self.ax_select_slice = plt.subplots(1, 1, figsize=(8, 8))
        self.fig_select_slice.canvas.mpl_connect('button_press_event', self.fig_select_slice_onclick)
        self.imax_select_slice = self.ax_select_slice.imshow(self.slice_image, cmap=plt.cm.get_cmap('magma'))
        self.slice_rect = Rectangle((self.slic[0].start, self.slic[1].start), self.window_size, self.window_size,
                                    fill=False, lw=3, ls='--')
        self.ax_select_slice.add_patch(self.slice_rect)

        self.fig_power_spectrum, self.ax_power_spectrum = plt.subplots(1, 1, figsize=(8, 8))
        self.imax_power_spectrum = self.ax_power_spectrum.imshow(gg1, cmap=plt.cm.get_cmap('magma'))

        best, thetas, intensities = find_rotation_angle_with_double_disk_overlap(self.G, self.meta.wavelength,
                                                                                 self.r_min, self.dxy,
                                                                                 self.meta.alpha_rad,
                                                                                 mask=cp.array(mask), ranges=ranges,
                                                                                 partitions=partitions, n_fit=n_fit,
                                                                                 verbose=False,
                                                                                 manual_frequencies=manual_frequencies,
                                                                                 aberrations=self.C)

        self.out.append_stdout(f"Best rotation angle: {np.rad2deg(thetas[best])}\n")

        fig, ax = plt.subplots()
        ax.scatter(np.rad2deg(thetas), intensities)
        ax.set_xlabel('STEM rotation [degrees]')
        ax.set_ylabel('Integrated G amplitude over double overlap')

        rot_box = VBox([fig.canvas, self.rotation_slider])

        window_size_box = VBox([self.window_size_slider, self.fig_select_slice.canvas])

        canvas_box = HBox([window_size_box, self.fig_power_spectrum.canvas, rot_box])
        gsl1 = VBox([canvas_box])
        gsl2 = GridspecLayout(1, 1)

        tab_contents = ['P0', 'P1', 'P2']
        children = [gsl1, self.gs]
        tab = widgets.Tab()
        tab.children = children
        tab.titles = ['fft', '00', '00']

        tab.observe(self.selected_index_changed, 'selected_index')

        gsl00 = GridspecLayout(20, 20)
        gsl00[3:, :3] = self.out
        gsl00[3:, 3:] = tab
        gsl00[:3, :] = out0
        return gsl00
