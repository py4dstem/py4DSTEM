import warnings

import numpy as np
import pylops
from py4DSTEM.process.phase.utils import (
    array_slice,
    estimate_global_transformation_ransac,
    fft_shift,
    fit_aberration_surface,
    regularize_probe_amplitude,
)
from py4DSTEM.process.utils import get_CoM


class PtychographicConstraints:
    """
    Container class for PtychographicReconstruction methods.
    """

    def _object_threshold_constraint(self, current_object, pure_phase_object):
        """
        Ptychographic threshold constraint.
        Used for avoiding the scaling ambiguity between probe and object.

        Parameters
        --------
        current_object: np.ndarray
            Current object estimate
        pure_phase_object: bool
            If True, object amplitude is set to unity

        Returns
        --------
        constrained_object: np.ndarray
            Constrained object estimate
        """
        xp = self._xp
        phase = xp.angle(current_object)

        if pure_phase_object:
            amplitude = 1.0
        else:
            amplitude = xp.minimum(xp.abs(current_object), 1.0)

        return amplitude * xp.exp(1.0j * phase)

    def _object_shrinkage_constraint(self, current_object, shrinkage_rad, object_mask):
        """
        Ptychographic shrinkage constraint.
        Used to ensure electrostatic potential is positive.

        Parameters
        --------
        current_object: np.ndarray
            Current object estimate
        shrinkage_rad: float
            Phase shift in radians to be subtracted from the potential at each iteration
        object_mask: np.ndarray (boolean)
            If not None, used to calculate additional shrinkage using masked-mean of object

        Returns
        --------
        constrained_object: np.ndarray
            Constrained object estimate
        """
        xp = self._xp

        if self._object_type == "complex":
            phase = xp.angle(current_object)
            amp = xp.abs(current_object)

            if object_mask is not None:
                shrinkage_rad += phase[..., object_mask].mean()

            phase -= shrinkage_rad

            current_object = amp * xp.exp(1.0j * phase)
        else:
            if object_mask is not None:
                shrinkage_rad += current_object[..., object_mask].mean()

            current_object -= shrinkage_rad

        return current_object

    def _object_positivity_constraint(self, current_object):
        """
        Ptychographic positivity constraint.
        Used to ensure potential is positive.

        Parameters
        --------
        current_object: np.ndarray
            Current object estimate

        Returns
        --------
        constrained_object: np.ndarray
            Constrained object estimate
        """
        xp = self._xp

        return xp.maximum(current_object, 0.0)

    def _object_gaussian_constraint(
        self, current_object, gaussian_filter_sigma, pure_phase_object
    ):
        """
        Ptychographic smoothness constraint.
        Used for blurring object.

        Parameters
        --------
        current_object: np.ndarray
            Current object estimate
        gaussian_filter_sigma: float
            Standard deviation of gaussian kernel in A
        pure_phase_object: bool
            If True, gaussian blur performed on phase only

        Returns
        --------
        constrained_object: np.ndarray
            Constrained object estimate
        """
        xp = self._xp
        gaussian_filter = self._gaussian_filter
        gaussian_filter_sigma /= self.sampling[0]

        if pure_phase_object:
            phase = xp.angle(current_object)
            phase = gaussian_filter(phase, gaussian_filter_sigma)
            current_object = xp.exp(1.0j * phase)
        else:
            current_object = gaussian_filter(current_object, gaussian_filter_sigma)

        return current_object

    def _object_butterworth_constraint(
        self,
        current_object,
        q_lowpass,
        q_highpass,
        butterworth_order,
    ):
        """
        Ptychographic butterworth filter.
        Used for low/high-pass filtering object.

        Parameters
        --------
        current_object: np.ndarray
            Current object estimate
        q_lowpass: float
            Cut-off frequency in A^-1 for low-pass butterworth filter
        q_highpass: float
            Cut-off frequency in A^-1 for high-pass butterworth filter
        butterworth_order: float
            Butterworth filter order. Smaller gives a smoother filter

        Returns
        --------
        constrained_object: np.ndarray
            Constrained object estimate
        """
        xp = self._xp
        qx = xp.fft.fftfreq(current_object.shape[0], self.sampling[0])
        qy = xp.fft.fftfreq(current_object.shape[1], self.sampling[1])

        qya, qxa = xp.meshgrid(qy, qx)
        qra = xp.sqrt(qxa**2 + qya**2)

        env = xp.ones_like(qra)
        if q_highpass:
            env *= 1 - 1 / (1 + (qra / q_highpass) ** (2 * butterworth_order))
        if q_lowpass:
            env *= 1 / (1 + (qra / q_lowpass) ** (2 * butterworth_order))

        current_object_mean = xp.mean(current_object)
        current_object -= current_object_mean
        current_object = xp.fft.ifft2(xp.fft.fft2(current_object) * env)
        current_object += current_object_mean

        if self._object_type == "potential":
            current_object = xp.real(current_object)

        return current_object

    def _object_denoise_tv_pylops(self, current_object, weight, iterations):
        """
        Performs second order TV denoising along x and y

        Parameters
        ----------
        current_object: np.ndarray
            Current object estimate
        weight : float
            Denoising weight. The greater `weight`, the more denoising (at
            the expense of fidelity to `input`).
        iterations: float
            Number of iterations to run in denoising algorithm.
            `niter_out` in pylops

        Returns
        -------
        constrained_object: np.ndarray
            Constrained object estimate

        """
        xp = self._xp

        if xp.iscomplexobj(current_object):
            current_object_tv = current_object
            warnings.warn(
                ("TV denoising is currently only supported for potential objects."),
                UserWarning,
            )

        else:
            nx, ny = current_object.shape
            niter_out = iterations
            niter_in = 1
            Iop = pylops.Identity(nx * ny)
            xy_laplacian = pylops.Laplacian(
                (nx, ny), axes=(0, 1), edge=False, kind="backward"
            )

            l1_regs = [xy_laplacian]

            current_object_tv = pylops.optimization.sparsity.splitbregman(
                Op=Iop,
                y=current_object.ravel(),
                RegsL1=l1_regs,
                niter_outer=niter_out,
                niter_inner=niter_in,
                epsRL1s=[weight],
                tol=1e-4,
                tau=1.0,
                show=False,
            )[0]

            current_object_tv = current_object_tv.reshape(current_object.shape)

        return current_object_tv

    def _object_denoise_tv_chambolle(
        self,
        current_object,
        weight,
        axis,
        pad_object,
        eps=2.0e-4,
        max_num_iter=200,
        scaling=None,
    ):
        """
        Perform total-variation denoising on n-dimensional images.

        Parameters
        ----------
        current_object: np.ndarray
            Current object estimate
        weight : float, optional
            Denoising weight. The greater `weight`, the more denoising (at
            the expense of fidelity to `input`).
        axis: int or tuple
            Axis for denoising, if None uses all axes
        pad_object: bool
            if True, pads object with zeros along axes of blurring
        eps : float, optional
            Relative difference of the value of the cost function that determines
            the stop criterion. The algorithm stops when:

                (E_(n-1) - E_n) < eps * E_0

        max_num_iter : int, optional
            Maximal number of iterations used for the optimization.
        scaling : tuple, optional
            Scale weight of tv denoise on different axes

        Returns
        -------
        constrained_object: np.ndarray
            Constrained object estimate

        Notes
        -----
        Rudin, Osher and Fatemi algorithm.
        Adapted skimage.restoration.denoise_tv_chambolle.
        """
        xp = self._xp
        if xp.iscomplexobj(current_object):
            updated_object = current_object
            warnings.warn(
                ("TV denoising is currently only supported for potential objects."),
                UserWarning,
            )
        else:
            current_object_sum = xp.sum(current_object)
            if axis is None:
                ndim = xp.arange(current_object.ndim).tolist()
            elif isinstance(axis, tuple):
                ndim = list(axis)
            else:
                ndim = [axis]

            if pad_object:
                pad_width = ((0, 0),) * current_object.ndim
                pad_width = list(pad_width)
                for ax in range(len(ndim)):
                    pad_width[ndim[ax]] = (1, 1)
                current_object = xp.pad(
                    current_object, pad_width=pad_width, mode="constant"
                )

            p = xp.zeros(
                (current_object.ndim,) + current_object.shape,
                dtype=current_object.dtype,
            )
            g = xp.zeros_like(p)
            d = xp.zeros_like(current_object)

            i = 0
            while i < max_num_iter:
                if i > 0:
                    # d will be the (negative) divergence of p
                    d = -p.sum(0)
                    slices_d = [
                        slice(None),
                    ] * current_object.ndim
                    slices_p = [
                        slice(None),
                    ] * (current_object.ndim + 1)
                    for ax in range(len(ndim)):
                        slices_d[ndim[ax]] = slice(1, None)
                        slices_p[ndim[ax] + 1] = slice(0, -1)
                        slices_p[0] = ndim[ax]
                        d[tuple(slices_d)] += p[tuple(slices_p)]
                        slices_d[ndim[ax]] = slice(None)
                        slices_p[ndim[ax] + 1] = slice(None)
                    updated_object = current_object + d
                else:
                    updated_object = current_object
                E = (d**2).sum()

                # g stores the gradients of updated_object along each axis
                # e.g. g[0] is the first order finite difference along axis 0
                slices_g = [
                    slice(None),
                ] * (current_object.ndim + 1)
                for ax in range(len(ndim)):
                    slices_g[ndim[ax] + 1] = slice(0, -1)
                    slices_g[0] = ndim[ax]
                    g[tuple(slices_g)] = xp.diff(updated_object, axis=ndim[ax])
                    slices_g[ndim[ax] + 1] = slice(None)
                if scaling is not None:
                    scaling /= xp.max(scaling)
                    g *= xp.array(scaling)[:, xp.newaxis, xp.newaxis]
                norm = xp.sqrt((g**2).sum(axis=0))[xp.newaxis, ...]
                E += weight * norm.sum()
                tau = 1.0 / (2.0 * len(ndim))
                norm *= tau / weight
                norm += 1.0
                p -= tau * g
                p /= norm
                E /= float(current_object.size)
                if i == 0:
                    E_init = E
                    E_previous = E
                else:
                    if xp.abs(E_previous - E) < eps * E_init:
                        break
                    else:
                        E_previous = E
                i += 1

            if pad_object:
                for ax in range(len(ndim)):
                    slices = array_slice(ndim[ax], current_object.ndim, 1, -1)
                    updated_object = updated_object[slices]
            updated_object = (
                updated_object / xp.sum(updated_object) * current_object_sum
            )

        return updated_object

    def _probe_center_of_mass_constraint(self, current_probe):
        """
        Ptychographic center of mass constraint.
        Used for centering corner-centered probe intensity.

        Parameters
        --------
        current_probe: np.ndarray
            Current probe estimate

        Returns
        --------
        constrained_probe: np.ndarray
            Constrained probe estimate
        """
        xp = self._xp

        probe_intensity = xp.abs(current_probe) ** 2

        probe_x0, probe_y0 = get_CoM(
            probe_intensity, device=self._device, corner_centered=True
        )
        shifted_probe = fft_shift(current_probe, -xp.array([probe_x0, probe_y0]), xp)

        return shifted_probe

    def _probe_amplitude_constraint(
        self, current_probe, relative_radius, relative_width
    ):
        """
        Ptychographic top-hat filtering of probe.

        Parameters
        ----------
        current_probe: np.ndarray
            Current positions estimate
        relative_radius: float
            Relative location of top-hat inflection point, between 0 and 0.5
        relative_width: float
            Relative width of top-hat sigmoid, between 0 and 0.5

        Returns
        --------
        constrained_probe: np.ndarray
            Constrained probe estimate
        """
        xp = self._xp
        erf = self._erf

        # probe_intensity = xp.abs(current_probe) ** 2
        # current_probe_sum = xp.sum(probe_intensity)

        X = xp.fft.fftfreq(current_probe.shape[0])[:, None]
        Y = xp.fft.fftfreq(current_probe.shape[1])[None]
        r = xp.hypot(X, Y) - relative_radius

        sigma = np.sqrt(np.pi) / relative_width
        tophat_mask = 0.5 * (1 - erf(sigma * r / (1 - r**2)))

        updated_probe = current_probe * tophat_mask
        # updated_probe_sum = xp.sum(xp.abs(updated_probe) ** 2)
        # normalization = xp.sqrt(current_probe_sum / updated_probe_sum)

        return updated_probe  # * normalization

    def _probe_fourier_amplitude_constraint(
        self,
        current_probe,
        width_max_pixels,
        enforce_constant_intensity,
    ):
        """
        Ptychographic top-hat filtering of Fourier probe.

        Parameters
        ----------
        current_probe: np.ndarray
            Current positions estimate
        threshold: np.ndarray
            Threshold value for current probe fourier mask. Value should
            be between 0 and 1, where 1 uses the maximum amplitude to threshold.
        relative_width: float
            Relative width of top-hat sigmoid, between 0 and 0.5

        Returns
        --------
        constrained_probe: np.ndarray
            Constrained probe estimate
        """
        xp = self._xp
        asnumpy = self._asnumpy

        # current_probe_sum = xp.sum(xp.abs(current_probe) ** 2)
        current_probe_fft = xp.fft.fft2(current_probe)

        updated_probe_fft, _, _, _ = regularize_probe_amplitude(
            asnumpy(current_probe_fft),
            width_max_pixels=width_max_pixels,
            nearest_angular_neighbor_averaging=5,
            enforce_constant_intensity=enforce_constant_intensity,
            corner_centered=True,
        )

        updated_probe_fft = xp.asarray(updated_probe_fft)
        updated_probe = xp.fft.ifft2(updated_probe_fft)
        # updated_probe_sum = xp.sum(xp.abs(updated_probe) ** 2)
        # normalization = xp.sqrt(current_probe_sum / updated_probe_sum)

        return updated_probe  # * normalization

    def _probe_aperture_constraint(
        self,
        current_probe,
        initial_probe_aperture,
    ):
        """
        Ptychographic constraint to fix Fourier amplitude to initial aperture.

        Parameters
        ----------
        current_probe: np.ndarray
            Current positions estimate

        Returns
        --------
        constrained_probe: np.ndarray
            Constrained probe estimate
        """
        xp = self._xp

        # current_probe_sum = xp.sum(xp.abs(current_probe) ** 2)
        current_probe_fft_phase = xp.angle(xp.fft.fft2(current_probe))

        updated_probe = xp.fft.ifft2(
            xp.exp(1j * current_probe_fft_phase) * initial_probe_aperture
        )
        # updated_probe_sum = xp.sum(xp.abs(updated_probe) ** 2)
        # normalization = xp.sqrt(current_probe_sum / updated_probe_sum)

        return updated_probe  # * normalization

    def _probe_aberration_fitting_constraint(
        self,
        current_probe,
        max_angular_order,
        max_radial_order,
    ):
        """
        Ptychographic probe smoothing constraint.
        Removes/adds known (initialization) aberrations before/after smoothing.

        Parameters
        ----------
        current_probe: np.ndarray
            Current positions estimate
        gaussian_filter_sigma: float
            Standard deviation of gaussian kernel in A^-1
        fix_amplitude: bool
            If True, only the phase is smoothed

        Returns
        --------
        constrained_probe: np.ndarray
            Constrained probe estimate
        """

        xp = self._xp

        fourier_probe = xp.fft.fft2(current_probe)
        fourier_probe_abs = xp.abs(fourier_probe)
        sampling = self.sampling
        energy = self._energy

        fitted_angle, _ = fit_aberration_surface(
            fourier_probe,
            sampling,
            energy,
            max_angular_order,
            max_radial_order,
            xp=xp,
        )

        fourier_probe = fourier_probe_abs * xp.exp(1.0j * fitted_angle)
        current_probe = xp.fft.ifft2(fourier_probe)

        return current_probe

    def _positions_center_of_mass_constraint(self, current_positions):
        """
        Ptychographic position center of mass constraint.
        Additionally updates vectorized indices used in _overlap_projection.

        Parameters
        ----------
        current_positions: np.ndarray
            Current positions estimate

        Returns
        --------
        constrained_positions: np.ndarray
            CoM constrained positions estimate
        """
        xp = self._xp

        current_positions -= xp.mean(current_positions, axis=0) - self._positions_px_com
        self._positions_px_fractional = current_positions - xp.round(current_positions)

        (
            self._vectorized_patch_indices_row,
            self._vectorized_patch_indices_col,
        ) = self._extract_vectorized_patch_indices()

        return current_positions

    def _positions_affine_transformation_constraint(
        self, initial_positions, current_positions
    ):
        """
        Constrains the updated positions to be an affine transformation of the initial scan positions,
        composing of two scale factors, a shear, and a rotation angle.

        Uses RANSAC to estimate the global transformation robustly.
        Stores the AffineTransformation in self._tf.

        Parameters
        ----------
        initial_positions: np.ndarray
            Initial scan positions
        current_positions: np.ndarray
            Current positions estimate

        Returns
        -------
        constrained_positions: np.ndarray
            Affine-transform constrained positions estimate
        """

        xp = self._xp

        tf, _ = estimate_global_transformation_ransac(
            positions0=initial_positions,
            positions1=current_positions,
            origin=self._positions_px_com,
            translation_allowed=True,
            min_sample=self._num_diffraction_patterns // 10,
            xp=xp,
        )

        self._tf = tf
        current_positions = tf(initial_positions, origin=self._positions_px_com, xp=xp)

        return current_positions
