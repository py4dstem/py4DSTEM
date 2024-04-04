import warnings

import numpy as np
from py4DSTEM.process.phase.utils import (
    array_slice,
    estimate_global_transformation_ransac,
    fft_shift,
    fit_aberration_surface,
    regularize_probe_amplitude,
)
from py4DSTEM.process.utils import get_CoM

try:
    import cupy as cp
except (ModuleNotFoundError, ImportError):
    cp = np
    import os

    # make sure pylops doesn't try to use cupy
    os.environ["CUPY_PYLOPS"] = "0"
import pylops  # this must follow the exception


class ObjectNDConstraintsMixin:
    """
    Mixin class for object constraints applicable to 2D,2.5D, and 3D objects.
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

        if self._object_type == "complex":
            phase = xp.angle(current_object)

            if pure_phase_object:
                amplitude = 1.0
            else:
                amplitude = xp.minimum(xp.abs(current_object), 1.0)

            return amplitude * xp.exp(1.0j * phase)
        else:
            return current_object

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
        if self._object_type == "complex":
            return current_object
        else:
            return current_object.clip(0.0)

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
        gaussian_filter = self._scipy.ndimage.gaussian_filter
        gaussian_filter_sigma /= self.sampling[0]

        if not pure_phase_object or self._object_type == "potential":
            current_object = gaussian_filter(current_object, gaussian_filter_sigma)
        else:
            phase = xp.angle(current_object)
            phase = gaussian_filter(phase, gaussian_filter_sigma)
            current_object = xp.exp(1.0j * phase)

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
        qx = xp.fft.fftfreq(current_object.shape[-2], self.sampling[0])
        qy = xp.fft.fftfreq(current_object.shape[-1], self.sampling[1])

        qya, qxa = xp.meshgrid(qy, qx)
        qra = xp.sqrt(qxa**2 + qya**2)

        env = xp.ones_like(qra)
        if q_highpass:
            env *= 1 - 1 / (1 + (qra / q_highpass) ** (2 * butterworth_order))
        if q_lowpass:
            env *= 1 / (1 + (qra / q_lowpass) ** (2 * butterworth_order))

        current_object_mean = xp.mean(current_object, axis=(-2, -1), keepdims=True)
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
        if self._object_type == "complex":
            current_object_tv = current_object
            warnings.warn(
                (
                    "TV denoising is currently only supported for object_type=='potential'."
                ),
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
        padding,
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

        if self._object_type == "complex":
            updated_object = current_object
            warnings.warn(
                (
                    "TV denoising is currently only supported for object_type=='potential'."
                ),
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

            if padding is not None:
                pad_width = ((0, 0),) * current_object.ndim
                pad_width = list(pad_width)

                for ax in range(len(ndim)):
                    pad_width[ndim[ax]] = (padding, padding)

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

            if padding is not None:
                for ax in range(len(ndim)):
                    slices = array_slice(
                        ndim[ax], current_object.ndim, padding, -padding
                    )
                    updated_object = updated_object[slices]

            updated_object = (
                updated_object / xp.sum(updated_object) * current_object_sum
            )

        return updated_object

    def _object_constraints(
        self,
        current_object,
        gaussian_filter,
        gaussian_filter_sigma,
        pure_phase_object,
        butterworth_filter,
        butterworth_order,
        q_lowpass,
        q_highpass,
        tv_denoise,
        tv_denoise_weight,
        tv_denoise_inner_iter,
        object_positivity,
        shrinkage_rad,
        object_mask,
        **kwargs,
    ):
        """ObjectNDConstraints wrapper function"""

        # smoothness
        if gaussian_filter:
            current_object = self._object_gaussian_constraint(
                current_object, gaussian_filter_sigma, pure_phase_object
            )
        if butterworth_filter:
            current_object = self._object_butterworth_constraint(
                current_object,
                q_lowpass,
                q_highpass,
                butterworth_order,
            )
        if tv_denoise:
            current_object = self._object_denoise_tv_pylops(
                current_object, tv_denoise_weight, tv_denoise_inner_iter
            )

        # L1-norm pushing vacuum to zero
        if shrinkage_rad > 0.0 or object_mask is not None:
            current_object = self._object_shrinkage_constraint(
                current_object,
                shrinkage_rad,
                object_mask,
            )

        # amplitude threshold (complex) or positivity (potential)
        if self._object_type == "complex":
            current_object = self._object_threshold_constraint(
                current_object, pure_phase_object
            )
        elif object_positivity:
            current_object = self._object_positivity_constraint(current_object)

        return current_object


class Object2p5DConstraintsMixin:
    """
    Mixin class for object constraints unique to 2.5D objects.
    Overwrites ObjectNDConstraintsMixin.
    """

    def _object_denoise_tv_pylops(self, current_object, weights, iterations, z_padding):
        """
        Performs second order TV denoising along x and y, and first order along z

        Parameters
        ----------
        current_object: np.ndarray
            Current object estimate
        weights : [float, float]
            Denoising weights[z_weight, r_weight]. The greater `weight`,
            the more denoising.
        iterations: float
            Number of iterations to run in denoising algorithm.
            `niter_out` in pylops
        z_padding: int
            Symmetric padding around the first axis

        Returns
        -------
        constrained_object: np.ndarray
            Constrained object estimate

        """
        xp = self._xp

        if self._object_type == "complex":
            current_object_tv = current_object
            warnings.warn(
                (
                    "TV denoising is currently only supported for object_type=='potential'."
                ),
                UserWarning,
            )

        else:
            # zero pad at top and bottom slice
            pad_width = ((z_padding, z_padding), (0, 0), (0, 0))
            current_object = xp.pad(
                current_object, pad_width=pad_width, mode="constant"
            )

            # run tv denoising
            nz, nx, ny = current_object.shape
            niter_out = iterations
            niter_in = 1
            Iop = pylops.Identity(nx * ny * nz)

            if weights[0] == 0:
                xy_laplacian = pylops.Laplacian(
                    (nz, nx, ny), axes=(1, 2), edge=False, kind="backward"
                )
                l1_regs = [xy_laplacian]

                current_object_tv = pylops.optimization.sparsity.splitbregman(
                    Op=Iop,
                    y=current_object.ravel(),
                    RegsL1=l1_regs,
                    niter_outer=niter_out,
                    niter_inner=niter_in,
                    epsRL1s=[weights[1]],
                    tol=1e-4,
                    tau=1.0,
                    show=False,
                )[0]

            elif weights[1] == 0:
                z_gradient = pylops.FirstDerivative(
                    (nz, nx, ny), axis=0, edge=False, kind="backward"
                )
                l1_regs = [z_gradient]

                current_object_tv = pylops.optimization.sparsity.splitbregman(
                    Op=Iop,
                    y=current_object.ravel(),
                    RegsL1=l1_regs,
                    niter_outer=niter_out,
                    niter_inner=niter_in,
                    epsRL1s=[weights[0]],
                    tol=1e-4,
                    tau=1.0,
                    show=False,
                )[0]

            else:
                z_gradient = pylops.FirstDerivative(
                    (nz, nx, ny), axis=0, edge=False, kind="backward"
                )
                xy_laplacian = pylops.Laplacian(
                    (nz, nx, ny), axes=(1, 2), edge=False, kind="backward"
                )
                l1_regs = [z_gradient, xy_laplacian]

                current_object_tv = pylops.optimization.sparsity.splitbregman(
                    Op=Iop,
                    y=current_object.ravel(),
                    RegsL1=l1_regs,
                    niter_outer=niter_out,
                    niter_inner=niter_in,
                    epsRL1s=weights,
                    tol=1e-4,
                    tau=1.0,
                    show=False,
                )[0]

            # remove padding
            current_object_tv = current_object_tv.reshape(current_object.shape)[
                z_padding:-z_padding
            ]

        return current_object_tv

    def _object_kz_regularization_constraint(
        self, current_object, kz_regularization_gamma, z_padding
    ):
        """
        Arctan regularization filter

        Parameters
        --------
        current_object: np.ndarray
            Current object estimate
        kz_regularization_gamma: float
            Slice regularization strength
        z_padding: int
            Symmetric padding around the first axis

        Returns
        --------
        constrained_object: np.ndarray
            Constrained object estimate
        """
        xp = self._xp

        # zero pad at top and bottom slice
        pad_width = ((z_padding, z_padding), (0, 0), (0, 0))
        current_object = xp.pad(current_object, pad_width=pad_width, mode="constant")

        qz = xp.fft.fftfreq(current_object.shape[0], self._slice_thicknesses[0])
        qx = xp.fft.fftfreq(current_object.shape[1], self.sampling[0])
        qy = xp.fft.fftfreq(current_object.shape[2], self.sampling[1])

        kz_regularization_gamma *= self._slice_thicknesses[0] / self.sampling[0]

        qza, qxa, qya = xp.meshgrid(qz, qx, qy, indexing="ij")
        qz2 = qza**2 * kz_regularization_gamma**2
        qr2 = qxa**2 + qya**2

        w = 1 - 2 / np.pi * xp.arctan2(qz2, qr2)

        current_object = xp.fft.ifftn(xp.fft.fftn(current_object) * w)
        current_object = current_object[z_padding:-z_padding]

        if self._object_type == "potential":
            current_object = xp.real(current_object)

        return current_object

    def _object_identical_slices_constraint(self, current_object):
        """
        Strong regularization forcing all slices to be identical

        Parameters
        --------
        current_object: np.ndarray
            Current object estimate

        Returns
        --------
        constrained_object: np.ndarray
            Constrained object estimate
        """
        object_mean = current_object.mean(0, keepdims=True)
        current_object[:] = object_mean

        return current_object

    def _object_constraints(
        self,
        current_object,
        gaussian_filter,
        gaussian_filter_sigma,
        pure_phase_object,
        butterworth_filter,
        butterworth_order,
        q_lowpass,
        q_highpass,
        identical_slices,
        kz_regularization_filter,
        kz_regularization_gamma,
        tv_denoise,
        tv_denoise_weights,
        tv_denoise_inner_iter,
        tv_denoise_chambolle,
        tv_denoise_weight_chambolle,
        tv_denoise_pad_chambolle,
        object_positivity,
        shrinkage_rad,
        object_mask,
        **kwargs,
    ):
        """Object2p5DConstraints wrapper function"""

        # smoothness
        if gaussian_filter:
            current_object = self._object_gaussian_constraint(
                current_object, gaussian_filter_sigma, pure_phase_object
            )
        if butterworth_filter:
            current_object = self._object_butterworth_constraint(
                current_object,
                q_lowpass,
                q_highpass,
                butterworth_order,
            )
        if identical_slices:
            current_object = self._object_identical_slices_constraint(current_object)
        elif kz_regularization_filter:
            current_object = self._object_kz_regularization_constraint(
                current_object,
                kz_regularization_gamma,
                z_padding=1,
            )
        elif tv_denoise:
            current_object = self._object_denoise_tv_pylops(
                current_object,
                tv_denoise_weights,
                tv_denoise_inner_iter,
                z_padding=1,
            )
        elif tv_denoise_chambolle:
            current_object = self._object_denoise_tv_chambolle(
                current_object,
                tv_denoise_weight_chambolle,
                axis=0,
                padding=tv_denoise_pad_chambolle,
            )

        # L1-norm pushing vacuum to zero
        if shrinkage_rad > 0.0 or object_mask is not None:
            current_object = self._object_shrinkage_constraint(
                current_object,
                shrinkage_rad,
                object_mask,
            )

        # amplitude threshold (complex) or positivity (potential)
        if self._object_type == "complex":
            current_object = self._object_threshold_constraint(
                current_object, pure_phase_object
            )
        elif object_positivity:
            current_object = self._object_positivity_constraint(current_object)

        return current_object


class Object3DConstraintsMixin:
    """
    Mixin class for object constraints unique to 3D objects.
    Overwrites ObjectNDConstraintsMixin and Object2p5DConstraintsMixin.
    """

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
        if self._object_type == "complex":
            current_object_tv = current_object
            warnings.warn(
                (
                    "TV denoising is currently only supported for object_type=='potential'."
                ),
                UserWarning,
            )

        else:
            nz, nx, ny = current_object.shape
            niter_out = iterations
            niter_in = 1
            Iop = pylops.Identity(nx * ny * nz)
            xyz_laplacian = pylops.Laplacian(
                (nz, nx, ny), axes=(0, 1, 2), edge=False, kind="backward"
            )

            l1_regs = [xyz_laplacian]

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

    def _object_butterworth_constraint(
        self, current_object, q_lowpass, q_highpass, butterworth_order
    ):
        """
        Butterworth filter

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
        qz = xp.fft.fftfreq(current_object.shape[0], self.sampling[1])
        qx = xp.fft.fftfreq(current_object.shape[1], self.sampling[0])
        qy = xp.fft.fftfreq(current_object.shape[2], self.sampling[1])
        qza, qxa, qya = xp.meshgrid(qz, qx, qy, indexing="ij")
        qra = xp.sqrt(qza**2 + qxa**2 + qya**2)

        env = xp.ones_like(qra)
        if q_highpass:
            env *= 1 - 1 / (1 + (qra / q_highpass) ** (2 * butterworth_order))
        if q_lowpass:
            env *= 1 / (1 + (qra / q_lowpass) ** (2 * butterworth_order))

        current_object_mean = xp.mean(current_object)
        current_object -= current_object_mean
        current_object = xp.fft.ifftn(xp.fft.fftn(current_object) * env)
        current_object += current_object_mean

        if self._object_type == "potential":
            current_object = xp.real(current_object)

        return current_object

    def _object_constraints(
        self,
        current_object,
        gaussian_filter,
        gaussian_filter_sigma,
        butterworth_filter,
        butterworth_order,
        q_lowpass,
        q_highpass,
        tv_denoise,
        tv_denoise_weights,
        tv_denoise_inner_iter,
        object_positivity,
        shrinkage_rad,
        object_mask,
        **kwargs,
    ):
        """Object3DConstraints wrapper function"""

        # smoothness
        if gaussian_filter:
            current_object = self._object_gaussian_constraint(
                current_object, gaussian_filter_sigma, pure_phase_object=False
            )
        if butterworth_filter:
            current_object = self._object_butterworth_constraint(
                current_object,
                q_lowpass,
                q_highpass,
                butterworth_order,
            )
        if tv_denoise:
            current_object = self._object_denoise_tv_pylops(
                current_object,
                tv_denoise_weights,
                tv_denoise_inner_iter,
            )

        # L1-norm pushing vacuum to zero
        if shrinkage_rad > 0.0 or object_mask is not None:
            current_object = self._object_shrinkage_constraint(
                current_object,
                shrinkage_rad,
                object_mask,
            )

        # Positivity
        if object_positivity:
            current_object = self._object_positivity_constraint(current_object)

        return current_object


class ProbeConstraintsMixin:
    """
    Mixin class for regularizations applicable to a single probe.
    """

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
        erf = self._scipy.special.erf

        probe_intensity = xp.abs(current_probe) ** 2
        current_probe_sum = xp.sum(probe_intensity)

        X = xp.fft.fftfreq(current_probe.shape[0])[:, None]
        Y = xp.fft.fftfreq(current_probe.shape[1])[None]
        r = xp.hypot(X, Y) - relative_radius

        sigma = np.sqrt(np.pi) / relative_width
        tophat_mask = 0.5 * (1 - erf(sigma * r / (1 - r**2)))

        updated_probe = current_probe * tophat_mask
        updated_probe_sum = xp.sum(xp.abs(updated_probe) ** 2)
        normalization = xp.sqrt(current_probe_sum / updated_probe_sum)

        return updated_probe * normalization

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

        current_probe_sum = xp.sum(xp.abs(current_probe) ** 2)
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
        updated_probe_sum = xp.sum(xp.abs(updated_probe) ** 2)
        normalization = xp.sqrt(current_probe_sum / updated_probe_sum)

        return updated_probe * normalization

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

        current_probe_sum = xp.sum(xp.abs(current_probe) ** 2)
        current_probe_fft_phase = xp.angle(xp.fft.fft2(current_probe))

        updated_probe = xp.fft.ifft2(
            xp.exp(1j * current_probe_fft_phase) * initial_probe_aperture
        )
        updated_probe_sum = xp.sum(xp.abs(updated_probe) ** 2)
        normalization = xp.sqrt(current_probe_sum / updated_probe_sum)

        return updated_probe * normalization

    def _probe_aberration_fitting_constraint(
        self,
        current_probe,
        max_angular_order,
        max_radial_order,
        remove_initial_probe_aberrations,
        use_scikit_image,
    ):
        """
        Ptychographic probe smoothing constraint.

        Parameters
        ----------
        current_probe: np.ndarray
            Current positions estimate
        max_angular_order: bool
            Max angular order of probe aberrations basis functions
        max_radial_order: bool
            Max radial order of probe aberrations basis functions
        remove_initial_probe_aberrations: bool, optional
            If true, initial probe aberrations are removed before fitting

        Returns
        --------
        constrained_probe: np.ndarray
            Constrained probe estimate
        """

        xp = self._xp

        fourier_probe = xp.fft.fft2(current_probe)
        if remove_initial_probe_aberrations:
            fourier_probe *= xp.conj(self._known_aberrations_array)

        fourier_probe_abs = xp.abs(fourier_probe)
        sampling = self.sampling
        energy = self._energy

        fitted_angle, _ = fit_aberration_surface(
            fourier_probe,
            sampling,
            energy,
            max_angular_order,
            max_radial_order,
            use_scikit_image,
            xp=xp,
        )

        fourier_probe = fourier_probe_abs * xp.exp(-1.0j * fitted_angle)
        if remove_initial_probe_aberrations:
            fourier_probe *= self._known_aberrations_array

        current_probe = xp.fft.ifft2(fourier_probe)

        return current_probe

    def _probe_constraints(
        self,
        current_probe,
        fix_probe_com,
        fit_probe_aberrations,
        fit_probe_aberrations_max_angular_order,
        fit_probe_aberrations_max_radial_order,
        fit_probe_aberrations_remove_initial,
        fit_probe_aberrations_using_scikit_image,
        fix_probe_aperture,
        initial_probe_aperture,
        constrain_probe_fourier_amplitude,
        constrain_probe_fourier_amplitude_max_width_pixels,
        constrain_probe_fourier_amplitude_constant_intensity,
        constrain_probe_amplitude,
        constrain_probe_amplitude_relative_radius,
        constrain_probe_amplitude_relative_width,
        **kwargs,
    ):
        """ProbeConstraints wrapper function"""

        # CoM corner-centering
        if fix_probe_com:
            current_probe = self._probe_center_of_mass_constraint(current_probe)

        # Fourier phase (aberrations) fitting
        if fit_probe_aberrations:
            current_probe = self._probe_aberration_fitting_constraint(
                current_probe,
                fit_probe_aberrations_max_angular_order,
                fit_probe_aberrations_max_radial_order,
                fit_probe_aberrations_remove_initial,
                fit_probe_aberrations_using_scikit_image,
            )

        # Fourier amplitude (aperture) constraints
        if fix_probe_aperture:
            current_probe = self._probe_aperture_constraint(
                current_probe,
                initial_probe_aperture,
            )
        elif constrain_probe_fourier_amplitude:
            current_probe = self._probe_fourier_amplitude_constraint(
                current_probe,
                constrain_probe_fourier_amplitude_max_width_pixels,
                constrain_probe_fourier_amplitude_constant_intensity,
            )

        # Real-space amplitude constraint
        if constrain_probe_amplitude:
            current_probe = self._probe_amplitude_constraint(
                current_probe,
                constrain_probe_amplitude_relative_radius,
                constrain_probe_amplitude_relative_width,
            )

        return current_probe


class ProbeMixedConstraintsMixin:
    """
    Mixin class for regularizations unique to mixed probes.
    Overwrites ProbeConstraintsMixin.
    """

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
        probe_intensity = xp.abs(current_probe[0]) ** 2

        probe_x0, probe_y0 = get_CoM(
            probe_intensity, device=self._device, corner_centered=True
        )
        shifted_probe = fft_shift(current_probe, -xp.array([probe_x0, probe_y0]), xp)

        return shifted_probe

    def _probe_orthogonalization_constraint(self, current_probe):
        """
        Ptychographic probe-orthogonalization constraint.
        Used to ensure mixed states are orthogonal to each other.
        Adapted from https://github.com/AdvancedPhotonSource/tike/blob/main/src/tike/ptycho/probe.py#L690

        Parameters
        --------
        current_probe: np.ndarray
            Current probe estimate

        Returns
        --------
        constrained_probe: np.ndarray
            Orthogonalized probe estimate
        """
        xp = self._xp
        n_probes = self._num_probes

        # compute upper half of P* @ P
        pairwise_dot_product = xp.empty((n_probes, n_probes), dtype=current_probe.dtype)

        for i in range(n_probes):
            for j in range(i, n_probes):
                pairwise_dot_product[i, j] = xp.sum(
                    current_probe[i].conj() * current_probe[j]
                )

        # compute eigenvectors (effectively cheaper way of computing V* from SVD)
        _, evecs = xp.linalg.eigh(pairwise_dot_product, UPLO="U")
        current_probe = xp.tensordot(evecs.T, current_probe, axes=1)

        # sort by real-space intensity
        intensities = xp.sum(xp.abs(current_probe) ** 2, axis=(-2, -1))
        intensities_order = xp.argsort(intensities, axis=None)[::-1]
        return current_probe[intensities_order]

    def _probe_constraints(
        self,
        current_probe,
        fix_probe_com,
        fit_probe_aberrations,
        fit_probe_aberrations_max_angular_order,
        fit_probe_aberrations_max_radial_order,
        fit_probe_aberrations_remove_initial,
        fit_probe_aberrations_using_scikit_image,
        num_probes_fit_aberrations,
        fix_probe_aperture,
        initial_probe_aperture,
        constrain_probe_fourier_amplitude,
        constrain_probe_fourier_amplitude_max_width_pixels,
        constrain_probe_fourier_amplitude_constant_intensity,
        constrain_probe_amplitude,
        constrain_probe_amplitude_relative_radius,
        constrain_probe_amplitude_relative_width,
        orthogonalize_probe,
        **kwargs,
    ):
        """ProbeMixedConstraints wrapper function"""

        # CoM corner-centering
        if fix_probe_com:
            current_probe = self._probe_center_of_mass_constraint(current_probe)

        # Fourier phase (aberrations) fitting
        if fit_probe_aberrations:
            if num_probes_fit_aberrations > self._num_probes:
                num_probes_fit_aberrations = self._num_probes
            for probe_idx in range(num_probes_fit_aberrations):
                current_probe[probe_idx] = self._probe_aberration_fitting_constraint(
                    current_probe[probe_idx],
                    fit_probe_aberrations_max_angular_order,
                    fit_probe_aberrations_max_radial_order,
                    fit_probe_aberrations_remove_initial,
                    fit_probe_aberrations_using_scikit_image,
                )

        # Fourier amplitude (aperture) constraints
        if fix_probe_aperture:
            current_probe[0] = self._probe_aperture_constraint(
                current_probe[0],
                initial_probe_aperture[0],
            )
        elif constrain_probe_fourier_amplitude:
            current_probe[0] = self._probe_fourier_amplitude_constraint(
                current_probe[0],
                constrain_probe_fourier_amplitude_max_width_pixels,
                constrain_probe_fourier_amplitude_constant_intensity,
            )

        # Real-space amplitude constraint
        if constrain_probe_amplitude:
            for probe_idx in range(self._num_probes):
                current_probe[probe_idx] = self._probe_amplitude_constraint(
                    current_probe[probe_idx],
                    constrain_probe_amplitude_relative_radius,
                    constrain_probe_amplitude_relative_width,
                )

        # Probe orthogonalization
        if orthogonalize_probe:
            current_probe = self._probe_orthogonalization_constraint(current_probe)

        return current_probe


class PositionsConstraintsMixin:
    """
    Mixin class for probe positions constraints.
    """

    def _positions_center_of_mass_constraint(
        self, current_positions, initial_positions_com
    ):
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
        current_positions -= current_positions.mean(0) - initial_positions_com

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

        xp_storage = self._xp_storage
        initial_positions_com = initial_positions.mean(0)

        tf, _ = estimate_global_transformation_ransac(
            positions0=initial_positions,
            positions1=current_positions,
            origin=initial_positions_com,
            translation_allowed=True,
            min_sample=initial_positions.shape[0] // 10,
            xp=xp_storage,
        )

        current_positions = tf(
            initial_positions, origin=initial_positions_com, xp=xp_storage
        )
        self._tf = tf

        return current_positions

    def _positions_constraints(
        self,
        current_positions,
        initial_positions,
        fix_positions,
        fix_positions_com,
        global_affine_transformation,
        **kwargs,
    ):
        """PositionsConstraints wrapper function"""

        if not fix_positions:
            if not fix_positions_com:
                current_positions = self._positions_center_of_mass_constraint(
                    current_positions, initial_positions.mean(0)
                )

            if global_affine_transformation:
                current_positions = self._positions_affine_transformation_constraint(
                    initial_positions, current_positions
                )

        return current_positions
