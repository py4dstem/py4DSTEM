from py4DSTEM.process.phase.utils import (
    estimate_global_transformation_ransac,
    fft_shift,
)
from py4DSTEM.process.utils import get_CoM


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
    phase = xp.exp(1.0j * xp.angle(current_object))
    if pure_phase_object:
        amplitude = 1.0
    else:
        amplitude = xp.minimum(xp.abs(current_object), 1.0)
    return amplitude * phase


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
        Standard deviation of gaussian kernel
    pure_phase_object: bool
        If True, gaussian blur performed on phase only

    Returns
    --------
    constrained_object: np.ndarray
        Constrained object estimate
    """
    xp = self._xp
    gaussian_filter = self._gaussian_filter

    if pure_phase_object:
        phase = xp.angle(current_object)
        phase = gaussian_filter(phase, gaussian_filter_sigma)
        current_object = xp.exp(1.0j * phase)
    else:
        current_object = gaussian_filter(current_object, gaussian_filter_sigma)

    return current_object


def _object_butterworth_constraint(self, current_object, q_lowpass, q_highpass):
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

    Returns
    --------
    constrained_object: np.ndarray
        Constrained object estimate
    """
    xp = self._xp
    qx = xp.fft.fftfreq(
        current_object.shape[0], self.sampling[0] if self._energy is not None else 1
    )
    qy = xp.fft.fftfreq(
        current_object.shape[1], self.sampling[1] if self._energy is not None else 1
    )
    qya, qxa = xp.meshgrid(qy, qx)
    qra = xp.sqrt(qxa**2 + qya**2)

    env = xp.ones_like(qra)
    if q_highpass:
        env *= 1 - 1 / (1 + (qra / q_highpass) ** 4)
    if q_lowpass:
        env *= 1 / (1 + (qra / q_lowpass) ** 4)

    current_object = xp.fft.ifft2(xp.fft.fft2(current_object) * env)
    return current_object


def _probe_center_of_mass_constraint(self, current_probe):
    """
    Ptychographic threshold constraint.
    Used for avoiding the scaling ambiguity between probe and object.

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
    asnumpy = self._asnumpy

    probe_center = xp.array(self._region_of_interest_shape) / 2
    probe_intensity = asnumpy(xp.abs(current_probe) ** 2)

    probe_x0, probe_y0 = get_CoM(probe_intensity)
    shifted_probe = fft_shift(
        current_probe, probe_center - xp.array([probe_x0, probe_y0]), xp
    )

    return shifted_probe


def _probe_fourier_amplitude_constraint(self, current_probe):
    """
    Ptychographic probe Fourier-amplitude constraint.
    Used for fixing the probe's amplitude in Fourier space.

    Parameters
    --------
    current_probe: np.ndarray
        Current probe estimate

    Returns
    --------
    constrained_probe: np.ndarray
        Fourier-amplitude constrained probe estimate
    """
    xp = self._xp

    current_probe_fft = xp.fft.fft2(current_probe)
    current_probe_fft_phase = xp.angle(current_probe_fft)

    constrained_probe_fft = self._probe_initial_fft_amplitude * xp.exp(
        1j * current_probe_fft_phase
    )
    constrained_probe = xp.fft.ifft2(constrained_probe_fft)

    return constrained_probe


def _probe_finite_support_constraint(self, current_probe):
    """
    Ptychographic probe support constraint.
    Used for penalizing focused probes to replicate sample periodicity.

    Parameters
    --------
    current_probe: np.ndarray
        Current probe estimate

    Returns
    --------
    constrained_probe: np.ndarray
        Finite-support constrained probe estimate
    """

    return current_probe * self._probe_support_mask


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
