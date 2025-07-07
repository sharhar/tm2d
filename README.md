# VkTM2D
A 2D template matcher written using the vkdispatch library.

## Installation

You can install the application using the command:

```
pip3 install .
```

## Imaging model

The projected potential $v_z(x,y)$ is modeled in one of two ways. It can be calculated from a slice through the Fourier transform of a density map (3D volume) or from a set of (3D) atomic coordinates. In the former case, different orientations are obtained by extracting different slices through the Fourier transform, while in the latter they are obtained by rotating the atomic coordinates, projecting along the $\hat{\mathbf{z}}$ (electron beam propagation) axis, and finally convolving the resulting 2D histogram of coordinates with a gaussian blur. The normalization of the gaussian kernel is set to approximately match a more rigorous calculation of the atomic projected potential in units of VÅ ([Kirkland 2020](https://doi.org/10.1007/978-3-030-33260-0)).

From the projected potential, the phase $\phi(x,y)=\sigma_ev_z(x,y)$ that is imparted to the exit wave function is calculated by scaling the projected potential by the interaction parameter $\sigma_e$, which for 300 kV accelerating voltage is approximately 0.652 mrad/VÅ. More generally, it is equal to $me\lambda_e/2\pi\hbar^2$, where $m$ is the electron mass, $e$ is the elementary charge, $\lambda_e$ is the wavelength of the electron, and $\hbar$ is the reduced Planck constant.

It is assumed that the projected potential $\phi$ is real and non-negative, with its 2D Fourier transform corresponding to the structure factor, denoted by $\Phi$:
```math
    \Phi(s_x, s_y) = \mathcal{F}[\phi](s_x, s_y).
```
Here, the coordinates $(s_x, s_y)$ denote (non-angular) spatial frequency. A spatial frequency $\mathbf{s}=(s_x,s_y)$ intersects the back focal plane at the physical coordinate $(f_\mathrm{OL}\lambda_e s_x, f_\mathrm{OL}\lambda_e s_y)$, where $f_\mathrm{OL}$ is the effective focal length of the objective lens.

The calculation of an image from the projected potential requires a contrast transfer function (CTF), denoted $\mathcal{C}$. The key relationship that enables calculation of an image $\mathfrak{m}(x, y)$ from the structure factor $\Phi$ and the CTF $\mathcal{C}$ is
```math
    \mathfrak{m}(x, y) \propto \mathcal{F}^{-1} [\Phi \cdot \mathcal{C}](x, y).
```
The CTF is assumed to have the form
```math
    \mathcal{C}(\mathbf{s}) = E_\mathrm{TC}(\mathbf{s}) E_\mathrm{SC}(\mathbf{s}) E_\mathrm{JN}(\mathbf{s}) E_\mathrm{B}(\mathbf{s}) \sin[\chi(\mathbf{s}) - \chi(\mathbf{0}) - \kappa].
```
In the above, the terms $E_j$ are various envelope functions, $\chi$ is the wavefront aberration, and $\kappa$ is the amplitude contrast, taken to be a small constant such as 0.07.

The temporal coherence envelope function is given by
```math
    E_\mathrm{TC}(\mathbf{s}) = \exp \left(-\frac{1}{2}\sigma_f^2\pi^2\lambda_e ^2|\mathbf{s}|^4\right)
```
where $\sigma_f$ is the "defocus spread," which is given by
```math
    \sigma_f = C_c\sqrt{ \left(\epsilon \frac{\sigma_{V_0}}{V_0}\right)^2 +  \left(\epsilon \frac{\sigma_{E_0}}{E_0}\right)^2 + \left(2\frac{\sigma_{I_0}}{I_0}\right)^2}
```
where $\Delta E_0 = 2\sqrt{2\ln 2}\sigma_{E_0}$ is the "energy spread" (typically quoted in FWHM, as opposed to the standard deviation $\sigma_{E_0}$), $C_c$ is the chromatic aberration coefficient, and $E_0$ is the electron energy. Relevant also are the objective lens current fluctuation $\sigma_{I_0}/I_0$ and accelerating voltage fluctation $\sigma_{V_0}/V_0$, which respectively have typical values of 0.1 ppm and 0.07 ppm. All the terms of the $\sigma_j$ denote standard deviations. The parameter $\epsilon$ is a relativistic correction factor given by
```math
    \epsilon = \frac{1 + \frac{E_0}{511 \text{ keV}}}{1 + \frac{E_0}{2\cdot 511 \text{ keV}}}
```
which is approximately 1.227 at 300 keV.

The spatial coherence envelope is given by
```math
    E_\mathrm{SC}(\mathbf{s}) = \exp\left[-2\sigma_\beta^2\pi^2|\mathbf{s}|^2\left( C_s\lambda_e ^2|\mathbf{s}|^2 - f_0 \right)^2 \right],
```
in which $C_s$ is the spherical aberration coefficient, $f_0$ is the mean defocus value, and $\sigma_\beta$ is the standard deviation of the distribution of beam angles (assumed to be gaussian, with a typical value of ~2.5 microradian) emerging from the gun.

The Johnson noise envelope is a gaussian envelope due to thermal magnetic field noise in the microscope column, with an associated variance $\sigma_\mathrm{J}^2$. This term is incorporated due to its potentially non-negligible values when using microscopes with relay optics ([Axelrod et al., Ultramicroscopy 2023](https://doi.org/10.1016/j.ultramic.2023.113730)). The functional form of this envelope is
```math
    E_\mathrm{JN}(\mathbf{s}) = \exp \left( -2\pi^2\sigma_J^2 |\mathbf{s}|^2 \right).
```

A generic B-factor envelope is added for convenience. It has the functional form
```math
    E_\mathrm{B}(\mathbf{s}) = \exp\left(-\frac{1}{4}B|\mathbf{s}|^2\right).
```

The wavefront aberration is given by
```math
    \chi(\mathbf{s}) = \frac{2\pi}{\lambda_e}\left( \frac{1}{4}C_s\lambda_e^4|\mathbf{s}|^4 - \frac{1}{2}f_0\lambda_e^2|\mathbf{s}|^2 \right) + \eta(\mathbf{s}),
```
where $f_0>0$ denotes _underfocus_. The term $\eta(\mathbf{s})$ is the phase shift due to a phase plate. For instance, an ideal Zernike phase plate imparts the phase shift
```math
    \eta(\mathbf{s}) = \frac{\pi}{2}\delta(\mathbf{s})
```
where $\delta(\mathbf{s})$ is the Dirac delta function.
