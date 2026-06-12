import numpy as np
from astropy.io import fits
from scipy.ndimage import shift as ndshift
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.widgets import Slider, RadioButtons, Button

targ1 = fits.getdata('./ocjc45030_flt.fits')
psf1 = fits.getdata('./ocjc47030_flt.fits')
targ2 = fits.getdata('./ocjc44030_flt.fits')
psf2 = fits.getdata('./ocjc43030_flt.fits')

targets = {
    'HR-4796A #1 (ocjc45030)': targ1,
    'HR-4796A #2 (ocjc44030)': targ2,
}
references = {
    'PSF #1 (ocjc47030)': psf1,
    'PSF #2 (ocjc43030)': psf2,
}

def compute_residual(target, ref, dx, dy, scale):
    shifted = ndshift(ref, shift=(dy, dx), order=1, mode='constant', cval=np.nanmedian(ref))
    scaled = shifted * scale
    residual = target - scaled
    return shifted, scaled, residual

def update(val=None):
    tkey = targ_radio.value_selected
    rkey = ref_radio.value_selected
    target = targets[tkey]
    ref_data = references[rkey]

    dx = dx_slider.val
    dy = dy_slider.val
    scale = scale_slider.val

    shifted_ref, scaled_ref, residual = compute_residual(target, ref_data, dx, dy, scale)

    ax_target_images.set_data(target)
    ax_target.set_title(f'Target: {tkey}')

    ax_ref_images.set_data(scaled_ref)
    ax_ref.set_title(f'Reference: {rkey}')

    ax_res_images.set_data(residual)
    ax_res.set_title('RDI Residual')

    fig.canvas.draw_idle()

fig = plt.figure(figsize=(12, 9))
fig.subplots_adjust(left=0.07, right=0.97, top=0.92, bottom=0.08, wspace=0.3, hspace=0.3)

ax_target = fig.add_subplot(2, 2, 1)
ax_ref = fig.add_subplot(2, 2, 2)
ax_res = fig.add_subplot(2, 2, 3)

t0 = list(targets.values())[0]
r0 = list(references.values())[0]

norm_log = LogNorm(vmin=1, vmax=3000)
ax_target_images = ax_target.imshow(t0, origin='lower', norm=norm_log, cmap='inferno')
ax_target.set_title('Target: HR-4796A #1')
ax_target.set_xlabel('x (pix)')
ax_target.set_ylabel('y (pix)')

r0_shifted = ndshift(r0, shift=(0, 0), order=1, mode='constant', cval=np.nanmedian(r0))
ax_ref_images = ax_ref.imshow(r0_shifted, origin='lower', norm=norm_log, cmap='inferno')
ax_ref.set_title('Reference: PSF #1')
ax_ref.set_xlabel('x (pix)')
ax_ref.set_ylabel('y (pix)')

res0 = t0 - r0_shifted
ax_res_images = ax_res.imshow(res0, origin='lower', cmap='RdBu_r', vmin=-150, vmax=150)
ax_res.set_title('RDI Residual')
ax_res.set_xlabel('x (pix)')
ax_res.set_ylabel('y (pix)')

plt.colorbar(ax_target_images, ax=ax_target, fraction=0.046, pad=0.04)
plt.colorbar(ax_ref_images, ax=ax_ref, fraction=0.046, pad=0.04)
plt.colorbar(ax_res_images, ax=ax_res, fraction=0.046, pad=0.04)

ax_controls = fig.add_subplot(2, 2, 4)
ax_controls.axis('off')

ax_targ_radio = fig.add_axes([0.55, 0.38, 0.17, 0.10])
targ_radio = RadioButtons(ax_targ_radio, list(targets.keys()), active=0)
targ_radio.on_clicked(update)

ax_ref_radio = fig.add_axes([0.76, 0.38, 0.17, 0.10])
ref_radio = RadioButtons(ax_ref_radio, list(references.keys()), active=0)
ref_radio.on_clicked(update)

ax_dx = fig.add_axes([0.645, 0.27, 0.19, 0.04])
dx_slider = Slider(ax_dx, 'dx offset (pix)', -3, 3, valinit=0.0, valstep=0.02)
dx_slider.on_changed(update)

ax_dy = fig.add_axes([0.645, 0.20, 0.19, 0.04])
dy_slider = Slider(ax_dy, 'dy offset (pix)', -3, 3, valinit=0.0, valstep=0.02)
dy_slider.on_changed(update)

ax_scale = fig.add_axes([0.645, 0.13, 0.19, 0.04])
scale_slider = Slider(ax_scale, 'scale factor', 0.8, 1.2, valinit=1.0, valstep=0.01)
scale_slider.on_changed(update)

def reset(val=None):
    dx_slider.set_val(0.0)
    dy_slider.set_val(0.0)
    scale_slider.set_val(1.0)

ax_reset = fig.add_axes([0.645, 0.07, 0.19, 0.04])
reset_button = Button(ax_reset, 'Reset')
reset_button.on_clicked(reset)

fig.suptitle('HST STIS RDI Post-Processing Demo', fontsize=14, fontweight='bold')

plt.show()
