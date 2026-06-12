import numpy as np
from astropy.io import fits
from scipy.ndimage import shift as ndshift, rotate as ndrotate
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.widgets import Slider, Button

targ1 = fits.getdata('./ocjc45030_flt.fits')
targ2 = fits.getdata('./ocjc44030_flt.fits')

TARGETS = {
    'HR-4796A #1 (ocjc45030)': targ1,
    'HR-4796A #2 (ocjc44030)': targ2,
}

ORIENTAT = {
    'ocjc45030': 45.516,
    'ocjc44030': -83.0777,
}

# Centers
C1 = np.array([50.02, 56.08])
C2 = np.array([48.72, 56.15])

H, W = targ1.shape
cy, cx = (H - 1) / 2.0, (W - 1) / 2.0

ang1 = -ORIENTAT['ocjc45030']
ang2 = -ORIENTAT['ocjc44030']

def rotate_coord(r, c, angle):
    theta = np.radians(angle)
    dr = r - cy
    dc = c - cx
    rr = cy + dr * np.cos(theta) - dc * np.sin(theta)
    cc = cx + dr * np.sin(theta) + dc * np.cos(theta)
    return rr, cc

c1r, c1c = rotate_coord(C1[0], C1[1], ang1)
c2r, c2c = rotate_coord(C2[0], C2[1], ang2)
SHIFT_DY = c1r - c2r
SHIFT_DX = c1c - c2c
MASK_CENTER = (C1[0], C1[1])

def x_mask(h, w, center, width):
    half = width / 2.0
    yy, xx = np.ogrid[:h, :w]
    d1 = np.abs((yy - center[0]) - (xx - center[1])) / np.sqrt(2)
    d2 = np.abs((yy - center[0]) + (xx - center[1])) / np.sqrt(2)
    in_x = (d1 < half) | (d2 < half)
    return ~in_x

def apply_mask_and_rotate(res, mask, angle):
    res_m = res.copy()
    res_m[~mask] = 0
    wt = mask.astype(np.float64)
    rotated = ndrotate(res_m, angle, reshape=False, order=1, mode='constant', cval=0)
    wrot = ndrotate(wt, angle, reshape=False, order=1, mode='constant', cval=0)
    return rotated, wrot

def compute_adi(t1, t2, dx1, dy1, s1, dx2, dy2, s2, mask_w):
    t1_s = ndshift(t1, (dy1, dx1), order=1, mode='constant', cval=np.nanmedian(t1)) * s1
    t2_s = ndshift(t2, (dy2, dx2), order=1, mode='constant', cval=np.nanmedian(t2)) * s2

    res_fwd = t1 - t2_s
    res_bwd = t2 - t1_s

    if mask_w > 0:
        m = x_mask(H, W, MASK_CENTER, mask_w)
        res_fwd_disp = res_fwd.copy()
        res_bwd_disp = res_bwd.copy()
        res_fwd_disp[~m] = np.nan
        res_bwd_disp[~m] = np.nan
        derot_fwd, w_fwd = apply_mask_and_rotate(res_fwd, m, ang1)
        derot_bwd, w_bwd = apply_mask_and_rotate(res_bwd, m, ang2)
        derot_bwd_reg = ndshift(derot_bwd, (SHIFT_DY, SHIFT_DX), order=1, mode='constant', cval=0)
        w_bwd_reg = ndshift(w_bwd, (SHIFT_DY, SHIFT_DX), order=1, mode='constant', cval=0)
        wt_sum = w_fwd + w_bwd_reg
        combined = np.divide(derot_fwd + derot_bwd_reg, wt_sum, where=wt_sum > 0)
        combined[wt_sum == 0] = np.nan
    else:
        derot_fwd = ndrotate(res_fwd, ang1, reshape=False, order=1, mode='constant', cval=np.nanmedian(res_fwd))
        derot_bwd = ndrotate(res_bwd, ang2, reshape=False, order=1, mode='constant', cval=np.nanmedian(res_bwd))
        derot_bwd_reg = ndshift(derot_bwd, (SHIFT_DY, SHIFT_DX), order=1, mode='constant', cval=np.nanmedian(derot_bwd))
        combined = (derot_fwd + derot_bwd_reg) / 2
        res_fwd_disp = res_fwd
        res_bwd_disp = res_bwd

    return t1_s, t2_s, res_fwd_disp, res_bwd_disp, combined

def update(val=None):
    dx1 = dx1_slider.val
    dy1 = dy1_slider.val
    s1 = scale1_slider.val
    dx2 = dx2_slider.val
    dy2 = dy2_slider.val
    s2 = scale2_slider.val
    mw = mask_width_slider.val

    t1_s, t2_s, res_fwd, res_bwd, combined = compute_adi(targ1, targ2, dx1, dy1, s1, dx2, dy2, s2, mw)

    ax1_img.set_data(t1_s)
    ax1.set_title(f'Target 1 shifted (dx={dx1:.2f}, dy={dy1:.2f}, s={s1:.2f})')
    ax1_img.set_norm(LogNorm(vmin=1, vmax=3000))

    ax2_img.set_data(t2_s)
    ax2.set_title(f'Target 2 shifted (dx={dx2:.2f}, dy={dy2:.2f}, s={s2:.2f})')
    ax2_img.set_norm(LogNorm(vmin=1, vmax=3000))

    ax3_img.set_data(combined)
    ax3.set_title('ADI Mean Combined (derotated)')

    ax4_img.set_data(res_fwd)
    ax4.set_title('T1 - T2 (pre-derotation)')
    ax4_img.set_clim(-150, 150)

    ax5_img.set_data(res_bwd)
    ax5.set_title('T2 - T1 (pre-derotation)')
    ax5_img.set_clim(-150, 150)

    fig.canvas.draw_idle()

fig = plt.figure(figsize=(18, 11))
fig.subplots_adjust(left=0.05, right=0.97, top=0.94, bottom=0.10, wspace=0.25, hspace=0.30)

norm_log = LogNorm(vmin=1, vmax=3000)
cmap_img = 'inferno'
cmap_res = plt.cm.RdBu_r.copy()

ax1 = fig.add_subplot(2, 3, 1)
ax1_img = ax1.imshow(targ1, origin='lower', norm=norm_log, cmap=cmap_img)
ax1.set_title('Target 1 (45030)')
ax1.set_xlabel('x (pix)'); ax1.set_ylabel('y (pix)')
plt.colorbar(ax1_img, ax=ax1, fraction=0.046, pad=0.04)

ax2 = fig.add_subplot(2, 3, 2)
ax2_img = ax2.imshow(targ2, origin='lower', norm=norm_log, cmap=cmap_img)
ax2.set_title('Target 2 (44030)')
ax2.set_xlabel('x (pix)'); ax2.set_ylabel('y (pix)')
plt.colorbar(ax2_img, ax=ax2, fraction=0.046, pad=0.04)

ax3 = fig.add_subplot(2, 3, 3)
ax3_img = ax3.imshow(np.zeros_like(targ1), origin='lower', cmap=cmap_res, vmin=-150, vmax=150)
ax3_img.cmap.set_bad(color='gray', alpha=0.3)
ax3.set_title('ADI Mean Combined')
ax3.set_xlabel('x (pix)'); ax3.set_ylabel('y (pix)')
plt.colorbar(ax3_img, ax=ax3, fraction=0.046, pad=0.04)

res0 = targ1 - targ2
ax4 = fig.add_subplot(2, 3, 4)
ax4_img = ax4.imshow(res0, origin='lower', cmap=cmap_res, vmin=-150, vmax=150)
ax4_img.cmap.set_bad(color='gray', alpha=0.3)
ax4.set_title('T1 - T2 (pre-derotation)')
ax4.set_xlabel('x (pix)'); ax4.set_ylabel('y (pix)')
plt.colorbar(ax4_img, ax=ax4, fraction=0.046, pad=0.04)

ax5 = fig.add_subplot(2, 3, 5)
ax5_img = ax5.imshow(-res0, origin='lower', cmap=cmap_res, vmin=-150, vmax=150)
ax5_img.cmap.set_bad(color='gray', alpha=0.3)
ax5.set_title('T2 - T1 (pre-derotation)')
ax5.set_xlabel('x (pix)'); ax5.set_ylabel('y (pix)')
plt.colorbar(ax5_img, ax=ax5, fraction=0.046, pad=0.04)

ax6 = fig.add_subplot(2, 3, 6)
ax6.axis('off')

# --- Controls ---
bx, bw = 0.722, 0.22
by_start = 0.48
bh = 0.032
gap = 0.042

# Target 1 controls
ax_dx1 = fig.add_axes([bx, by_start, bw, bh])
dx1_slider = Slider(ax_dx1, 'dx1 (pix)', -5, 5, valinit=0.0, valstep=0.02)
dx1_slider.on_changed(update)

ax_dy1 = fig.add_axes([bx, by_start - gap, bw, bh])
dy1_slider = Slider(ax_dy1, 'dy1 (pix)', -5, 5, valinit=0.0, valstep=0.02)
dy1_slider.on_changed(update)

ax_s1 = fig.add_axes([bx, by_start - 2*gap, bw, bh])
scale1_slider = Slider(ax_s1, 'scale1', 0.5, 1.5, valinit=1.0, valstep=0.01)
scale1_slider.on_changed(update)

# Target 2 controls
ax_dx2 = fig.add_axes([bx, by_start - 3*gap, bw, bh])
dx2_slider = Slider(ax_dx2, 'dx2 (pix)', -5, 5, valinit=0.0, valstep=0.02)
dx2_slider.on_changed(update)

ax_dy2 = fig.add_axes([bx, by_start - 4*gap, bw, bh])
dy2_slider = Slider(ax_dy2, 'dy2 (pix)', -5, 5, valinit=0.0, valstep=0.02)
dy2_slider.on_changed(update)

ax_s2 = fig.add_axes([bx, by_start - 5*gap, bw, bh])
scale2_slider = Slider(ax_s2, 'scale2', 0.5, 1.5, valinit=1.0, valstep=0.01)
scale2_slider.on_changed(update)

ax_mask = fig.add_axes([bx, by_start - 6*gap, bw, bh])
mask_width_slider = Slider(ax_mask, 'X-mask width (pix)', 0, 50, valinit=0.0, valstep=1)
mask_width_slider.on_changed(update)

def reset(val=None):
    dx1_slider.set_val(0.0)
    dy1_slider.set_val(0.0)
    scale1_slider.set_val(1.0)
    dx2_slider.set_val(0.0)
    dy2_slider.set_val(0.0)
    scale2_slider.set_val(1.0)
    mask_width_slider.set_val(0.0)

ax_reset = fig.add_axes([bx, by_start - 7*gap, bw, bh])
reset_button = Button(ax_reset, 'Reset')
reset_button.on_clicked(reset)

fig.suptitle('HST STIS ADI Post-Processing Demo', fontsize=14, fontweight='bold')

update()

plt.show()
