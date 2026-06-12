import numpy as np
from astropy.io import fits
import matplotlib
from scipy.ndimage import rotate
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from matplotlib.colors import LogNorm, Normalize
from matplotlib.gridspec import GridSpec

TARGETS = {
    'HD 36112': {
        'left':   'HD__36112_2020-12-19_cube_left_frames.fits',
        'right':  'HD__36112_2020-12-19_cube_right_frames.fits',
        'parangs':'HD__36112_2020-12-19_parangs.fits',
        'Qphi_t': 'HD__36112_2020-12-19_Q_phi_star_pol_subtr.fits',
        'Uphi_t': 'HD__36112_2020-12-19_U_phi_star_pol_subtr.fits',
    },
    'HD 191089': {
        'left':   'HD_191089_2021-09-04_cube_left_frames.fits',
        'right':  'HD_191089_2021-09-04_cube_right_frames.fits',
        'parangs':'HD_191089_2021-09-04_parangs.fits',
        'Qphi_t': 'HD_191089_2021-09-04_Q_phi_star_pol_subtr.fits',
        'Uphi_t': 'HD_191089_2021-09-04_U_phi_star_pol_subtr.fits',
    },
}

EPS = 1e-12

pupil_offset = 135.99
true_north_correction = -1.75


def compute_mean_angle(angles, degree_radian='degree', axis=None):
    if degree_radian == 'degree':
        angles = np.deg2rad(angles)
    y = np.mean(np.sin(angles), axis=axis)
    x = np.mean(np.cos(angles), axis=axis)
    mean_angle = np.arctan2(y, x)
    if degree_radian == 'degree':
        mean_angle = np.rad2deg(mean_angle)
    return mean_angle


def compute_azimuthal_stokes_parameters(frame_Q, frame_U, rotation_angle=0, center_coordinates=None):
    x = np.arange(0, frame_Q.shape[-1])
    y = np.arange(0, frame_Q.shape[-2])
    xm, ym = np.meshgrid(x, y)
    if center_coordinates is None:
        x_center = 0.5 * x[-1]
        y_center = 0.5 * y[-1]
    else:
        x_center = center_coordinates[0]
        y_center = center_coordinates[1]
    phi = np.arctan2((x_center - xm), (ym - y_center)) + np.deg2rad(rotation_angle)
    frame_Q_phi = -frame_Q * np.cos(2 * phi) - frame_U * np.sin(2 * phi)
    frame_U_phi = frame_Q * np.sin(2 * phi) - frame_U * np.cos(2 * phi)
    return frame_Q_phi, frame_U_phi, np.rad2deg(phi)


def load_target(name):
    info = TARGETS[name]
    L = fits.getdata(info['left']).astype(np.float64)
    R = fits.getdata(info['right']).astype(np.float64)
    p = fits.getdata(info['parangs'])
    Qt = fits.getdata(info['Qphi_t']).astype(np.float64)
    Ut = fits.getdata(info['Uphi_t']).astype(np.float64)
    return L, R, p, Qt, Ut


def process_target(left, right, p):
    NCY = left.shape[0] // 4
    idx = np.array([1, 2, 3, 4] * NCY)
    diff = left - right
    Qdd = 0.5 * (diff[idx == 1] - diff[idx == 2])
    Udd = 0.5 * (diff[idx == 3] - diff[idx == 4])

    pQ = compute_mean_angle(np.vstack([p[idx == 1], p[idx == 2]]), axis=0)
    pU = compute_mean_angle(np.vstack([p[idx == 3], p[idx == 4]]), axis=0)
    rQ = -pQ - pupil_offset - true_north_correction
    rU = -pU - pupil_offset - true_north_correction

    Qd = np.zeros_like(Qdd)
    Ud = np.zeros_like(Udd)
    for i, (fq, aq) in enumerate(zip(Qdd, rQ)):
        Qd[i] = rotate(fq, aq, reshape=False)
    for i, (fu, au) in enumerate(zip(Udd, rU)):
        Ud[i] = rotate(fu, au, reshape=False)

    Qphi = np.array([compute_azimuthal_stokes_parameters(Qd[i], Ud[i])[0] for i in range(NCY)])
    Uphi = np.array([compute_azimuthal_stokes_parameters(Qd[i], Ud[i])[1] for i in range(NCY)])

    return Qd, Ud, Qphi, Uphi, NCY


# ── Initial load ──────────────────────────────────────────────────
cur_label = list(TARGETS.keys())[0]
left, right, p, Qtrue, Utrue = load_target(cur_label)
ALL_Q, ALL_U, ALL_Qphi, ALL_Uphi, NCY = process_target(left, right, p)
LEFT = left

# ── Figure layout (3 rows x 6 cols) ─────────────────────────────
fig = plt.figure(figsize=(16, 9))
gs = GridSpec(3, 6, width_ratios=[1.5, 1.5, 1.5, 1.5, 0.25, 1.15],
              hspace=0.35, wspace=0.30,
              left=0.04, right=0.96, top=0.95, bottom=0.05)

# Row 0: raw left-camera frames for current cycle
ax_qp = fig.add_subplot(gs[0, 0])
ax_qm = fig.add_subplot(gs[0, 1])
ax_up = fig.add_subplot(gs[0, 2])
ax_um = fig.add_subplot(gs[0, 3])
ax_cb_top = fig.add_subplot(gs[0, 4])

# Row 1: Q and U double difference (cumulative)
ax_q  = fig.add_subplot(gs[1, 0:2])
ax_u  = fig.add_subplot(gs[1, 2:4])

# Row 2: computed and true Q_phi / U_phi
ax_qphi = fig.add_subplot(gs[2, 0:2])
ax_true = fig.add_subplot(gs[2, 2:4])

# Controls (span all 3 rows)
ax_ctrl = fig.add_subplot(gs[:, 5])
ax_ctrl.axis('off')

cmap_inf = plt.cm.inferno
cmap_inf.set_bad(color='k')
cmap_bwr = plt.cm.bwr

norm_log_1k = LogNorm(vmin=1, vmax=1000)
norm_log_10k = LogNorm(vmin=1, vmax=1e4)
norm_lin_U = Normalize(vmin=-200, vmax=200)

# ── Plot handles ──────────────────────────────────────────────────
im_qp = im_qm = im_up = im_um = None
im_q = im_u = im_qphi = im_tru = None
cb_qp = cb_qm = cb_up = cb_um = None
cb_q = cb_u = cb_qphi = cb_tru = None


def _plot(ax, data, label, cmap=None, norm=None, cbar=True):
    if cmap is None:
        cmap = cmap_inf
    if norm is None:
        norm = norm_log_10k
    im = ax.imshow(data, cmap=cmap, origin='lower', norm=norm)
    if cbar:
        cb = plt.colorbar(im, ax=ax, label=label, shrink=0.75, pad=0.04)
    else:
        cb = None
    return im, cb


def _get_cycle_frames(cycle_1idx):
    i = (cycle_1idx - 1) * 4
    return LEFT[i], LEFT[i + 1], LEFT[i + 2], LEFT[i + 3]


# ── Initial plot ──────────────────────────────────────────────────
n0 = 2

qp, qm, up, um = _get_cycle_frames(n0)
im_qp, cb_qp = _plot(ax_qp, qp, '', cbar=False)
im_qm, cb_qm = _plot(ax_qm, qm, '', cbar=False)
im_up, cb_up = _plot(ax_up, up, '', cbar=False)
im_um, cb_um = _plot(ax_um, um, '', cbar=False)
cb_top = fig.colorbar(im_um, cax=ax_cb_top, label='')
cb_top.ax.yaxis.set_ticks_position('left')

ax_qp.set_title(f'Q+ left — cyc {n0}')
ax_qm.set_title(f'Q− left — cyc {n0}')
ax_up.set_title(f'U+ left — cyc {n0}')
ax_um.set_title(f'U− left — cyc {n0}')

im_q,  cb_q   = _plot(ax_q,   np.nanmedian(ALL_Q[:n0], axis=0),     'Q',      norm=norm_log_1k)
im_u,  cb_u   = _plot(ax_u,   np.nanmedian(ALL_U[:n0], axis=0),     'U',      norm=norm_log_1k)
im_qphi,cb_qphi= _plot(ax_qphi,np.nanmedian(ALL_Qphi[:n0], axis=0),  'Q_phi',  norm=norm_log_1k)
im_tru, cb_tru = _plot(ax_true, Qtrue,                                 'True Q_phi', norm=norm_log_1k)

ax_q.set_title(f'Q (double diff) — {n0}/{NCY} cycles')
ax_u.set_title(f'U (double diff) — {n0}/{NCY} cycles')
ax_qphi.set_title(f'Computed Q_phi — {n0}/{NCY} cycles')
ax_true.set_title('True Q_phi')

# ── Controls (well-spaced) ────────────────────────────────────────
ax_tgt = plt.axes([0.82, 0.85, 0.15, 0.07])
tgt = RadioButtons(ax_tgt, list(TARGETS.keys()))
for lbl in tgt.labels:
    lbl.set_fontsize(9)

ax_sld = plt.axes([0.82, 0.63, 0.15, 0.03])
sld = Slider(ax_sld, 'Cycles', 1, NCY, valinit=n0, valstep=1, valfmt='%d')
sld.label.set_fontsize(10)

ax_tog = plt.axes([0.82, 0.40, 0.15, 0.07])
tog = RadioButtons(ax_tog, ['Q_phi', 'U_phi'], active=0)
for lbl in tog.labels:
    lbl.set_fontsize(9)

# Section labels
ax_ctrl.text(0.5, 0.96, 'TARGET',  ha='center', va='top', fontsize=10, fontweight='bold',
             transform=ax_ctrl.transAxes)
ax_ctrl.text(0.5, 0.72, 'CYCLES',  ha='center', va='top', fontsize=10, fontweight='bold',
             transform=ax_ctrl.transAxes)
ax_ctrl.text(0.5, 0.53, 'DISPLAY', ha='center', va='top', fontsize=10, fontweight='bold',
             transform=ax_ctrl.transAxes)


# ── Update on slider / toggle change ──────────────────────────────
def update(val=None):
    n = int(sld.val)
    use_qphi = tog.value_selected == 'Q_phi'

    # raw data row
    qp, qm, up, um = _get_cycle_frames(n)
    im_qp.set_data(qp)
    im_qm.set_data(qm)
    im_up.set_data(up)
    im_um.set_data(um)
    ax_qp.set_title(f'Q+ left — cyc {n}')
    ax_qm.set_title(f'Q− left — cyc {n}')
    ax_up.set_title(f'U+ left — cyc {n}')
    ax_um.set_title(f'U− left — cyc {n}')

    # Q / U double diff
    d = np.nanmedian(ALL_Q[:n], axis=0)
    im_q.set_data(d)
    ax_q.set_title(f'Q (double diff) — {n}/{NCY} cycles')

    d = np.nanmedian(ALL_U[:n], axis=0)
    im_u.set_data(d)
    ax_u.set_title(f'U (double diff) — {n}/{NCY} cycles')

    # computed / true φ panels
    comp = ALL_Qphi[:n] if use_qphi else ALL_Uphi[:n]
    d = np.nanmedian(comp, axis=0)
    im_qphi.set_data(d)
    lbl = 'Q_phi' if use_qphi else 'U_phi'

    if use_qphi:
        im_qphi.set_cmap(cmap_inf)
        im_qphi.set_norm(norm_log_1k)
        im_tru.set_cmap(cmap_inf)
        im_tru.set_norm(norm_log_1k)
    else:
        im_qphi.set_cmap(cmap_bwr)
        im_qphi.set_norm(norm_lin_U)
        im_tru.set_cmap(cmap_bwr)
        im_tru.set_norm(norm_lin_U)

    ax_qphi.set_title(f'Computed {lbl} — {n}/{NCY} cycles')
    cb_qphi.set_label(lbl)

    td = Qtrue if use_qphi else Utrue
    im_tru.set_data(td)
    ax_true.set_title(f'True {lbl}')
    cb_tru.set_label(f'True {lbl}')

    fig.canvas.draw_idle()


sld.on_changed(update)
tog.on_clicked(update)


# ── Target change (full replot) ───────────────────────────────────
def change_target(label):
    global ALL_Q, ALL_U, ALL_Qphi, ALL_Uphi, NCY, Qtrue, Utrue, LEFT
    global im_q, cb_q, im_u, cb_u, im_qphi, cb_qphi, im_tru, cb_tru
    global im_qp, cb_qp, im_qm, cb_qm, im_up, cb_up, im_um, cb_um

    left, right, p, Qtrue, Utrue = load_target(label)
    ALL_Q, ALL_U, ALL_Qphi, ALL_Uphi, NCY = process_target(left, right, p)
    LEFT = left

    n = min(int(sld.val), NCY)

    for cb in [cb_q, cb_u, cb_qphi, cb_tru, cb_qp, cb_qm, cb_up, cb_um]:
        if cb:
            cb.remove()
    for ax in [ax_q, ax_u, ax_qphi, ax_true, ax_qp, ax_qm, ax_up, ax_um]:
        ax.clear()

    use_qphi = tog.value_selected == 'Q_phi'
    lbl = 'Q_phi' if use_qphi else 'U_phi'
    cmap_bot = cmap_inf if use_qphi else cmap_bwr
    norm_bot = norm_log_1k if use_qphi else norm_lin_U

    qp, qm, up, um = _get_cycle_frames(n)
    im_qp, cb_qp = _plot(ax_qp, qp, '', cbar=False)
    im_qm, cb_qm = _plot(ax_qm, qm, '', cbar=False)
    im_up, cb_up = _plot(ax_up, up, '', cbar=False)
    im_um, cb_um = _plot(ax_um, um, '', cbar=False)
    ax_qp.set_title(f'Q+ left — cyc {n}')
    ax_qm.set_title(f'Q− left — cyc {n}')
    ax_up.set_title(f'U+ left — cyc {n}')
    ax_um.set_title(f'U− left — cyc {n}')

    im_q,  cb_q   = _plot(ax_q,   np.nanmedian(ALL_Q[:n], axis=0),     'Q',      norm=norm_log_1k)
    im_u,  cb_u   = _plot(ax_u,   np.nanmedian(ALL_U[:n], axis=0),     'U',      norm=norm_log_1k)
    comp = ALL_Qphi[:n] if use_qphi else ALL_Uphi[:n]
    im_qphi, cb_qphi = _plot(ax_qphi, np.nanmedian(comp, axis=0),        lbl,
                              cmap=cmap_bot, norm=norm_bot)
    td = Qtrue if use_qphi else Utrue
    im_tru,  cb_tru  = _plot(ax_true, td,                                f'True {lbl}',
                              cmap=cmap_bot, norm=norm_bot)

    ax_q.set_title(f'Q (double diff) — {n}/{NCY} cycles')
    ax_u.set_title(f'U (double diff) — {n}/{NCY} cycles')
    ax_qphi.set_title(f'Computed {lbl} — {n}/{NCY} cycles')
    ax_true.set_title(f'True {lbl}')

    sld.valmax = NCY
    sld.ax.set_xlim(1, NCY)
    sld.set_val(n)

    fig.canvas.draw_idle()


tgt.on_clicked(change_target)

plt.show()
