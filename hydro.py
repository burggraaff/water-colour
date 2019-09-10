import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from spectacle import raw, io, plot, flat
from spectacle.general import gaussMd, RMS

plt.style.use("dark_background")

water_path, sky_path, card_path, reference_path = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(water_path)
phone = io.read_json(root/"info.json")

colours = io.load_colour(stacks)
bias = np.load(products/"bias.npy")
dark = np.load(products/"dark.npy")
flat_field_correction = io.read_flat_field_correction(products, colours.shape)

# may be allowed to vary between images?
iso = 23
exposure_time = 1/1012

def load_img_or_stack(path):
    try:
        data = np.load(path)
    except OSError:
        data = io.load_dng_raw(path).raw_image
    data = data.astype(np.float64)
    return data

def show_triple(w, s, c, gauss=False, label="", saveto=None):
    fig, axs = plt.subplots(ncols=3, figsize=(7,3), sharex=True, sharey=True, tight_layout=True, gridspec_kw={"wspace":0, "hspace":0}, squeeze=True)
    for ax, data, title in zip(axs, [w, s, c], ["water", "sky", "grey card"]):
        if gauss:
            D = gaussMd(data, 5)
            title = title + " (smoothed)"
        else:
            D = data
        vmin, vmax = np.nanpercentile(D.ravel(), 1), np.nanpercentile(D.ravel(), 99)
        img = ax.imshow(D, vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
        colorbar_here = plot.colorbar(img)
        colorbar_here.set_label(label)
        colorbar_here.locator = plot.ticker.MaxNLocator(nbins=4)
        colorbar_here.update_ticks()
    if saveto is not None:
        plt.savefig(saveto)
    plt.show()

def show_triple_RGBG(wC, sC, cC, gauss=False, label="", saveto=None):
    fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(6,6), sharex=True, sharey=True, tight_layout=True, gridspec_kw={"wspace":0, "hspace":0})
    vmin = np.nanpercentile(np.stack([wC, sC, cC]).ravel(), 1)
    vmax = np.nanpercentile(np.stack([wC, sC, cC]).ravel(), 99)
    for ax_col, dataC, title in zip(axs.T, [wC, sC, cC], ["water", "sky", "grey card"]):
#        vmin = np.nanpercentile(dataC.ravel(), 1)
#        vmax = np.nanpercentile(dataC.ravel(), 99)
        for ax, data, c in zip(ax_col, dataC, "RGBG"):
            if gauss:
                D = gaussMd(data, 5)
                title = title + " (smoothed)"
            else:
                D = data
            img = ax.imshow(D, cmap=plot.cmaps[c+"r"], vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
        ax_col[0].set_title(title)
        #colorbar_here = plot.colorbar(img)
        #colorbar_here.set_label(label)
        #colorbar_here.locator = plot.ticker.MaxNLocator(nbins=4)
        #colorbar_here.update_ticks()
    if saveto is not None:
        plt.savefig(saveto)
    plt.show()

def show_triple_hist(w, s, c, saveto=None):
    fig, axs = plt.subplots(ncols=3, figsize=(7,3), sharex=False, tight_layout=True, gridspec_kw={"wspace":0, "hspace":0}, squeeze=True)
    for ax, data, title in zip(axs, [w, s, c], ["water", "sky", "grey card"]):
        ax.hist(data[~np.isnan(data)].ravel(), bins=250)
        ax.set_title(title)
        ax.set_yticks([])
    if saveto is not None:
        plt.savefig(saveto)
    plt.show()

def show_triple_hist_RGBG(wC, sC, cC, saveto=None):
    fig, axs = plt.subplots(ncols=3, figsize=(7,3), sharex=False, sharey=False, tight_layout=True, gridspec_kw={"wspace":0, "hspace":0}, squeeze=True)
    for ax, dataC, title in zip(axs, [wC, sC, cC], ["water", "sky", "grey card"]):
        RGB = [dataC[0].ravel(), dataC[1::2].ravel(), dataC[2].ravel()]
        for d, c in zip(RGB, "rgb"):
            ax.hist(d[~np.isnan(d)].ravel(), bins=250, color=c, edgecolor=c, density=True)
        ax.set_title(title)
        ax.set_yticks([])
    if saveto is not None:
        plt.savefig(saveto)
    plt.show()

water = load_img_or_stack(water_path)
sky = load_img_or_stack(sky_path)
card = load_img_or_stack(card_path)

max_value = 0.95 * (2**phone["camera"]["bits"] - 1)

water[water >= max_value] = np.nan
sky[sky >= max_value] = np.nan
card[card >= max_value] = np.nan

show_triple(water, sky, card, label="Uncorrected (ADU)", saveto="results/hydro/raw_data.pdf")
show_triple(water, sky, card, gauss=True, label="Uncorrected (ADU)", saveto="results/hydro/raw_data_smooth.pdf")

water_RGBG,_ = raw.pull_apart(water, colours)
sky_RGBG,_ = raw.pull_apart(sky, colours)
card_RGBG,_ = raw.pull_apart(card, colours)
show_triple_RGBG(water_RGBG, sky_RGBG, card_RGBG)

water_corr = water - bias - dark * exposure_time
sky_corr = sky - bias - dark * exposure_time
card_corr = card - bias - dark * exposure_time

show_triple(water_corr, sky_corr, card_corr, gauss=True, label="Corrected (ADU)")

water_flat = flat_field_correction * water_corr
sky_flat = flat_field_correction * sky_corr
card_flat = flat_field_correction * card_corr

show_triple(water_flat, sky_flat, card_flat, gauss=True, label="Flat-corrected (a.u.)", saveto="results/hydro/raw_data_flatfielded.pdf")

water_RGBG,_ = raw.pull_apart(water_flat, colours)
sky_RGBG,_ = raw.pull_apart(sky_flat, colours)
card_RGBG,_ = raw.pull_apart(card_flat, colours)
show_triple_RGBG(water_RGBG, sky_RGBG, card_RGBG, saveto="results/hydro/raw_data_flatfielded_RGBG.pdf")

midx, midy = np.array(water.shape)//2

cut = np.s_[midx-200:midx+200, midy-200:midy+200]
cut_water = np.s_[midx-200:midx+200, midy+300:midy+700]
water_cut = water_flat[cut_water]
sky_cut = sky_flat[cut]
card_cut = card_flat[cut]
colours_cut = colours[cut]

plt.imshow(gaussMd(water_flat, 5), vmin=np.nanpercentile(water_flat.ravel(), 1), vmax=np.nanpercentile(water_flat.ravel(), 99))
plt.colorbar()
plt.hlines([midx-200, midx+200], xmin=midy+300, xmax=midy+700, colors="white")
plt.vlines([midy+300, midy+700], ymin=midx-200, ymax=midx+200, colors="white")
plt.title("Water cutout")
plt.show()

plt.imshow(gaussMd(sky_flat, 5), vmin=np.nanpercentile(sky_flat.ravel(), 1), vmax=np.nanpercentile(sky_flat.ravel(), 99))
plt.colorbar()
plt.hlines([midx-200, midx+200], xmin=midy-200, xmax=midy+200, colors="white")
plt.vlines([midy-200, midy+200], ymin=midx-200, ymax=midx+200, colors="white")
plt.title("Sky cutout")
plt.show()

plt.imshow(gaussMd(card_flat, 5), vmin=np.nanpercentile(card_flat.ravel(), 1), vmax=np.nanpercentile(card_flat.ravel(), 99))
plt.colorbar()
plt.hlines([midx-200, midx+200], xmin=midy-200, xmax=midy+200, colors="white")
plt.vlines([midy-200, midy+200], ymin=midx-200, ymax=midx+200, colors="white")
plt.title("Card cutout")
plt.show()

show_triple(water_cut, sky_cut, card_cut, gauss=True, label="Corrected (ADU)")

show_triple_hist(water_cut, sky_cut, card_cut)

water_RGBG,_ = raw.pull_apart(water_cut, colours_cut)
sky_RGBG,_ = raw.pull_apart(sky_cut, colours_cut)
card_RGBG,_ = raw.pull_apart(card_cut, colours_cut)

show_triple_RGBG(water_RGBG, sky_RGBG, card_RGBG)
show_triple_hist_RGBG(water_RGBG, sky_RGBG, card_RGBG)

#def mean_std_RGB(RGBG):
#    RGB = [RGBG[0], RGBG[1::2], RGBG[2]]
#    means = np.array([np.mean(c) for c in RGB])
#    stds = np.array([np.std(c) for c in RGB])
#    return means, stds

#card_mean, card_std = mean_std_RGB(card_RGBG)
#sky_mean, sky_std = mean_std_RGB(sky_RGBG)

card_mean, card_std = card_RGBG.mean(axis=(1,2)), card_RGBG.std(axis=(1,2))
sky_mean, sky_std = sky_RGBG.mean(axis=(1,2)), sky_RGBG.std(axis=(1,2))
water_mean, water_std = water_RGBG.mean(axis=(1,2)), water_RGBG.std(axis=(1,2))

def R_RS(L_t, L_s, L_c, rho=0.028, R_ref=0.18):
    return (L_t - rho * L_s) / ((np.pi / R_ref) * L_c)

def R_RS_err(L_t, L_s, L_c, L_t_err, L_s_err, L_c_err, rho=0.028, R_ref=0.18):
    L_t_term = L_t_err * R_ref / (np.pi * L_c)
    L_s_term = L_s_err * R_ref * rho / (np.pi * L_c)
    L_c_term = L_c_err * R_ref * (L_t - rho * L_s) / (np.pi * L_c**2)
    total = np.sqrt(L_t_term**2 + L_s_term**2 + L_c_term**2)
    return total

# [:3] to throw away G_2
Rrs = R_RS(water_mean, sky_mean, card_mean)[:3]
Rrs_err = R_RS_err(water_mean, sky_mean, card_mean, water_std, sky_std, card_std)[:3]

#plot.show_RGBG(Rrs, colorbar_label="$R_{rs}$")
#
#Rrs_RGB = [Rrs[0], Rrs[1::2], Rrs[2]]
#Rrs_RGB_mean = np.array([np.mean(R) for R in Rrs_RGB])
#Rrs_RGB_std = np.array([np.std(R) for R in Rrs_RGB])

wavelength, response_RGBG, _ = io.read_spectral_responses(results)
argmaxes = response_RGBG.argmax(axis=1)
wavelengths = wavelength[argmaxes]
wavelengths_RGB = np.array([wavelengths[0], np.mean(wavelengths[1::2]), wavelengths[2]])
spectral_bandwidths = io.read_spectral_bandwidths(products)

ref_wvl, Es_ref, Lu_ref, Rrs_ref = np.loadtxt(reference_path, delimiter=",")
Es_interp = np.interp(wavelength, ref_wvl, Es_ref)
Lu_interp = np.interp(wavelength, ref_wvl, Lu_ref)
Es_response = Es_interp * response_RGBG
Lu_response = Lu_interp * response_RGBG
R_rs_response = np.trapz(Lu_response, wavelength) / np.trapz(Es_response, wavelength)
R_rs_response = R_rs_response[:3]

#Rrs_bins = np.arange(0, np.nanpercentile(Rrs.ravel(), 99.9), 0.0002)
#plt.hist(Rrs_RGB[0].ravel(), bins=Rrs_bins, density=True, color="red", alpha=0.8)
#plt.hist(Rrs_RGB[1].ravel(), bins=Rrs_bins, density=True, color="green", alpha=0.8)
#plt.hist(Rrs_RGB[2].ravel(), bins=Rrs_bins, density=True, color="blue", alpha=0.8)
#for R, c in zip(R_rs_response, "rgb"):
#    plt.axvline(R, c=c, ls="--", lw=2)
#plt.xlabel("$R_{rs}$ [sr$^{-1}$]")
#plt.show()

plt.figure(figsize=(3,3), tight_layout=True)
for i in range(3):
    plt.errorbar(wavelengths[i], Rrs[i], xerr=spectral_bandwidths[i]/2, yerr=Rrs_err[i], color="rgb"[i], fmt="o")
plt.xlim(400, 700)
plt.ylim(ymin=0)
plt.xlabel("Wavelength [nm]")
plt.ylabel("$R_{rs}$ [sr$^{-1}$]")
plt.savefig("results/hydro/R_rs_onlyHC.pdf")
plt.show()

plt.figure(figsize=(3,3), tight_layout=True)
for i in range(3):
    plt.errorbar(wavelengths[i], Rrs[i], xerr=spectral_bandwidths[i]/2, yerr=Rrs_err[i], color="rgb"[i], fmt="o")
    plt.errorbar(wavelengths[i], R_rs_response[i], xerr=spectral_bandwidths[i]/2, yerr=0, color="rgb"[i], fmt="^")
plt.plot(ref_wvl, Rrs_ref, c='white')
plt.xlim(400, 700)
plt.ylim(ymin=0)
plt.xlabel("Wavelength [nm]")
plt.ylabel("$R_{rs}$ [sr$^{-1}$]")
plt.savefig("results/hydro/R_rs_reference_RAW.pdf")
plt.show()

rms = RMS(R_rs_response - Rrs)
rms_rel = 100*RMS(1 - Rrs/R_rs_response)
x = np.linspace(0, 1, 5)
xticks = np.arange(0, 0.01, 0.002)
Rmax = np.max(np.stack([Rrs, R_rs_response]).max()) * 1.15
plt.figure(figsize=(3,3), tight_layout=True)
for i in range(3):
    plt.errorbar(R_rs_response[i], Rrs[i], xerr=0, yerr=Rrs_err[i], color="rgb"[i], fmt="o")
plt.plot(x, x, c='white', ls="--")
plt.xlim(0, Rmax)
plt.ylim(0, Rmax)
plt.xticks(xticks)
plt.yticks(xticks)
plt.xlabel("Reference $R_{rs}$ [sr$^{-1}$]")
plt.ylabel("iPhone $R_{rs}$ [sr$^{-1}$]")
plt.title(f"RMS diff. {rms:.4f} sr" + "$^{-1}$" + f" ({rms_rel:.0f}%)")
plt.gca().set_aspect("equal", "box")
plt.grid(True, ls="--", color="0.5")
plt.savefig("results/hydro/R_rs_correlation_RAW.pdf")
plt.show()
