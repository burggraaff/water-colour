import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from spectacle import raw, io, plot, flat
from spectacle.general import gaussMd, RMS

plt.style.use("dark_background")

water_path, sky_path, card_path, reference_path = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(water_path)
phone = io.read_json(root/"info.json")

# may be allowed to vary between images?
iso = 23
exposure_time = 1/1012

rho = 0.028

water = io.load_jpg(water_path)
sky = io.load_jpg(sky_path)
card = io.load_jpg(card_path)

def show_triple(w, s, c, label=""):
    fig, axs = plt.subplots(ncols=3, figsize=(7,3), sharex=True, sharey=True, tight_layout=True, gridspec_kw={"wspace":0, "hspace":0}, squeeze=True)
    for ax, data, title in zip(axs, [w, s, c], ["water", "sky", "grey card"]):
        ax.imshow(data)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
    plt.show()

show_triple(water, sky, card, label="JPEG D.N.")

max_value = 0.95 * 255

#water[water >= max_value] = np.nan
#sky[sky >= max_value] = np.nan
#card[card >= max_value] = np.nan

def show_triple_RGB(wC, sC, cC, label=""):
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(6,6), sharex=True, sharey=True, tight_layout=True, gridspec_kw={"wspace":0, "hspace":0})
    for ax_col, dataC, title in zip(axs.T, [wC, sC, cC], ["water", "sky", "grey card"]):
        vmin = np.nanpercentile(dataC.ravel(), 1)
        vmax = np.nanpercentile(dataC.ravel(), 99)
        for ax, data, c in zip(ax_col, np.rollaxis(dataC, 2), "RGBG"):
            img = ax.imshow(data, cmap=plot.cmaps[c+"r"], vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
        ax_col[0].set_title(title)
        #colorbar_here = plot.colorbar(img)
        #colorbar_here.set_label(label)
        #colorbar_here.locator = plot.ticker.MaxNLocator(nbins=4)
        #colorbar_here.update_ticks()
    plt.show()

show_triple_RGB(water, sky, card, label="JPEG D.N.")

midx, midy = np.array(water.shape)[:2]//2

cut = np.s_[midx-200:midx+200, midy-200:midy+200]
cut_water = np.s_[midx-200:midx+200, midy+300:midy+700]
water_cut = water[cut_water]
sky_cut = sky[cut]
card_cut = card[cut]

plt.imshow(water)
plt.hlines([midx-200, midx+200], xmin=midy+300, xmax=midy+700, colors="white")
plt.vlines([midy+300, midy+700], ymin=midx-200, ymax=midx+200, colors="white")
plt.title("Water cutout")
plt.show()

plt.imshow(sky)
plt.hlines([midx-200, midx+200], xmin=midy-200, xmax=midy+200, colors="white")
plt.vlines([midy-200, midy+200], ymin=midx-200, ymax=midx+200, colors="white")
plt.title("Sky cutout")
plt.show()

plt.imshow(card)
plt.hlines([midx-200, midx+200], xmin=midy-200, xmax=midy+200, colors="white")
plt.vlines([midy-200, midy+200], ymin=midx-200, ymax=midx+200, colors="white")
plt.title("Card cutout")
plt.show()

show_triple_RGB(water_cut, sky_cut, card_cut, label="JPEG D.N.")

def show_triple_hist(w, s, c):
    fig, axs = plt.subplots(ncols=3, figsize=(7,3), sharex=False, tight_layout=True, gridspec_kw={"wspace":0, "hspace":0}, squeeze=True)
    for ax, data, title in zip(axs, [w, s, c], ["water", "sky", "grey card"]):
        ax.hist(data[~np.isnan(data)].ravel(), bins=np.arange(0,256,1))
        ax.set_title(title)
        ax.set_yticks([])
    plt.show()

def show_triple_hist_RGB(wC, sC, cC):
    fig, axs = plt.subplots(ncols=3, figsize=(7,3), sharex=False, sharey=False, tight_layout=True, gridspec_kw={"wspace":0, "hspace":0}, squeeze=True)
    for ax, dataC, title in zip(axs, [wC, sC, cC], ["water", "sky", "grey card"]):
        for d, c in zip(np.rollaxis(dataC, 2), "rgb"):
            ax.hist(d[~np.isnan(d)].ravel(), bins=np.arange(0,256,1), color=c, edgecolor=c, density=True)
        ax.set_title(title)
        ax.set_yticks([])
    plt.show()

show_triple_hist(water_cut, sky_cut, card_cut)
show_triple_hist_RGB(water_cut, sky_cut, card_cut)

card_mean, card_std = card_cut.mean(axis=(0,1)), card_cut.std(axis=(0,1))
sky_mean, sky_std = sky_cut.mean(axis=(0,1)), sky_cut.std(axis=(0,1))
water_mean, water_std = water_cut.mean(axis=(0,1)), water_cut.std(axis=(0,1))

def R_RS(L_t, L_s, L_c, rho=0.028, R_ref=0.18):
    return (L_t - rho * L_s) / ((np.pi / R_ref) * L_c)

def R_RS_err(L_t, L_s, L_c, L_t_err, L_s_err, L_c_err, rho=0.028, R_ref=0.18):
    L_t_term = L_t_err * R_ref / (np.pi * L_c)
    L_s_term = L_s_err * R_ref * rho / (np.pi * L_c)
    L_c_term = L_c_err * R_ref * (L_t - rho * L_s) / (np.pi * L_c**2)
    total = np.sqrt(L_t_term**2 + L_s_term**2 + L_c_term**2)
    return total

Rrs = R_RS(water_mean, sky_mean, card_mean)
Rrs_err = R_RS_err(water_mean, sky_mean, card_mean, water_std, sky_std, card_std)
#Rrs_RGB_mean = Rrs_RGB.mean(axis=(0,1))
#Rrs_RGB_std = Rrs_RGB.std(axis=(0,1))

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

#Rrs_bins = np.linspace(0, np.nanpercentile(Rrs_RGB.ravel(), 99.9), 101)
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
    plt.errorbar(wavelengths[i], R_rs_response[i], xerr=spectral_bandwidths[i]/2, yerr=0, color="rgb"[i], fmt="^")
plt.plot(ref_wvl, Rrs_ref, c='white')
plt.xlim(400, 700)
plt.ylim(ymin=0)
plt.xlabel("Wavelength [nm]")
plt.ylabel("$R_{rs}$ [sr$^{-1}$]")
plt.savefig("results/hydro/R_rs_reference_JPEG.pdf")
plt.show()

rms = RMS(R_rs_response - Rrs)
rms_rel = 100*RMS(1 - Rrs/R_rs_response)
x = np.linspace(0, 1, 5)
xticks = np.arange(0, 0.03, 0.005)
Rmax = np.max(np.stack([Rrs, R_rs_response]).max()) * 1.15
plt.figure(figsize=(3,3), tight_layout=True)
for i in range(3):
    plt.errorbar(R_rs_response[i], Rrs[i], xerr=0, yerr=Rrs_err[i], color="rgb"[i], fmt="o")
plt.plot(x, x, c='white', ls="--")
plt.xlim(0, Rmax)
plt.ylim(0, Rmax)
plt.xticks(xticks, rotation=45)
plt.yticks(xticks)
plt.xlabel("Reference $R_{rs}$ [sr$^{-1}$]")
plt.ylabel("iPhone $R_{rs}$ [sr$^{-1}$]")
plt.title(f"RMS diff. {rms:.4f} sr" + "$^{-1}$" + f" ({rms_rel:.0f}%)")
plt.gca().set_aspect("equal", "box")
plt.grid(True, ls="--", color="0.5")
plt.savefig("results/hydro/R_rs_correlation_JPEG.pdf")
plt.show()