import sys
import time
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import math
from spectral_cube import SpectralCube
import astropy.units as u
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as mticker
import matplotlib.lines as mlines
import matplotlib.patches as patches
from astropy.table import Table
import corner
import pandas as pd
import emcee_sample
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from tqdm import tqdm
import warnings
from sklearn.isotonic import IsotonicRegression
import ast


class Dorian:
    
    def c2v(self, header, channel):
        channel = np.array(channel)
        nc = header['NAXIS3']
        v0 = header['CRVAL3']
        c0 = header['CRPIX3']
        dv = header['CDELT3']
        velocity = (channel - c0 + 1) * dv + v0
        return velocity/1e3

    def v2c(self, header, velocity):
        velocity = np.array(velocity)
        nc = header['NAXIS3']
        v0 = header['CRVAL3']
        c0 = header['CRPIX3']
        dv = header['CDELT3'] / 1e3
        channel = (velocity - v0) / dv + c0 - 1
        return channel.astype(int)

    def p2c_l(self, hdr, position):
        position = np.array(position)
        c_r = hdr['CRPIX1']
        delt = hdr['CDELT1']
        c_n = (position / delt) + c_r
        c_n = np.ceil(c_n)
        c_n = c_n.astype(int)
        return c_n

    def c2p_l(self, hdr, channel):
        total = hdr['NAXIS1'] - 1
        c_r = hdr['CRPIX1']
        delt = hdr['CDELT1']
        position = (channel - c_r) * delt
        return position

    def c2p_b(self, hdr, channel):
        total = hdr['NAXIS2'] - 1
        c_r = hdr['CRPIX2']
        delt = hdr['CDELT2']
        position = (channel - c_r) * delt
        return position

    def p2c_b(self, hdr, position):
        position = np.array(position)
        c_r = hdr['CRPIX2']
        delt = hdr['CDELT2']
        c_n = (position / delt) + c_r
        c_n = np.ceil(c_n)
        c_n = c_n.astype(int)
        return c_n

    def CutFits(self, *CR, Dimension, inpath, outpath, Layer=True):

        """
        Instruction
            CR: the cut range given by the order [v, b, l]-6 or [b, l]-4
            Dimension: 3 or 2
            inpath: the input data path (fits)
            outpath: the output data path (fits)
            Layer: bool, True --> self adjusting range; False --> using CR to operate
        """
        time_start = time.time()
        print('starting to cut fits !')
        hdu = fits.open(inpath)
        data = hdu[0].data
        header = hdu[0].header
        data = np.squeeze(data)
        wcs = WCS(header)
        new_header = header.copy()
        new_wcs = wcs.copy()

        print(data.shape)

        if Dimension == 3:
            if Layer:
                del CR
                CR = np.zeros(6, dtype=int)
                indices = np.where(~np.isnan(data[:, :, :]))
                CR[0] = max(1, indices[0][0] - 5)
                CR[1] = min(header['NAXIS3'] - 1, indices[0][-1] + 5)
                CR[2] = max(1, np.min(indices[1]) - 20)
                CR[3] = min(header['NAXIS2'] - 1, np.max(indices[1]) + 20)
                CR[4] = max(1, np.min(indices[2]) - 20)
                CR[5] = min(header['NAXIS1'] - 1, np.max(indices[2]) + 20)
                del indices
            sub_data = data[CR[0]-1:CR[1], CR[2]-1:CR[3], CR[4]-1:CR[5]]
            print([CR[0], CR[1], CR[2], CR[3], CR[4], CR[5]])

            new_header['NAXIS1'] = sub_data.shape[2]
            new_header['NAXIS2'] = sub_data.shape[1]
            new_header['NAXIS3'] = sub_data.shape[0]

            new_wcs.wcs.crpix[0] -= CR[4]-1
            new_wcs.wcs.crpix[1] -= CR[2]-1
            new_wcs.wcs.crpix[2] -= CR[0]-1
        elif Dimension == 2:
            if Layer:
                del CR
                CR = np.zeros(4, dtype=int)
                indices = np.where(~np.isnan(data[:, :]))
                CR[0] = max(1, np.min(indices[0]) - 20)
                CR[1] = min(header['NAXIS1'] - 1, np.max(indices[0]) + 20)
                CR[2] = max(1, np.min(indices[1]) - 20)
                CR[3] = min(header['NAXIS2'] - 1, np.max(indices[1]) + 20)
                del indices
            sub_data = data[CR[0]-1:CR[1], CR[2]-1:CR[3]]
            print(CR[0], CR[1], CR[2], CR[3])

            new_header['NAXIS1'] = sub_data.shape[1]
            new_header['NAXIS2'] = sub_data.shape[0]

            new_wcs.wcs.crpix[0] -= CR[2]-1
            new_wcs.wcs.crpix[1] -= CR[0]-1
        else:
            print('Error dimension not match !')
            sys.exit()

        new_header.update(new_wcs.to_header())
        hdu = fits.PrimaryHDU(sub_data, header=new_header)
        hdu.writeto(outpath, overwrite=True)
        print('Cut fits file has been established')
        time_end = time.time()
        print('Takes %f seconds to finish\n----------' % (time_end-time_start))

    def CalculateTex(self, inpath,  outpath, FITS=True):

        """
        Instruction
            inpath: the path of input fits (cube file)
            outpath: the path of output fits (image file)
            FITS: set default to be True-->create fits, False-->only return peak value
        return: the peak excitation temperature of this area
        """
        time_start = time.time()
        print('starting to calculate excitation temperature')
        hdu = fits.open(inpath)[0]
        data = hdu.data
        hdr = hdu.header
        MaxIntensity = np.nanmax(data, axis=0)
        MaxIntensity[np.isnan(MaxIntensity)] = 0
        TexData = np.divide(5.532, np.log(1 + (5.532 / (MaxIntensity + 0.819))))
        Tmean = np.nanmean(TexData)
        Tpeak = np.nanmax(TexData)
        if FITS:
            hduN = fits.PrimaryHDU(TexData)
            hduN.header = hdr
            hduN.header['BUNIT'] = 'K'
            hduN.writeto(outpath, overwrite=True)
            print('Excitation temperature fits file has been established')
        time_end = time.time()
        print('Done in %f seconds\n----------' % (time_end - time_start))
        return Tmean, Tpeak

    def CalculateMoment(self, inpath, Order, Range, outpath, ty):

        """
        Explanation
            inpath: the path of the fits file
            Order: the order of the moment
            Range: the range of the integrated velocity
            outpath: the output path
            ty: Ture-->for data with a lot of NaN values, False-->for normal calculation
        """
        print('Starting to calculate moment !')
        hdu = fits.open(inpath)[0]
        # hdu.data = hdu.data.squeeze()
        hdr = hdu.header
        delta = hdu.header['CDELT3']
        if ty:
            del Range
            for k in list(hdr.keys()):
                if k[0] == 'C' and (k[-1] == '4' or k[-1] == '3'):
                    if k in hdr:
                        del hdr[k]
            hdr['BUNIT'] = 'K km/s'
            data = hdu.data
            delta_scaled = delta / 1e3
            dataOut = np.nansum(data, axis=0) * delta_scaled
            dataOut[dataOut == 0] = np.nan
            average = np.nanmean(dataOut)
            hduOut = fits.PrimaryHDU(data=dataOut, header=hdr)
            hduOut.writeto(outpath, overwrite=True)
            return average
        else:
            cube = SpectralCube.read(hdu)
            sub_cube_slab = cube.spectral_slab(Range[0] * u.km / u.s, Range[1] * u.km / u.s)
            if Order == 0:
                moment_0 = sub_cube_slab.with_spectral_unit(u.km / u.s).moment(order=Order, axis=0)
                moment_0.write(outpath, overwrite=True)
            elif Order == 1:
                moment_1 = sub_cube_slab.with_spectral_unit(u.km / u.s).moment(order=Order, axis=0)
                moment_1.write(outpath, overwrite=True)
            elif Order == 2:
                moment_2 = sub_cube_slab.with_spectral_unit(u.km / u.s).moment(order=Order, axis=0)
                moment_2.write(outpath, overwrite=True)
            else:
                print('Error: invalid order !')
        print('Moment fits has been established !\n----------')

    def Caculate12CD(self, inpath, outpath, Fits=True):

        """
        Instruction
            inpath: the input data path (moment.fits)
            Fits: the hdus of the fits file
        return: the total value of the column density of this area
        """
        print('starting to calculate column density !')
        hdu = fits.open(inpath)[0]
        data = hdu.data
        X = 2.0
        output = np.multiply(data, X)
        CDmax = np.nanmax(output)
        CDmean = np.nanmean(output)
        if Fits:
            hduN = fits.PrimaryHDU(output)
            hduN.header = hdu.header
            hduN.header['BUNIT'] = '$10^{20} cm^{-2}$'
            hduN.writeto(outpath + 'ColumnDensity.fits', overwrite=True)
            print('Column density fits file has been established')
        print('the calculation of Column density is done !\n----------')
        return CDmean, CDmax

    def MomentPlot(self, inpath, Direction, outpath, contour):  
        # 画出fits文件的图像
        """
            path : the path of input fits file
            dire : integrated direction with parameters as follows
                0 --> L-B map
                1 --> L-V map
                2 --> B-V map
            out_pa : the path of output pdf figure
            contour
                True --> with contour
                False --> without contour
        """

        time_start = time.time()
        print('Starting to plot figures !')
        image_file = fits.open(inpath)[0]
        image_data = image_file.data
        image_hdr = image_file.header
        p_start = (image_hdr['NAXIS1'] - image_hdr['CRPIX1']) * image_hdr['CDELT1']
        p_end = (0 - image_hdr['CRPIX1']) * image_hdr['CDELT1']
        v_start = ((image_hdr['NAXIS2'] - image_hdr['CRPIX2']) * image_hdr['CDELT2'])
        v_end = ((0 - image_hdr['CRPIX2']) * image_hdr['CDELT2'])
        p_r = (p_start, p_end)
        v_r = (v_start, v_end)
        fig, bx = plt.subplots()
        if Direction == 0:
            # bx.set_title('Integrated Intensity')
            bx.set_xlabel("Longitude")
            bx.set_ylabel("Latitude")
            img = bx.imshow(image_data, extent=[p_r[1], p_r[0], v_r[1], v_r[0]], cmap='hot', origin='lower', vmin=0, vmax=1)
            # bx.annotate('R1', xy=(155, 5.5), xytext=(155, 6))
            # bx.annotate('R2', xy=(152.5, 5.5), xytext=(152.5, 6))
            # bx.annotate('R3', xy=(153.3, 3.5), xytext=(153.7, 3.5))
            # rect = patches.Rectangle((153.4, 4.2), 3,  1.7, angle=0, fill=False)
            # rect_1 = patches.Rectangle((148, 1.6), 5.2, 5, angle=0, fill=False)
            # circle = patches.Circle((153.3, 3.6), 1.3, fill=False)
            # bx.add_patch(rect)
            # bx.add_patch(rect_1)
            # bx.add_patch(circle)
            bx.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d°'))
            bx.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d°'))
            divider = make_axes_locatable(bx)
            cax1 = divider.append_axes("right", size="5%", pad=0.05)
            cax1 = fig.colorbar(img, cax=cax1)
            cax1.set_label('K km/s')
        elif Direction == 1:
            img = bx.imshow(image_data, extent=[p_r[1], p_r[0], v_r[1] / 1e3, v_r[0] / 1e3], cmap='ocean', origin='lower', vmin=0, vmax=3, aspect='auto')
            fig.set_size_inches(10, 2)    # 第一个参数是图的宽度， 第二个参数是图的高度
            bx.set_title('Longitude-Velocity Map')
            bx.set_xlabel("Longitude")
            bx.set_ylabel("Velocity (km/s)")
            bx.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d°'))
            divider = make_axes_locatable(bx)
            cax1 = divider.append_axes("right", size="5%", pad=0)
            cax1 = fig.colorbar(img, cax=cax1)
            cax1.set_label('K Arcdeg')
        elif Direction == 2:
            img = bx.imshow(image_data, extent=[p_r[1], p_r[0], v_r[1] / 1e3, v_r[0] / 1e3], cmap='hot', origin='lower', vmin=0, vmax=15, aspect='auto')
            # 在imshow中可以添加vmin和vmax来调节色彩条的范围
            # fig.set_size_inches(10, 5) # 第一个参数是图的宽度， 第二个参数是图的高度
            bx.set_title('Latitude-Velocity Figure')
            bx.set_xlabel("Latitude")
            bx.set_ylabel("Velocity (km/s)")
            bx.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d°'))
            divider = make_axes_locatable(bx)
            cax1 = divider.append_axes("right", size="5%", pad=0)
            cax1 = fig.colorbar(img, cax=cax1)
            cax1.set_label('K Arcdeg')
        bx.tick_params(axis='x', direction='in', bottom=True, top=True)
        bx.tick_params(axis='y', direction='in', left=True, right=True)
        if contour:
            data = image_data[1:20, 1:20]
            rms = np.sqrt(np.mean(data ** 2))
            print(rms)
            ls = np.linspace(2.5 * rms, 3 * rms, 2)
            # image_file_1 = fits.open('/Users/administrator/PMO/Zephyrus/Zephyr_1/U/Moment/Moment_V_0.fits')[0]
            # image_data_1 = image_file_1.data
            # data_1 = image_data_1[5:50, 5:50]
            # rms_1 = np.sqrt(np.mean(data_1 ** 2))
            # ls_1 = np.linspace(10 * rms_1, 20 * rms_1, 2)
            if Direction == 0:
                x = np.arange(p_r[1], p_r[0])
                y = np.arange(v_r[1], v_r[0])
                np.meshgrid(x, y)
                bx.contour(image_data, levels=ls, extent=[p_r[1], p_r[0], v_r[1], v_r[0]], colors='white', alpha=0.7, linewidths=0.5)
                # plt.contour(image_data_1, levels=ls_1, extent=[p_r[1], p_r[0], v_r[1], v_r[0]], colors='green', alpha=0.5, linewidths=0.1)
            else:
                x = np.arange(p_r[0], p_r[1])
                y = np.arange(v_r[0], v_r[1])
                np.meshgrid(x, y)
                # 第一个contour
                bx.contour(data, levels=ls, extent=[p_r[0], p_r[1], v_r[0], v_r[1]], colors='red', alpha=1, linewidths=0.6)
        fig.savefig(outpath, dpi=500)
        fig.show()
        time_end = time.time()
        print('finished in %f seconds\n----------' % (time_end - time_start))

    def process_sub_cube(self, sub_cube, l_offset, b_offset, image_hdr, columns):
        alpha, beta = np.where(sub_cube == 1)
        theta = np.array(list(zip(alpha, beta)))
        alpha = np.unique(alpha)

        index_alpha = []
        for k in alpha:
            gamma = theta[theta[:, 0] == k, 1] + l_offset

            max_pl = np.max(gamma) + 100
            min_pl = np.min(gamma) - 100

            max_l = self.c2p_l(image_hdr, min_pl)
            min_l = self.c2p_l(image_hdr, max_pl)

            max_b = self.c2p_b(image_hdr, k + b_offset) + image_hdr['CDELT2']
            min_b = self.c2p_b(image_hdr, k + b_offset)

            index_alpha += list(np.where(
                (columns['b'] > min_b) & (columns['b'] < max_b) & (columns['l'] < max_l) & (columns['l'] > min_l)))

        index_beta = []
        for sublist in index_alpha:
            index_beta.extend(sublist)
        index_beta = list(set(index_beta))

        return index_beta

    def process_columns(self, columns, index_beta, image_data, image_hdr):
        ag = columns['A'][index_beta]
        delta_ag = np.multiply(1 / 2, columns['Ax'][index_beta] - columns['An'][index_beta])
        dist = columns['d'][index_beta]
        lon = columns['l'][index_beta]
        lat = columns['b'][index_beta]
        der = columns['der']
        index = np.argsort(dist)
        arr = (np.array(lon)[index], np.array(lat)[index], np.array(dist)[index], np.array(ag)[index],
               np.array(delta_ag)[index], np.array(der)[index])
        mask = np.zeros_like(arr[0], dtype=int)
        mat = np.zeros_like(arr[0], dtype=int)
        for i in range(len(arr[0])):
            if not image_data[self.p2c_b(image_hdr, arr[1][i]), self.p2c_l(image_hdr, arr[0][i])] > 0:
                mask[i] = 1
            else:
                mat[i] = 1

        D_out = [d for i, d in enumerate(arr[2]) if mask[i]]
        A_out = [a for i, a in enumerate(arr[3]) if mask[i]]
        A_delta = [a for i, a in enumerate(arr[4]) if mask[i]]
        L_out = [d for i, d in enumerate(arr[0]) if mask[i]]
        B_out = [a for i, a in enumerate(arr[1]) if mask[i]]

        A_delta = np.array(A_delta)
        cuto_ff = np.where(A_delta <= 0.05)
        D_out = np.delete(D_out, cuto_ff)
        A_out = np.delete(A_out, cuto_ff)
        L_out = np.delete(L_out, cuto_ff)
        B_out = np.delete(B_out, cuto_ff)
        A_delta = np.delete(A_delta, cuto_ff)
        A_delta = np.array(A_delta)

        D_on = [d for i, d in enumerate(arr[2]) if mat[i]]
        A_on = [a for i, a in enumerate(arr[3]) if mat[i]]
        delta_ag_on = [k for i, k in enumerate(arr[4]) if mat[i]]
        L_on = [a for i, a in enumerate(arr[0]) if mat[i]]
        B_on = [a for i, a in enumerate(arr[1]) if mat[i]]
        der_on = [a for i, a in enumerate(arr[5]) if mat[i]]

        if not A_out.shape[0] > 0:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        ir = IsotonicRegression()
        weights = np.divide(1, A_delta ** 2)
        ir.fit_transform(D_out, A_out, sample_weight=weights)

        A_on = np.array(A_on)
        if not A_on.shape[0] > 0:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        baseline = ir.predict(D_on)
        D_on = np.array(D_on)
        delta_ag_on = np.array(delta_ag_on)
        der_on = np.array(der_on)
        A_on = A_on - baseline
        return A_on, D_on, delta_ag_on, L_on, B_on, A_out, D_out, L_out, B_out, der_on

    def Distance(self, inpathS, inpathM, outpathc, outpathd):

        """
        Instruction
            inpathS: the path of gaia star catalog (csv)
            inpathM: the path of input moment fits (fits)
            outpathof: the path of the distribution of binned on & off cloud stars
            outpaths: the path of stars in moment figure
            outpathc: the path of corner map
            outpathd: the path of distance figure
        return: distance of the sample molecular cloud
        """

        """筛选数据"""

        print('starting to calculate distance !')
        df = pd.read_csv(inpathS)
        lon = df.loc[:, 'l'].values
        lat = df.loc[:, 'b'].values
        Ag = df.loc[:, 'ag_gspphot'].values
        Ag_max = df.loc[:, 'ag_gspphot_upper'].values
        Ag_min = df.loc[:, 'ag_gspphot_lower'].values
        parallax = df.loc[:, 'parallax'].values
        d_i = 1 / parallax * 1e3
        parallax_error = df.loc[:, 'parallax_error'].values
        d_i_err = 1 / parallax_error
        columns = {'l': lon, 'b': lat, 'A': Ag, 'Ax': Ag_max, 'An': Ag_min, 'd': d_i, 'der': d_i_err}
        for i in columns:
            if not isinstance(columns[i], np.ndarray):
                print('Error the input data (%s) is not a numpy ndarray' % i)
                sys.exit()
        print('Raw data columns are all arrays, proceeding......')
        index_0 = np.where((Ag == 0) | (lon == 0) | (lat == 0) | (Ag_max == 0) | (Ag_min == 0) | (parallax == 0) | (
                parallax_error == 0))
        for i in columns:
            columns[i] = np.delete(columns[i], index_0)
        print('Data with 0-value has been removed, proceeding......')
        index_1 = np.where(
            (Ag == np.inf) | (lon == np.inf) | (lat == np.inf) | (Ag_max == np.inf) | (Ag_min == np.inf) | (
                    d_i == np.inf) | (d_i_err == np.inf))
        for i in columns:
            columns[i] = np.delete(columns[i], index_1)
        print('Data with infinite-value has been removed, proceeding......')
        index_2 = np.where(
            (Ag == np.nan) | (lon == np.nan) | (lat == np.nan) | (Ag_max == np.nan) | (Ag_min == np.nan) | (
                    d_i == np.nan) | (d_i_err == np.nan))
        for i in columns:
            columns[i] = np.delete(columns[i], index_2)
        print('Data with nan-value has been removed, proceeding......')
        image_file = fits.open(inpathM)[0]
        image_hdr = image_file.header
        image_data = image_file.data
        l_range = (self.c2p_l(image_hdr, image_hdr['NAXIS1'] - 1), self.c2p_l(image_hdr, 0))
        b_range = (self.c2p_b(image_hdr, 0), self.c2p_b(image_hdr, image_hdr['NAXIS2'] - 1))
        print('l range =', np.round(l_range, 2))
        print('b range =', np.round(b_range, 2))
        indice_1 = np.where((columns['l'] < l_range[0]) | (columns['l'] > l_range[1]))
        for i in columns:
            columns[i] = np.delete(columns[i], indice_1)
        indice_2 = np.where((columns['b'] < b_range[0]) | (columns['b'] > b_range[1]))
        for i in columns:
            columns[i] = np.delete(columns[i], indice_2)
        indices_1 = np.where(columns['d'] > 2000)
        for i in columns:
            columns[i] = np.delete(columns[i], indices_1)
        print('Detailed selection has completed !')
        del lon, lat, Ag, Ag_max, parallax, parallax_error, d_i, d_i_err

        lantern = np.where(~np.isnan(image_data))
        lran = (np.min(lantern[1]), np.max(lantern[1]))
        lcenter = int(np.mean(lantern[1]))
        bran = (np.min(lantern[0]), np.max(lantern[0]))
        bcenter = int(np.mean(lantern[0]))

        frame = np.zeros_like(image_data)
        frame[lantern] = 1
        sub_cube_1 = frame[bcenter:bran[1], lran[0]:lcenter]
        sub_cube_2 = frame[bcenter:bran[1], lcenter:lran[1]]
        sub_cube_3 = frame[bran[0]:bcenter, lran[0]:lcenter]
        sub_cube_4 = frame[bran[0]:bcenter, lcenter:lran[1]]

        index_beta_1 = self.process_sub_cube(sub_cube_1, lran[0], bcenter, image_hdr, columns)
        index_beta_2 = self.process_sub_cube(sub_cube_2, lcenter, bcenter, image_hdr, columns)
        index_beta_3 = self.process_sub_cube(sub_cube_3, lran[0], bran[0], image_hdr, columns)
        index_beta_4 = self.process_sub_cube(sub_cube_4, lcenter, bran[0], image_hdr, columns)

        P1 = self.process_columns(columns, index_beta_1, image_data, image_hdr)
        P2 = self.process_columns(columns, index_beta_2, image_data, image_hdr)
        P3 = self.process_columns(columns, index_beta_3, image_data, image_hdr)
        P4 = self.process_columns(columns, index_beta_4, image_data, image_hdr)

        A_on = np.hstack([P1[0], P2[0], P3[0], P4[0]])
        D_on = np.hstack([P1[1], P2[1], P3[1], P4[1]])
        delt_Ag = np.hstack([P1[2], P2[2], P3[2], P4[2]])
        L_on = np.hstack([P1[3], P2[3], P3[3], P4[3]])
        B_on = np.hstack([P1[4], P2[4], P3[4], P4[4]])
        A_out = np.hstack([P1[5], P2[5], P3[5], P4[5]])
        D_out = np.hstack([P1[6], P2[6], P3[6], P4[6]])
        L_out = np.hstack([P1[7], P2[7], P3[7], P4[7]])
        B_out = np.hstack([P1[8], P2[8], P3[8], P4[8]])
        der_on = np.hstack([P1[9], P2[9], P3[9], P4[9]])

        non = np.where(~np.isnan(A_on))
        A_on = A_on[non]
        D_on = D_on[non]
        L_on = L_on[non]
        B_on = B_on[non]
        der_on = der_on[non]
        delt_Ag = delt_Ag[non]

        print('The number of on cloud stars is : ', len(A_on))
        if not len(A_on) > 0:
            return 0

        """计算emcee参数"""

        D_min = np.min(D_on)
        D_max = np.max(D_on)
        index_3 = np.argsort(D_on)[-50:]
        Ag_gt50 = A_on[index_3]
        mu2 = np.mean(Ag_gt50)
        std_gt50 = np.std(Ag_gt50)

        """开始emcee过程"""

        emc = emcee_sample.emo()
        a = (D_min + 200, D_max, 0.5, mu2, 0.5, std_gt50)
        chain = emc.mcmc_sample(D_on, A_on, der_on, delt_Ag, delt_Ag ** 2, len(A_on), a, thin=10)

        """画图"""

        plt.figure(figsize=(14, 8))
        figcor = corner.corner(chain, show_titles=True, title_kwargs={"fontsize": 10})

        axcor = np.array(figcor.axes).reshape([5, 5])

        axtil0 = 'D = ' + axcor[0, 0].title.get_text() + ' pc'
        axtil1 = '$\mu_1$ = ' + axcor[1, 1].title.get_text() + ' mag'
        axtil2 = '$\sigma_1$ = ' + axcor[2, 2].title.get_text() + ' mag'
        axtil3 = '$\mu_2$ = ' + axcor[3, 3].title.get_text() + ' mag'
        axtil4 = '$\sigma_2$ = ' + axcor[4, 4].title.get_text() + ' mag'

        axcor[0, 0].set_title(axtil0, fontsize=10)
        axcor[1, 1].set_title(axtil1, fontsize=10)
        axcor[2, 2].set_title(axtil2, fontsize=10)
        axcor[3, 3].set_title(axtil3, fontsize=10)
        axcor[4, 4].set_title(axtil4, fontsize=10)

        axcor[0, 0].axvline(chain[:, 0].mean(), linestyle='--', lw=1.0, color='Black')
        axcor[1, 1].axvline(chain[:, 1].mean(), linestyle='--', lw=1.0, color='Black')
        axcor[2, 2].axvline(chain[:, 2].mean(), linestyle='--', lw=1.0, color='Black')
        axcor[3, 3].axvline(chain[:, 3].mean(), linestyle='--', lw=1.0, color='Black')
        axcor[4, 4].axvline(chain[:, 4].mean(), linestyle='--', lw=1.0, color='Black')
        plt.savefig(outpathc, dpi=500)

        fig, (ax, bx) = plt.subplots(2)
        ax.set_position([0.1, 0.5, 0.8, 0.4])
        ax.imshow(image_data, extent=[l_range[1], l_range[0], b_range[0], b_range[1]], cmap='twilight',
                  origin='lower')
        ax.scatter(L_on, B_on, color='green', s=0.1)
        ax.scatter(L_out, B_out, color='blue', s=0.1)
        ax.tick_params(axis='x', direction='in')
        ax.tick_params(axis='y', direction='in')
        ax.set_title('Star distribution')
        ax.set_xlabel('Longitude (degree)')
        ax.set_ylabel('Latitude (degree)')

        bins = np.linspace(0, 2500, 100)
        counts, bin_edges = np.histogram(columns['d'], bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        average_y1 = []
        average_y2 = []
        for i in range(len(bin_centers)):
            y1_values = A_on[(D_on >= bin_edges[i]) & (D_on < bin_edges[i + 1])]
            y2_values = A_out[(D_out >= bin_edges[i]) & (D_out < bin_edges[i + 1])]
            if y1_values.size > 0:
                average_y1.append(np.mean(y1_values))
            else:
                average_y1.append(np.nan)
            if y2_values.size > 0:
                average_y2.append(np.mean(y2_values))
            else:
                average_y2.append(np.nan)
        bx.set_position([0.1, 0.1, 0.8, 0.3])
        bx.scatter(bin_centers, average_y1, color='green', s=0.5)
        bx.scatter(bin_centers, average_y2, color='blue', s=0.5)
        bx.tick_params(axis='x', direction='in')
        bx.tick_params(axis='y', direction='in')
        bx.set_xlabel('Distance (pc)')
        bx.set_ylabel('$A_G (mag)$')
        x_lim = 2500
        x_m = np.median(chain[:, 0]).item()
        x_up = np.percentile(chain[:, 0], 84).item()
        x_lw = np.percentile(chain[:, 0], 16).item()
        mu_1 = chain[:, 1].mean().item()
        mu_2 = chain[:, 3].mean().item()
        bx.set_xlim(0, x_lim)
        bx.set_ylim(-1, 2.5)
        bx.vlines(x=x_m, ymin=-0.5, ymax=2, color='black', linestyles='solid', linewidths=0.7)
        bx.hlines(y=mu_1, xmin=0, xmax=x_m, color='black', linestyles='dashed', linewidths=0.4)
        bx.hlines(y=mu_2, xmin=x_m, xmax=x_lim, color='black', linestyles='dashed', linewidths=0.4)
        x_color = np.linspace(x_up, x_lw, 1000)
        bx.fill_between(x_color, -0.5, 2, color='cyan', alpha=0.8)
        bx.text(x_m, 2, '$%.2f^{+%.2f}_{-%.2f}$ pc' % (x_m, x_up - x_m, x_m - x_lw), ha='center', va='bottom')
        plt.savefig(outpathd, dpi=500)
        x_m = round(x_m, 1)
        print("the cloud's distance is %f pc\n----------" % x_m)
        return x_m

    def CoordinateFix(self, path, path_ori, outpath):
        hdu_1 = fits.open(path)[0]
        hdu_1_data = hdu_1.data
        # a = np.shape(hdu_1_data)
        # print(a)
        hdu_2 = fits.open(path_ori)[0]
        hdr = hdu_1.header
        hdr_ori = hdu_2.header
        new_wcs = WCS(hdr_ori)
        hdr.update(new_wcs.to_header())
        # -----------Two-dimensional----------------
        # hdu = fits.ImageHDU(data=hdu_1_data, header=hdr)
        # primary_hdu = fits.PrimaryHDU(header=hdr)
        # hdul = fits.HDUList([primary_hdu, hdu])
        # hdul.writeto('Moment_0.fits', overwrite=True)
        # ----------Three-dimensional---------------
        hduf = fits.PrimaryHDU(hdu_1_data)
        hduf.header = hdr
        hduf.writeto(outpath, overwrite=True)
        print('finished')

    def Mass(self, distance, pixels, cd):
        """
        分子云尺度 D
            1. A 是立体角大小
            2. theta_MB 是主波束宽度
            3. Max 是峰值谱线的积分强度
        """
        conversion_distance = 3.085678e18  # 1pc到1cm的换算
        d = distance
        mu = 2.83
        m_H = 1.674e-27
        theta_MB = 50/3600
        area = pixels * 0.25    # each pixel is 0.25 acrmin^2
        Omega = area * (math.pi / 180) ** 2
        Diameter = distance * np.sqrt(4 * Omega / math.pi - theta_MB ** 2)
        Radii = round((Diameter / 2) / conversion_distance, 1)
        Diameter = Diameter * conversion_distance
        M = mu * m_H * math.pi * (Diameter / 2)**2 * cd * 10e20
        M_sun = 1.989e30
        M_mc = M / M_sun
        if not math.isnan(M_mc):
            M_mc = math.ceil(M_mc)
        print("the mass of the cloud is %i times Solar Mass\n----------" % M_mc)
        return M_mc, Radii

    def ReadFit(self, path):
        """
        It is created for the moderation of DBSCAN output data
        """
        t = Table.read(path)
        table_info = t.info()
        print(table_info)
        t.write('~/catalogue.csv', format='csv', overwrite=True)

    def MaskMatch(self, datapath, maskpath, layer, outpath):
        print('starting to match layer %i ' % layer)
        hdu_o = fits.open(datapath)[0]
        hdr_o = hdu_o.header
        data_o = hdu_o.data

        hdu_m = fits.open(maskpath)[0]
        data_m = hdu_m.data
        mask = np.where(data_m != layer, np.nan, data_o)
        hdu = fits.PrimaryHDU(mask, header=hdr_o)
        hdu.writeto(outpath, overwrite=True)
        print('match complete !\n----------')

    def Spectra(self, path, x, y, output):

        """
        This function is designed to calculate the average spectra of a given area
        Use the pixel coordinates first, with x as the longitude axis and y as the latitude axis
        """
        # load fits file
        cube = SpectralCube.read(path)
        # create an empty array to storage the summed spectra
        # load fits file
        cube = SpectralCube.read(path)
        # convert spectral axis to km/s
        cube = cube.with_spectral_unit(u.km / u.s)
        # extract a spectral slap by indexing
        subcube = cube.spectral_slab(-100 * u.km / u.s, +100 * u.km / u.s)
        subcube = subcube[:, y, x]
        plt.plot(subcube.spectral_axis, subcube)
        plt.xlabel('Velocity (km/s)')
        plt.ylabel('Temperature (K)')
        plt.ylim(-2, 8)
        plt.title('(%s,%s) pixel spectrum' % (x, y))
        plt.savefig(output)
        plt.show()
        # find indices where velocity is between signal range
        mask = np.where((subcube.spectral_axis.value > -20) & (subcube.spectral_axis.value < -0))
        print(subcube.shape)
        # calculating rms value
        sub_cube = np.delete(subcube, mask)
        print(sub_cube.shape)
        squares = np.square(sub_cube)
        mean_of_squares = np.mean(squares)
        rms = np.sqrt(mean_of_squares)
        print(rms)
        return rms

    def StackBump(self, pathClsfits, pathThi, pathThiRms, Layers, outtable):
        """Introduction
            pathCls : the twelve CO clustered fits path
            pathThi : the raw Thirteen CO fits data
            pathThiRms : the thirteen CO rms fits file
            Layer : those Layers with thirteen CO emission
        """
        # 打开第一个 FITS 文件并获取数据
        hduClsF = fits.open(pathClsfits)[0]
        dataClsF = hduClsF.data

        # 打开第二个 FITS 文件并获取数据
        hduThi = fits.open(pathThi)[0]
        dataThi = hduThi.data
        dataThi = np.squeeze(dataThi)

        # 打开第三个 FITS 文件并获取数据
        hduThirms = fits.open(pathThiRms)[0]
        dataRms = hduThirms.data

        v, b, l = np.where(dataClsF == Layers)
        P = list(zip(b, l))

        v12L, v12U = self.c2v(hduClsF.header, min(v)), self.c2v(hduClsF.header, max(v))
        c13L, c13U = int(np.ceil(self.v2c(hduThi.header, v12L))), int(np.ceil(self.v2c(hduThi.header, v12U)))
        dataThi = dataThi[c13L:c13U, :, :]

        indices = []
        for p in P:
            for i in range(dataThi.shape[0] - 2):
                if all(dataThi[i:i + 3, p[0], p[1]] > 2 * dataRms[p[0], p[1]]):
                    indices.append(p)
        indices = list(set(indices))

        summed_spectrum = np.zeros_like(np.arange(c13L, c13U + 1), dtype=float)
        df = pd.DataFrame(columns=['Pixel_L', 'Pixel_B', 'Pixel_C0', 'Pixel_C1'])
        if len(indices) > 2:
            for ind in indices:
                summed_spectrum += sum(dataThi[:, ind[0], ind[1]])
                data = pd.Series({
                'Pixel_L': ind[1],
                'Pixel_B': ind[0],
                'Pixel_C0': c13L,
                'Pixel_C1': c13U
                                    })
                df = df._append(data, ignore_index=True)
            df.to_csv(outtable + '%i_Pixel.csv' % Layers, index=True)
            summed_rms = sum(dataRms[ind[0], ind[1]] for ind in indices)
            average = summed_spectrum / len(indices)
            average_rms = summed_rms / len(indices)
            rms = average_rms / math.sqrt(len(indices))
            for i in range(len(average) - 2):
                if all(average[i:i + 3] > 3 * rms):
                    print('\n%i满足条件' % Layers)
                    return Layers
            else:
                os.remove(outtable + '%i_Pixel.csv' % Layers)
            print('\n不满足3sigma条件')
            return np.nan
        else:
            print('\n像素数太少')
            return np.nan

    def AverageSpectra(self, ipt, opt):
        """
        This function is designed to calculate the average spectra of a given area
        """
        cube = SpectralCube.read(ipt)
        velocity_axis = cube.spectral_axis
        velocity_axis = velocity_axis.to(u.km / u.s)
        spectra = []
        for i in tqdm(range(cube.shape[0])):
            v = np.nansum(cube[i])
            spectra.append(v)
        spectra = np.divide(spectra, (cube.shape[1] * cube.shape[2]))

        # cube1 = SpectralCube.read('./Output/Program1_12.fits')
        # velocity_axis1 = cube1.spectral_axis
        # velocity_axis1 = velocity_axis1.to(u.km / u.s)
        # spectra1 = []
        # for i in tqdm(range(cube1.shape[0])):
        #     v_1 = np.nansum(cube1[i])
        #     spectra1.append(v_1)
        # spectra1 = np.divide(spectra1, (cube1.shape[1] * cube1.shape[2]))

        plt.figure()
        plt.plot(velocity_axis, spectra, color='orange')
        plt.step(velocity_axis, spectra, color='orange')

        # plt.plot(velocity_axis1, spectra1, color='green')
        # plt.step(velocity_axis1, spectra1, color='green')
        # orange_line = mlines.Line2D([], [], color='orange', linewidth=2.0, label='5I$_{^{13}CO}$')
        # green_line = mlines.Line2D([], [], color='green', linewidth=2.0, label='I$_{^{12}CO}$')
        # plt.legend(handles=[orange_line, green_line])

        plt.xlabel('Radial Velocity (km/s)')
        plt.ylabel('Intensity (K)')
        plt.grid()
        plt.savefig(opt, dpi=500)
        plt.show()

    def ChannelMap(self, ipa, opa, interval, cmax, autoLayout=False):

        """
        Introduction
        :param ipa: input cube file path
        :param opa: output data png & pdf path
        :param interval: integrated channel number
        :param cmax: color bar maximum value
        :return: None
        """

        hdr = fits.open(ipa)[0].header
        cdelt = hdr['cdelt3'] / 1e3
        cglon = hdr['NAXIS1']
        cglat = hdr['NAXIS2']
        cvelo = hdr['NAXIS3']

        warnings.filterwarnings("ignore")
        sys.stdout = open(os.devnull, 'w')
        if isinstance(cvelo / interval, int) or cvelo % interval == 1:
            num = math.floor(cvelo / interval)
            for i in tqdm(range(num)):
                ran = (1 + interval * i, interval * (i + 1), 1, cglat, 1, cglon)
                self.CutFits(*ran, Dimension=3, inpath=ipa, outpath='./%ith.fits' % i, Layer=False)
                hdr_i = fits.open('./%ith.fits' % i)[0].header
                if i == 0:
                    vrange = (self.c2v(hdr_i, 1 + interval * i), self.c2v(hdr_i, interval * (i + 1)))
                else:
                    vrange = (self.c2v(hdr_i, interval * i), self.c2v(hdr_i, interval * (i + 1)))
                self.CalculateMoment('./%ith.fits' % i, 0, vrange, './%ith_moment.fits' % i, ty=False)
                os.remove('./%ith.fits' % i)
        else:
            num = math.floor((cvelo / interval))
            num = num + 1
            for i in tqdm(range(num)):
                if i == num - 1:
                    ran = (1 + interval * i, cvelo, 1, cglat, 1, cglon)
                else:
                    ran = (1 + interval * i, interval * (i + 1), 1, cglat, 1, cglon)
                self.CutFits(*ran, Dimension=3, inpath=ipa, outpath='./%ith.fits' % i, Layer=False)
                hdr_i = fits.open('./%ith.fits' % i)[0].header
                if i == 0:
                    vrange = (self.c2v(hdr_i, 1 + interval * i), self.c2v(hdr_i, interval * (i + 1)))
                elif i == num - 1:
                    vrange = (self.c2v(hdr_i, interval * i), self.c2v(hdr_i, cvelo))
                else:
                    vrange = (self.c2v(hdr_i, interval * i), self.c2v(hdr_i, interval * (i + 1)))
                self.CalculateMoment('./%ith.fits' % i, 0, vrange, './%ith_moment.fits' % i, ty=False)
                os.remove('./%ith.fits' % i)
        sys.stdout = sys.__stdout__
        print('\n%i figures to be plotted\n' % num)
        image_file = fits.open('0th_moment.fits')[0]
        image_hdr = image_file.header
        p_start = (image_hdr['NAXIS1'] - image_hdr['CRPIX1']) * image_hdr['CDELT1']
        p_end = (0 - image_hdr['CRPIX1']) * image_hdr['CDELT1']
        v_start = ((image_hdr['NAXIS2'] - image_hdr['CRPIX2']) * image_hdr['CDELT2'])
        v_end = ((0 - image_hdr['CRPIX2']) * image_hdr['CDELT2'])
        p_r = (p_start, p_end)
        v_r = (v_start, v_end)
        ratio = (v_r[0] - v_r[1]) / (p_r[1] - p_r[0])
        row = math.ceil(math.sqrt(num))

        if autoLayout:
            if isinstance(row, int):
                fig = plt.figure(figsize=(10, 10 * ratio))
                gs = gridspec.GridSpec(row, row)
                index = [[i, j] for i in range(row) for j in range(row)]
            else:
                ratio = ratio * ((row + 1) / (row - 1))
                fig = plt.figure(figsize=(10, 10 * ratio))
                gs = gridspec.GridSpec(row + 1, row - 1)
                index = [[i, j] for i in range(row + 1) for j in range(row - 1)]
        else:
            ratio = ratio * ((row + 1) / (row - 1))
            fig = plt.figure(figsize=(10, 10 * ratio))
            gs = gridspec.GridSpec(row + 1, row - 1)
            index = [[i, j] for i in range(row + 1) for j in range(row - 1)]

        plt.subplots_adjust(wspace=0, hspace=0)
        # fig.patch.set_facecolor('gray')
        print('\nStarting to plot figures >>>>>>\n')
        for i in tqdm(range(num)):
            # ax = plt.subplot(gs[index[i][0], index[i][-1]])
            ax = fig.add_subplot(gs[index[i][0], index[i][-1]])
            data = fits.open('./%ith_moment.fits' % i)[0].data
            img = ax.imshow(data, extent=(p_r[1], p_r[0], v_r[1], v_r[0]), cmap='Purples', origin='lower', vmin=0,
                            vmax=cmax, aspect='auto')
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.tick_params(length=6, axis='x', color='black', direction='in', bottom=True, top=True)
            ax.tick_params(length=6, axis='y', color='black', direction='in', left=True, right=True)
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
            ax.tick_params(which='both', width=1)
            ax.tick_params(which='minor', length=3, color='black', axis='x', direction='in', bottom=True, top=True)
            ax.tick_params(which='minor', length=3, color='black', axis='y', direction='in', right=True, left=True)

            if i == 0:
                xcoor1 = ax.get_position()
            if i == row - 1:
                xcoor2 = ax.get_position()
            if index[i][-1] == 0:
                ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d°'))
            if autoLayout:
                if isinstance(row, int):
                    if index[i][0] == row - 1:
                        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d°'))
                else:
                    if index[i][0] == row:
                        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d°'))
            else:
                if index[i][0] == row:
                    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d°'))
            if i == 0:
                vrange = (self.c2v(hdr, 1 + interval * i) / 1e3, self.c2v(hdr, interval * (i + 1)) / 1e3)
            else:
                vrange = (self.c2v(hdr, interval * i) / 1e3, self.c2v(hdr, interval * (i + 1)) / 1e3)
            ax.annotate('[%.1f, %.1f]' % (vrange[0], vrange[-1]),
                        xy=(self.c2p_l(hdr, 100), self.c2p_b(hdr, hdr['NAXIS2'] - 171)), fontsize=6, color='black')
            os.remove('./%ith_moment.fits' % i)
        axcbar = fig.add_axes((xcoor1.x0, xcoor1.y1, (xcoor2.x1 - xcoor1.x0), 0.2 * (xcoor2.y1 - xcoor2.y0)))
        cbar = fig.colorbar(img, cax=axcbar, orientation='horizontal', ticklocation='top')
        cbar.set_label('K km/s')
        fig.supxlabel('Longitude')
        fig.supylabel('Latitude')
        fig.savefig(opa + 'ChannelMap.pdf', bbox_inches='tight')
        fig.savefig(opa + 'ChannelMap.png', dpi=500, bbox_inches='tight')
        fig.show()
        # else:
        #     print('Velocity interval is beyond the velocity resolution')

    def LTE(self, ipa, opa):
        """
        Returns
        1. optical depth
        2. column density
        -------

        """
        T_110 = 5.28864
        T_109 = 5.26852

        hdu = fits.open(ipa)[0]
        hdr = hdu.header
        data = hdu.data

        self.CalculateMoment(ipa, 0, (-15, 8), './moment.fits', ty=True)
        hdu_mom = fits.open('./moment.fits')[0]
        I = hdu_mom.data
        os.remove('./moment.fits')

        self.CalculateTex('./Output/Program1_12.fits', './tex.fits', FITS=True)
        hdu_tex = fits.open('./tex.fits')[0]
        tex = hdu_tex.data
        os.remove('./tex.fits')

        # for each individual pixel, there is a maximum intensity in here
        MaxIntensity = np.nanmax(data, axis=0)
        MaxIntensity[np.isnan(MaxIntensity)] = 0


        fml_11 = np.divide(MaxIntensity, T_110)
        fml_12 = np.divide(1, (np.exp(T_110 / tex) - 1)) - 0.167667
        tau = - np.log(1 - np.divide((fml_11, fml_12)))

        fml_21 = np.divide(MaxIntensity, T_109)
        fml_22 = np.divide(1, (np.exp(T_110 / tex) - 1)) - 0.169119
        tau_18 = - np.log(1 - np.divide((fml_21, fml_22)))

        Q = np.divide(tau, 1 - np.exp(-tau)) * np.divide(1, 1 - np.exp(-T_110/tex))

        LTE_CD = 3.0e14 * Q * I

        hduOut = fits.PrimaryHDU(data=LTE_CD, header=hdu_mom.header)
        hduOut.header['BUNIT'] = ' /$cm^{-2}$'
        hduOut.writeto(opa, overwrite=True)