import math
import sys
import corner
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression


class Stat:
    """Coordinate Transition"""

    def __init__(self):
        self.V0 = None
        self.CV = None
        self.DV = None
        self.CL = None
        self.DL = None
        self.CB = None
        self.DB = None

    def ValueGiven(self, hdr):
        for i in range(1, 4):
            ctype = hdr.get(f'CTYPE{i}', None)
            if ctype == 'VELOCITY':
                self.VR = hdr.get(f'CRVAL{i}', None)
                self.DV = hdr.get(f'CDELT{i}', None)
                self.CV = hdr.get(f'CRPIX{i}', None)
            elif ctype == 'GLON-CAR':
                self.CL = hdr.get(f'CRPIX{i}', None)
                self.DL = hdr.get(f'CDELT{i}', None)
            elif ctype == 'GLAT-CAR':
                self.CB = hdr.get(f'CRPIX{i}', None)
                self.DB = hdr.get(f'CDELT{i}', None)

    def C2V(self, channel):
        channel = np.array(channel)
        return ((channel - self.CV + 1) * self.DV + self.VR) / 1e3

    def C2L(self, channel):
        channel = np.array(channel)
        return (channel - self.CL) * self.DL

    def C2B(self, channel):
        channel = np.array(channel)
        return (channel - self.CB) * self.DB

    def V2C(self, velocity):
        velocity = np.array(velocity) * 1e3
        return ((velocity - self.VR) / self.DV + self.CV - 1).astype(int)

    def L2C(self, position):
        position = np.array(position)
        return ((position / self.DL) + self.CL).astype(int)

    def B2C(self, position):
        position = np.array(position)
        return ((position / self.DB) + self.CB).astype(int)

    @staticmethod
    def AxisRan(hdr, Type='LB'):
        ext = 0
        AxeL0 = (hdr['NAXIS1'] - hdr['CRPIX1']) * hdr['CDELT1']     # smaller Value
        AxeL1 = (0 - hdr['CRPIX1']) * hdr['CDELT1']     # Larger Value
        AxeB0 = ((0 - hdr['CRPIX2']) * hdr['CDELT2'])       # Smaller Value
        AxeB1 = ((hdr['NAXIS2'] - hdr['CRPIX2']) * hdr['CDELT2'])       # Large Value
        if hdr['NAXIS'] == 3:
            AxeV0 = ((0 - hdr['CRPIX3']) * hdr['CDELT3']) / 1e3
            AxeV1 = ((hdr['NAXIS3'] - hdr['CRPIX3']) * hdr['CDELT3']) / 1e3
            if Type == 'LV':
                ext = (AxeL1, AxeL0, AxeV0, AxeV1)
            if Type == 'VL':
                ext = (AxeV0, AxeV1, AxeL0, AxeL1)
            if Type == 'BV':
                ext = (AxeB0, AxeB1, AxeV0, AxeV1)
            if Type == 'VB':
                ext = (AxeV0, AxeV1, AxeB0, AxeB1)
            if Type == 'V':
                ext = (AxeV0, AxeV1)
        if Type == 'LB':
            ext = (AxeL1, AxeL0, AxeB0, AxeB1)
        if Type == 'BL':
            ext = (AxeB0, AxeB1, AxeL0, AxeL1)
        if Type == 'L':
            ext = (AxeL0, AxeL1)
        if Type == 'B':
            ext = (AxeB0, AxeB1)
        return ext

    def Spectra(self, hdr, cube):
        Spec = []
        self.ValueGiven(hdr)
        for i in range(cube.shape[0]):
            Velo = np.nansum(cube[i])
            Spec.append(Velo)
        Spec = np.divide(Spec, cube.shape[1] * cube.shape[2])
        Axe = np.arange(self.C2V(hdr[0]), self.C2V(hdr['NAXIS3'] + 1), hdr['CDELT3'] / 1e3)
        return Spec, Axe

    @staticmethod
    def Rms(Image):
        data_ = Image.copy()
        data_[data_ > 0] = np.nan
        sigmaData = np.nanstd(data_, ddof=1)
        sigmaData = sigmaData / np.sqrt(1 - 2. / np.pi)
        return sigmaData

    @staticmethod
    def Click(Region, time=60):
        coords = plt.ginput(Region + 1, timeout=time)
        coords = coords[1:]
        coords = [(int(x), int(y)) for x, y in coords]
        return coords

    @staticmethod
    def Extraction(Img, Coor):
        im = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
        mask = np.zeros_like(im)
        cv2.drawContours(mask, [np.array(Coor)], -1, (255, 255, 255), -1)
        gray_mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        Mask = np.where(gray_mask > 0, 1, np.nan)
        return Mask

    @staticmethod
    def RmsCube(data):
        data_ = data.copy()
        data_[data_ > 0] = np.nan
        sigmaData = np.nanstd(data_, axis=0, ddof=1)
        sigmaData = sigmaData / np.sqrt(1 - 2. / np.pi)
        return sigmaData

    def GoodLooking(self, Cube):
        Cube[Cube < -100] = 0
        Nan = ~(np.isfinite(Cube).any(axis=1))
        RmsData = self.RmsCube(Cube)
        print(np.nanmean(RmsData))
        mskcube = np.ndarray(Cube.shape, dtype=np.bool_)
        for i in range(Cube.shape[0]):
            mskcube[i, :, :] = Cube[i, :, :] > RmsData[:, :] * 3
        mskcube = mskcube & np.roll(mskcube, 1, axis=0) & np.roll(mskcube, 2, axis=0)
        mskcube = mskcube | np.roll(mskcube, -1, axis=0) | np.roll(mskcube, -2, axis=0)
        datacube = Cube * mskcube
        return datacube

    @staticmethod
    def Cat(ipa):
        df = pd.read_csv(ipa)
        return df


class Tools:
    """Calculating Moments"""

    def __init__(self, header, cube):
        self.hdr = header
        self.cube = cube
        self.Nan = None

    def Checking(self):
        self.cube[self.cube < -100] = 0
        self.Nan = ~(np.isfinite(self.cube).any(axis=0))

    def Moment0(self):
        self.Checking()
        Moment0 = np.nansum(self.cube, axis=0) * self.hdr['CDELT3'] / 1e3
        Moment0[self.Nan] = np.nan
        return Moment0

    def Weight(self):
        Weight = self.cube.copy()
        M0 = self.Moment0() / (self.hdr['CDELT3'] / 1e3)
        M0[M0 == 0] = np.nan
        for i in range(self.cube.shape[0]):
            Weight[i, :, :] = Weight[i, :, :] / M0[:, :]
        return Weight

    def Velocity(self):
        S = Stat()
        S.ValueGiven(self.hdr)
        V0, V1 = S.C2V([1, self.hdr['NAXIS3']])
        Velo = np.arange(V0, V1, self.hdr['CDELT3'] / 1e3)
        Mask = self.cube.copy()
        for i in range(len(Velo)):
            Mask[i, :, :] = Velo[i]
        return Mask

    def Moment1(self):
        Weight = self.Weight()
        Mask = self.Velocity()
        Mask = Mask * Weight
        EV = np.nansum(Mask, axis=0)
        EV[self.Nan] = np.nan
        return EV

    def Moment2(self, LW=False):
        Weight = self.Weight()
        EV = self.Moment1()
        Mask = self.Velocity()
        Mask = ((Mask - EV) ** 2)
        EV2_i = Mask * Weight
        if not LW:
            EV2 = np.sqrt(np.nansum(EV2_i, axis=0))
            EV2[self.Nan] = np.nan
            return EV2
        else:
            FWHM = np.sqrt(np.nansum(Mask, axis=0)) * np.sqrt(8 * np.log(2))
            FWHM[self.Nan] = np.nan
            return FWHM


class Properties:
    """
        Calculate Properties
            1. Excitation Temperature
            2. Optical Depth
            3. Column Density
            4. Mass
    """

    def __init__(self, header12, cube12, header, cube):
        self.hdr12 = header12
        self.hdr = header
        self.cube12 = cube12
        self.cube = cube

        """Constants"""
        # mean molecular weight
        self.Mu = 2.83
        # mass of the hydrogen atom
        self.Ma = 1.674e-27
        # solar mass
        self.Ms = 1.989e30
        # conversion between 1pc and 1cm
        self.Conversion = 3.085678e18
        # X factor
        self.X = 1.8e20
        # pixel resolution in arc-sec
        self.resolution = 0.5

    @staticmethod
    def MaxI(data):
        MaxIMap = np.nanmax(data, axis=0)
        MaxIMap[np.isnan(MaxIMap)] = 0
        return MaxIMap

    def Tex(self):
        if self.hdr12['LINE'] == '12CO(1-0)':
            MaxI = self.MaxI(self.cube12)
            TexMap = np.divide(5.532, np.log(1 + (5.532 / (MaxI + 0.819))))
            return TexMap
        else:
            print('Error ! Need 12CO to Calculate Excitation Temperature')
            return None

    def Tau(self):
        Tbg, Formula2 = 0, 0
        MaxIMap = self.MaxI(self.cube)
        if not self.hdr['LINE'] == '13CO(1-0)' or self.hdr['LINE'] == 'C18O(1-0)':
            print('Error ! Only 13CO and C18O are considered as Optical Thin')
        elif self.hdr['LINE'] == '13CO(1-0)':
            Tbg = 5.29
            Formula2 = np.divide(1, (np.exp(Tbg / self.Tex()) - 1)) - 0.164
        elif self.hdr['LINE'] == 'C18O(1-0)':
            Tbg = 5.27
            Formula2 = np.divide(1, (np.exp(Tbg / self.Tex()) - 1)) - 0.166
        Formula1 = np.divide(MaxIMap, Tbg)
        OpticalDepth = - np.log(1 - np.divide(Formula1, Formula2))
        return OpticalDepth

    def ColumnDensity(self, Iso=True, H2=True):
        ColumnDensity = 0
        if Iso:
            T = Tools(header=self.hdr12, cube=self.cube12)
            ColumnDensity = self.X * T.Moment0()
            return ColumnDensity
        elif self.hdr['LINE'] == '13CO(1-0)':
            T = Tools(header=self.hdr, cube=self.cube)
            expo = np.exp(-5.29 / self.Tex())
            N13CO = 2.42e14 * np.divide(1 + 0.88 / self.Tex(), 1 - expo) * np.divide(self.Tau(),
                                                                             1 - np.exp(-self.Tau())) * T.Moment0()
            if not H2:
                return N13CO
            else:
                ColumnDensity = 8.5e5 * N13CO
        elif self.hdr['LINE'] == 'C18O(1-0)':
            T = Tools(header=self.hdr, cube=self.cube)
            expo = np.exp(-5.27 / self.Tex())
            NC18O = 2.54e14 * np.divide(1 + 0.88 / self.Tex(), 1 - expo) * np.divide(self.Tau(),
                                                                             1 - np.exp(-self.Tau())) * T.Moment0()
            if not H2:
                return NC18O
            else:
                ColumnDensity = 6.2e6 * NC18O
        return ColumnDensity

    def Mass(self, MeanNH2, Pixel, Dist):
        Factor1 = self.Mu * self.Ma / self.Ms
        Factor2 = ((np.pi * Dist * self.Conversion * self.resolution) / (180 * 60)) ** 2
        mass = Factor1 * MeanNH2 * Factor2 * Pixel
        return np.round(mass, 1)

    @staticmethod
    def Radius(Pixel, Dist):
        Theta_Beam = 50 / 3600
        Theta = Theta_Beam * math.pi / 180
        Area = Pixel * 0.25 / 3600
        Omega = Area * (math.pi / 180) ** 2
        Diameter = Dist * np.sqrt(4 * Omega / math.pi - Theta ** 2)
        Radii = Diameter / 2
        return np.round(Radii, 1)     # in pc

    def Mass_Clump(self, Radii, MeanNH2):
        mass = (self.Mu * self.Ma * MeanNH2 * math.pi * (Radii * self.Conversion) ** 2) / self.Ms   # Radii in pc
        return np.round(mass, 1)


class Distance:

    def __init__(self, Gaia, ImgOn, headerOn, ImgBack, headerBack, DistLimits, PathOut, ParaMCMC, expand):
        """
        Parameters
        ----------
        Gaia: Path to Gaia star catalogue
        cube: Cloud's distance to be estimated
        header: Cloud's header
        image: Integrated map to remove background stars
        imagehdr: Image's header
        """
        plt.ioff()
        self.expand = expand
        self.Gaia = Gaia
        self.ImgOn = ImgOn
        self.hdrOn = headerOn
        self.ImgBack = ImgBack
        self.hdrBack = headerBack
        self.DistLimit = DistLimits
        self.PathOut = PathOut
        self.ParaMCMC = ParaMCMC

        self.df = None
        self.Doff, self.Aoff, self.Loff, self.Boff, self.AgDelt = None, None, None, None, None
        self.chain = None
        self.alpha = None
        self.Btop = None
        self.Bbottom = None
        self.Lleft = None
        self.Lright = None

    def Gaia_Checking(self):
        print('Checking Stars !')
        df = pd.read_csv(self.Gaia)
        lon = df.loc[:, 'l'].values
        lat = df.loc[:, 'b'].values
        Ag = df.loc[:, 'ag_gspphot'].values
        AgX = df.loc[:, 'ag_gspphot_upper'].values
        AgN = df.loc[:, 'ag_gspphot_lower'].values
        Para = df.loc[:, 'parallax'].values
        Para_err = df.loc[:, 'parallax_error'].values
        Dist = 1 / Para * 1e3
        Dist_err = (1000 / Para**2) * Para_err
        columns = {'l': lon, 'b': lat, 'A': Ag, 'Ax': AgX, 'An': AgN, 'd': Dist, 'der': Dist_err}
        for i in columns:
            if not isinstance(columns[i], np.ndarray):
                print('Error the input data (%s) is not a numpy ndarray' % i)
                sys.exit()
        print('Raw data columns are all arrays, proceeding......')
        index_0 = np.where(Para_err / Para > 0.2)
        for i in columns:
            columns[i] = np.delete(columns[i], index_0)
        index_1 = np.where((Ag <= 0) | (lon == 0) | (lat == 0) | (AgX == 0) | (AgN == 0) | (Para == 0) | (
                Para_err == 0))
        for i in columns:
            columns[i] = np.delete(columns[i], index_1)
        print('Data with 0-value has been removed, proceeding......')
        index_2 = np.where((Ag == np.inf) | (lon == np.inf) | (lat == np.inf) | (AgX == np.inf) | (AgX == np.inf)
                           | (Dist == np.inf) | (Dist_err == np.inf))
        for i in columns:
            columns[i] = np.delete(columns[i], index_2)
        print('Data with infinite-value has been removed, proceeding......')
        index_3 = np.where((Ag == np.nan) | (lon == np.nan) | (lat == np.nan) | (AgX == np.nan) | (AgN == np.nan)
                           | (Dist == np.nan) | (Dist_err == np.nan))
        for i in columns:
            columns[i] = np.delete(columns[i], index_3)
        print('Data with nan-value has been removed, proceeding......')
        index_4 = np.where(columns['d'] > self.DistLimit)
        for i in columns:
            columns[i] = np.delete(columns[i], index_4)
        self.df = columns
        del lon, lat, Ag, AgX, Para, Para_err, Dist, Dist_err

    def Range_Checking(self):
        S = Stat()
        S.ValueGiven(self.hdrBack)
        lat, lon = np.where(self.ImgOn == 1)

        self.Lleft = max(0, np.min(lon)-self.expand)
        self.Lright = min(self.hdrBack['NAXIS1'], np.max(lon)+self.expand)
        self.Btop = min(self.hdrBack['NAXIS2'], np.max(lat)+self.expand)
        self.Bbottom = max(0, np.min(lat)-self.expand)

        Lran = (S.C2L(self.Lright), S.C2L(self.Lleft))
        Bran = (S.C2B(self.Bbottom), S.C2B(self.Btop))
        print('l range is =', Lran)
        print('b range is =', Bran)
        indices_0 = np.where((self.df['l'] < Lran[0]) | (self.df['l'] > Lran[1]))
        for i in self.df:
            self.df[i] = np.delete(self.df[i], indices_0)
        indices_1 = np.where((self.df['b'] < Bran[0]) | (self.df['b'] > Bran[1]))
        for i in self.df:
            self.df[i] = np.delete(self.df[i], indices_1)
        print('Detailed selection has completed !')

    def OnCloud(self):
        S = Stat()
        S.ValueGiven(self.hdrOn)
        lat, lon = np.where(self.ImgOn == 1)
        alpha = np.zeros_like(self.ImgOn)
        alpha[lat, lon] = 1
        rms_off = S.Rms(self.ImgBack[np.where(alpha == 0)])
        index_on, index_off = [], []
        for i in range(len(self.df['b'])):
            if alpha[S.B2C(self.df['b'][i]), S.L2C(self.df['l'][i])] == 1 and \
                    (self.ImgOn[S.B2C(self.df['b'][i]), S.L2C(self.df['l'][i])] > 0):
                index_on.append(i)
            elif alpha[S.B2C(self.df['b'][i]), S.L2C(self.df['l'][i])] == 0 and \
                    (self.ImgBack[S.B2C(self.df['b'][i]), S.L2C(self.df['l'][i])] < 3 * rms_off):
                index_off.append(i)

        AG_delta = np.multiply(1 / 2, self.df['Ax'][index_off] - self.df['An'][index_off])
        cut_off = np.where(AG_delta >= 0.05)
        D_out = self.df['d'][index_off]
        D_off = D_out[cut_off]
        A_out = self.df['A'][index_off]
        A_off = A_out[cut_off]
        L_out = self.df['l'][index_off]
        L_off = L_out[cut_off]
        B_out = self.df['b'][index_off]
        B_off = B_out[cut_off]
        AG_delta = np.multiply(1 / 2, self.df['Ax'][index_off] - self.df['An'][index_off])[cut_off]
        self.Doff, self.Aoff, self.Loff, self.Boff, self.AgDelt = D_off, A_off, L_off, B_off, AG_delta
        self.alpha = alpha
        for i in self.df:
            self.df[i] = self.df[i][index_on]

    def Fitting(self):
        ir = IsotonicRegression()
        weights = np.divide(1, self.AgDelt ** 2)
        x = ir.fit_transform(self.Doff, self.Aoff, sample_weight=weights)

        plt.figure()
        plt.scatter(self.Doff, x, color='black', s=0.7)
        plt.scatter(self.df['d'], self.df['A'], color='red', s=0.7)

        baseline = ir.predict(self.df['d'])
        self.df['A'] = self.df['A'] - baseline
        plt.scatter(self.df['d'], baseline, color='orange', s=0.7)

    def EmceeParameters(self):

        # plt.scatter(self.df['d'], self.df['A'], color='green', s=0.1)
        # plt.xlim(0, self.DistLimit)
        # plt.close()
        # plt.savefig(self.PathOut + 'StarDistribution.pdf')
        # plt.savefig(self.PathOut + 'StarDistribution.png', dpi=500)

        indices = np.where(np.isnan(self.df['A']) | np.isnan(self.df['l']) | np.isnan(self.df['b']) |
                           np.isnan(self.df['Ax']) | np.isnan(self.df['An']) |
                           np.isnan(self.df['d']) | np.isnan(self.df['der']))
        # AG_delta = np.delete(AG_delta, indices)
        for i in self.df:
            self.df[i] = np.delete(self.df[i], indices)

        print('The number of on cloud stars is : ', len(self.df['A']))
        # set minimum on-cloud stars
        if len(self.df['A']) < 60:
            print('Not enough stars')
            sys.exit()

        # # for correcting error : store == True
        # zero = np.where(self.df['A'] > -3)
        # self.df['A'] = self.df['A'][zero]
        # self.df['d'] = self.df['d'][zero]
        # self.df['l'] = self.df['l'][zero]
        # self.df['b'] = self.df['b'][zero]
        # self.df['Ax'] = self.df['Ax'][zero]
        # self.df['An'] = self.df['An'][zero]
        # self.df['der'] = self.df['der'][zero]

        """计算emcee参数"""
        MuBack = int(0.2 * len(self.df['d']))
        # print(MuBack)
        MuBack = 50
        D_min = np.min(self.df['d'])
        D_max = np.max(self.df['d'])
        index_3 = np.argsort(self.df['d'])[-MuBack:]
        Ag_gt50 = self.df['A'][index_3]
        mu2 = np.mean(Ag_gt50)
        std_gt50 = np.std(Ag_gt50)
        AG_delta = np.multiply(1 / 2, self.df['Ax'] - self.df['An'])

        """开始emcee过程"""
        import emcee_sample
        emc = emcee_sample.emo()
        a = (D_min, D_max, self.ParaMCMC, mu2, 0.5, std_gt50)  # 将mu1修改成0.1
        chain = emc.mcmc_sample(self.df['d'], self.df['A'], self.df['der'], AG_delta, AG_delta ** 2,
                                len(self.df['A']), a, thin=10)
        self.chain = chain

    def Plotting(self):
        """画图"""

        plt.figure(figsize=(14, 8))
        figcor = corner.corner(self.chain, show_titles=True, title_kwargs={"fontsize": 10})

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

        axcor[0, 0].axvline(self.chain[:, 0].mean(), linestyle='--', lw=1.0, color='Black')
        axcor[1, 1].axvline(self.chain[:, 1].mean(), linestyle='--', lw=1.0, color='Black')
        axcor[2, 2].axvline(self.chain[:, 2].mean(), linestyle='--', lw=1.0, color='Black')
        axcor[3, 3].axvline(self.chain[:, 3].mean(), linestyle='--', lw=1.0, color='Black')
        axcor[4, 4].axvline(self.chain[:, 4].mean(), linestyle='--', lw=1.0, color='Black')

        # plt.savefig(self.PathOut + 'CornerMap.png', dpi=500)
        plt.savefig(self.PathOut + 'CornerMap.pdf')
        plt.close()

        fig, (ax, bx) = plt.subplots(2)
        ax.set_position([0.1, 0.5, 0.8, 0.4])
        S = Stat()
        S.ValueGiven(self.hdrBack)
        Ext = S.AxisRan(self.hdrBack, Type='LB')

        ax.imshow(self.ImgBack, extent=Ext, cmap='twilight', origin='lower')
        ax.contour(self.alpha, levels=[0.1], extent=Ext, colors='white', linewidths=0.8)
        ax.scatter(self.df['l'], self.df['b'], c='green', s=0.1)
        ax.scatter(self.Loff, self.Boff, c='blue', s=0.1)
        ax.tick_params(axis='x', direction='in')
        ax.tick_params(axis='y', direction='in')
        ax.set_title('Star distribution')
        ax.set_xlabel('Longitude (degree)')
        ax.set_ylabel('Latitude (degree)')
        ax.set_xlim(S.C2L(self.Lleft), S.C2L(self.Lright))
        ax.set_ylim(S.C2B(self.Bbottom), S.C2B(self.Btop))

        bins = np.linspace(0, self.DistLimit, 200)
        counts, bin_edges = np.histogram(self.df['d'], bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        average_y1 = []
        average_y2 = []
        for i in range(len(bin_centers)):
            y1_values = self.df['A'][(self.df['d'] >= bin_edges[i]) & (self.df['d'] < bin_edges[i + 1])]
            y2_values = self.Aoff[(self.Doff >= bin_edges[i]) & (self.Doff < bin_edges[i + 1])]
            if y1_values.size > 0:
                average_y1.append(np.mean(y1_values))
            else:
                average_y1.append(np.nan)
            if y2_values.size > 0:
                average_y2.append(np.mean(y2_values))
            else:
                average_y2.append(np.nan)
        bx.set_position([0.1, 0.1, 0.8, 0.3])
        bx.scatter(bin_centers, average_y1, color='green', s=0.7)
        # bx.scatter(bin_centers, average_y2, color='blue', s=0.5)
        bx.tick_params(axis='x', direction='in')
        bx.tick_params(axis='y', direction='in')
        bx.set_xlabel('Distance (pc)')
        bx.set_ylabel('$A_G (mag)$')
        x_lim = self.DistLimit
        x_m = np.median(self.chain[:, 0]).item()
        x_up = np.percentile(self.chain[:, 0], 84).item()
        x_lw = np.percentile(self.chain[:, 0], 16).item()
        mu_1 = self.chain[:, 1].mean()
        mu_2 = self.chain[:, 3].mean()
        bx.set_xlim(0, x_lim)
        bx.set_ylim(-1, 3.5)
        bx.vlines(x=x_m, ymin=-1, ymax=2.6, color='black', linestyles='solid', linewidths=0.7)
        bx.hlines(y=mu_1, xmin=0, xmax=x_m, color='blue', linestyles='dashed', linewidths=0.4)
        bx.hlines(y=mu_2, xmin=x_m, xmax=x_lim, color='blue', linestyles='dashed', linewidths=0.4)
        x_color = np.linspace(x_up, x_lw, 1000)
        bx.fill_between(x_color, -1, 2.6, color='cyan', alpha=0.8)
        bx.text(x_m, 2.6, '$%.2f^{+%.2f}_{-%.2f}$ pc' % (x_m, x_up - x_m, x_m - x_lw), ha='center', va='bottom')
        # plt.savefig(self.PathOut + 'Distance.png', dpi=500)
        plt.savefig(self.PathOut + 'Distance.pdf')
        plt.close()
        x_m = round(x_m, 1)
        print("the cloud's distance is %f pc\n----------" % x_m)
        return x_m