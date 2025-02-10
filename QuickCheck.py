from tkinter import filedialog
import matplotlib.lines as mlines
import cv2
import functools
from astropy.coordinates import Galactic
from astropy.io import fits
from matplotlib import ticker, patches
from pvextractor import Path, extract_pv_slice
from tqdm import tqdm
import matplotlib.ticker as mticker
from astropy import units as u
from matplotlib.widgets import RectangleSelector
from matplotlib import patches as mpatches
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
import numpy as np
import pandas as pd
import os
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import filedialog, messagebox
import logging
from mpl_toolkits.axes_grid1 import make_axes_locatable


class Transition:
    """Coordinate Transition"""

    def __init__(self, hdr):
        self.V0 = None
        self.CV = None
        self.DV = None
        self.CL = None
        self.DL = None
        self.CB = None
        self.DB = None
        self.hdr = hdr

        for i in range(1, 4):
            ctype = self.hdr.get(f'CTYPE{i}', None)
            if ctype == 'VELOCITY':
                self.VR = self.hdr.get(f'CRVAL{i}', None)
                self.DV = self.hdr.get(f'CDELT{i}', None)
                self.CV = self.hdr.get(f'CRPIX{i}', None)
            elif ctype == 'GLON-CAR':
                self.CL = self.hdr.get(f'CRPIX{i}', None)
                self.DL = self.hdr.get(f'CDELT{i}', None)
            elif ctype == 'GLAT-CAR':
                self.CB = self.hdr.get(f'CRPIX{i}', None)
                self.DB = self.hdr.get(f'CDELT{i}', None)

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


class Static:

    @staticmethod
    def Rms(Image):
        data_ = Image.copy()
        data_[data_ > 0] = np.nan
        sigmaData = np.nanstd(data_, ddof=1)
        sigmaData = sigmaData / np.sqrt(1 - 2. / np.pi)
        return sigmaData

    @staticmethod
    def RmsCube(data):
        data_ = data.copy()
        data_[data_ > 0] = np.nan
        sigmaData = np.nanstd(data_, axis=0, ddof=1)
        sigmaData = sigmaData / np.sqrt(1 - 2. / np.pi)
        return sigmaData

    @staticmethod
    def get_int_rms(rmsdata, dv, deltv):
        # get_int_rms 通过fits的rms直接通过误差传递公式来得到背景噪声。rmsdata是rms.fits求均值得到的结果。dv是数据的速度间隔，如12CO是0.159
        # deltv是数据的速度范围，如NGC2264是20km/s
        return rmsdata * deltv / (np.sqrt(deltv / dv))

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

    def GoodLooking(self, Cube):
        Cube[Cube < -100] = 0
        Nan = ~(np.isfinite(Cube).any(axis=1))
        RmsData = self.RmsCube(Cube)
        mskcube = np.ndarray(Cube.shape, dtype=np.bool_)
        for i in range(Cube.shape[0]):
            mskcube[i, :, :] = Cube[i, :, :] > RmsData[:, :] * 3
        mskcube = mskcube & np.roll(mskcube, 1, axis=0) & np.roll(mskcube, 2, axis=0)
        mskcube = mskcube | np.roll(mskcube, -1, axis=0) | np.roll(mskcube, -2, axis=0)
        datacube = Cube * mskcube
        return datacube

    @staticmethod
    def ChannelMap(V, row, column, drop, datacube, header, delta, opa):
        """
        Param:
            1. V: Interval of integration in real units
            2. row: number of rows
            3. column: number of columns
            4. drop: drop the first #drop values in real unit
            5. datacube: data array
            6. header: header of the data array
            7. delta: resolution of V
            8. opa: output path and name
        """
        data = datacube
        hdr = header
        cdelt = delta / 1e3

        channel = int(round(V / cdelt))
        s = Transition(hdr)
        ext = s.AxisRan(hdr, Type='LB')
        start = s.V2C(drop)
        data = data[start:, :, :]
        img = None
        fig, ax = plt.subplots(row, column)
        plt.subplots_adjust(wspace=0, hspace=0)
        for i in tqdm(range(0, row * column)):
            sum = np.nansum(data[i * channel:(i + 1) * channel, :, :], axis=0) * cdelt
            cmax = 0.7 * np.max(sum)
            R = int(i / column)
            C = i - R * column
            if not np.max(sum) == 0:
                img = ax[R, C].imshow(sum, extent=ext, cmap='Purples', origin='lower', vmin=0, vmax=cmax, aspect='auto')
                ax[R, C].xaxis.set_ticklabels([])
                ax[R, C].yaxis.set_ticklabels([])
                ax[R, C].tick_params(length=4, axis='x', color='gray', direction='in', bottom=True, top=True)
                ax[R, C].tick_params(length=4, axis='y', color='gray', direction='in', left=True, right=True)
                ax[R, C].xaxis.set_minor_locator(ticker.AutoMinorLocator())
                ax[R, C].yaxis.set_minor_locator(ticker.AutoMinorLocator())
                ax[R, C].tick_params(which='both', width=1)
                ax[R, C].tick_params(which='minor', length=2, color='gray', axis='x', direction='in', bottom=True,
                                     top=True)
                ax[R, C].tick_params(which='minor', length=2, color='gray', axis='y', direction='in', right=True,
                                     left=True)
                ax[R, C].annotate('[%.1f, %.1f]' % (s.C2V(start) + i * V, s.C2V(start) + (i + 1) * V),
                                  xy=(158.5, 6.7), fontsize=6, color='black')
            else:
                break
        [ax[j, 0].yaxis.set_major_formatter(mticker.FormatStrFormatter('%d°')) for j in range(row)]
        [ax[row - 1, k].xaxis.set_major_formatter(mticker.FormatStrFormatter('%d°')) for k in range(column)]
        xcoor1 = ax[0, 0].get_position()
        xcoor2 = ax[0, column - 1].get_position()
        axcbar = fig.add_axes((xcoor1.x0, xcoor1.y1, (xcoor2.x1 - xcoor1.x0), 0.1 * (xcoor2.y1 - xcoor2.y0)))
        cbar = fig.colorbar(img, cax=axcbar, orientation='horizontal', ticklocation='top')
        cbar.set_label('K km/s')
        fig.supxlabel('Galactic Longitude')
        fig.supylabel('Galactic Latitude')
        fig.savefig(opa, bbox_inches='tight')
        fig.savefig(opa, dpi=500, bbox_inches='tight')


# class ZoomAndSelector:
#     def __init__(self, axis: object, callback=None, Reverse=True) -> object:
#         self.x_indices = None
#         self.y_indices = None
#         self.selections = []
#         self.ax = axis
#         self.Reverse = Reverse
#         self.canvas = axis.figure.canvas
#         self.original_xlim = axis.get_xlim()
#         self.original_ylim = axis.get_ylim()
#         self.zoom_stack = []
#         self.selected_rect = None
#         self.callback = callback
#         self.selector = RectangleSelector(
#             self.ax, self.onselect,
#             useblit=False,
#             button=[1],
#             minspanx=5, minspany=5,
#             spancoords='pixels',
#             interactive=True
#         )
#         self.canvas.mpl_connect('key_press_event', self.on_key_press)
#         self.canvas.mpl_connect('button_press_event', self.on_press)
#
#     def onselect(self, eclick, erelease):
#         self.selected_rect = (eclick.xdata, eclick.ydata, erelease.xdata, erelease.ydata)
#         if self.selected_rect is not None:
#             x1, y1, x2, y2 = self.selected_rect
#             if x2 < x1:
#                 x1, x2 = x2, x1
#             if y2 < y1:
#                 y1, y2 = y2, y1
#             self.zoom_stack.append((self.ax.get_xlim(), self.ax.get_ylim()))
#             if self.selected_rect is not None:
#                 if self.Reverse:
#                     self.ax.set_xlim([x1, x2])
#                     self.ax.set_ylim([y1, y2])
#                 else:
#                     self.ax.set_xlim([x2, x1])
#                     self.ax.set_ylim([y1, y2])
#                 self.canvas.draw()
#             self.selections.append((x1, y1, x2, y2))
#
#     def on_key_press(self, event):
#         if event.key == 'enter' and self.selected_rect is not None:
#             if self.callback:
#                 self.callback(self.selected_rect)
#             return self.selected_rect
#         return None
#
#     def on_press(self, event):
#         if event.button == 3:
#             if self.zoom_stack:
#                 xlim, ylim = self.zoom_stack.pop()
#                 self.ax.set_xlim(xlim)
#                 self.ax.set_ylim(ylim)
#                 self.canvas.draw()
#             else:
#                 messagebox.showinfo("Info", "Already at the original view")

class ZoomAndSelector:
    def __init__(self, axis: object, callback=None, reverse=True) -> None:
        self.ax = axis
        self.callback = callback
        self.reverse = reverse
        self.canvas = axis.figure.canvas
        self.original_xlim = axis.get_xlim()
        self.original_ylim = axis.get_ylim()
        self.zoom_stack = []
        self.selected_rect = None

        self.selector = RectangleSelector(
            self.ax, self.on_select,
            useblit=True,
            button=[1],
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=True
        )
        self.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.canvas.mpl_connect('button_press_event', self.on_press)

    def on_select(self, eclick, erelease):
        self.selected_rect = (eclick.xdata, eclick.ydata, erelease.xdata, erelease.ydata)
        x1, y1, x2, y2 = self.selected_rect
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])

        self.zoom_stack.append((self.ax.get_xlim(), self.ax.get_ylim()))

        if self.reverse:
            self.ax.set_xlim([x1, x2])
            self.ax.set_ylim([y1, y2])
        else:
            self.ax.set_xlim([x2, x1])
            self.ax.set_ylim([y1, y2])

        self.canvas.draw()

    def on_key_press(self, event):
        if event.key == 'enter' and self.selected_rect is not None:
            if self.callback:
                self.callback(self.selected_rect)
            self.reset_selection()
        return None

    def on_press(self, event):
        if event.button == 3:
            self.reset_view()

    def reset_selection(self):
        self.selected_rect = None

    def reset_view(self):
        if self.zoom_stack:
            xlim, ylim = self.zoom_stack.pop()
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)
            self.canvas.draw()
        else:
            print("已经在原始视图了")


def execute_specific_after_0(child_func_name):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            child_func = getattr(self, child_func_name, None)
            if callable(child_func):
                child_func()
            return result
        return wrapper
    return decorator


def execute_specific_after(method_name):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            getattr(self, method_name)(*args, **kwargs)
            return result
        return wrapper
    return decorator


def update_comboboxes_decorator(*combobox_attrs):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            for combobox_attr in combobox_attrs:
                combobox = getattr(self, combobox_attr)
                if isinstance(result, pd.DataFrame):
                    column_names = result.columns.tolist()
                    combobox['values'] = column_names
            return result
        return wrapper
    return decorator


def monitor_levels_factor(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        levels_factor = kwargs.get('levels_factor', 1)
        result = func(*args, **kwargs)
        if levels_factor != wrapper.last_levels_factor:
            result = func(*args, **kwargs)
            wrapper.last_levels_factor = levels_factor
        return result
    wrapper.last_levels_factor = None
    return wrapper


class Components:

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Karl-El")
        Name = ['Select Position', 'Average Spectrum', 'Integration', 'P-V diagram']
        self.frames = {f'{i}': tk.LabelFrame(self.root, text=Name[i - 1]) for i in range(1, 5)}
        self.F_1 = self.frames['1']
        self.F_2 = self.frames['2']
        self.F_3 = self.frames['3']
        self.F_4 = self.frames['4']

        for i in range(2):
            self.root.grid_columnconfigure(i, weight=1)

        button_frame = tk.Frame(self.root)
        button_frame.grid(row=0, column=0, columnspan=2, pady=10)

        for i in range(1, 5):
            tk.Button(button_frame, text=f"Frame {i} (L)", command=lambda i=i: self.show_frame(i, 'left')).grid(
                row=(i - 1) // 2, column=(i - 1) % 2 * 2, padx=5, pady=5)
            tk.Button(button_frame, text=f"Frame {i} (R)", command=lambda i=i: self.show_frame(i, 'right')).grid(
                row=(i - 1) // 2, column=(i - 1) % 2 * 2 + 1, padx=5, pady=5)

        self.frames['1'].grid(row=1, column=0, padx=10, pady=10, sticky='nsew')
        self.frames['2'].grid(row=1, column=1, padx=10, pady=10, sticky='nsew')

    def show_frame(self, frame_id, position):
        if position == 'left':
            for frame in self.frames.values():
                if frame.grid_info().get('column') == 0:
                    frame.grid_remove()
            self.frames[str(frame_id)].grid(row=1, column=0, padx=10, pady=10, sticky='nsew')
        elif position == 'right':
            for frame in self.frames.values():
                if frame.grid_info().get('column') == 1:
                    frame.grid_remove()
            self.frames[str(frame_id)].grid(row=1, column=1, padx=10, pady=10, sticky='nsew')

    @staticmethod
    def add_checkbox(master, text, row, column, sticky='ew', command=None):
        var = tk.BooleanVar()
        checkbox = tk.Checkbutton(master, text=text, variable=var, command=command)
        checkbox.grid(row=row, column=column, sticky=sticky)
        return var

    @staticmethod
    def add_button(master, text, row, column, command, rowspan=1, columnspan=1, sticky='w'):
        button = tk.Button(master, text=text, command=command)
        button.grid(row=row, column=column, rowspan=rowspan, columnspan=columnspan, sticky=sticky)
        return button

    @staticmethod
    def create_canvas(master, row, row_span, column, column_span, sticky='nsew', width=6, height=5):
        # fig = Figure(figsize=(width, height), constrained_layout=True)
        fig = Figure(figsize=(width, height))
        fig.clear()
        ax = fig.add_subplot(111)
        canvas = FigureCanvasTkAgg(fig, master=master)
        canvas.draw()
        canvas.get_tk_widget().grid(row=row, rowspan=row_span, column=column, columnspan=column_span, sticky=sticky)
        return ax, canvas

    @staticmethod
    def create_spinbox(frame, row, column, from_, to, increment, command, sticky='ew'):
        def on_focus_out(event):
            command()
        spinbox = ttk.Spinbox(frame, from_=from_, to=to, increment=increment, command=command)
        spinbox.grid(row=row, column=column, sticky=sticky)
        spinbox.bind("<FocusOut>", on_focus_out)
        return spinbox

    @staticmethod
    def create_combobox(frame, row, column, values, sticky='ew'):
        dropdown = ttk.Combobox(frame, values=values)
        dropdown.grid(row=row, column=column, sticky=sticky)
        return dropdown


class Widget_1(Components):

    def __init__(self):
        super().__init__()
        self.pix_l, self.pix_b = None, None
        self.dat1, self.hdr1, self.hdu1 = None, None, None
        self.dat2, self.hdr2, self.hdu2 = None, None, None
        self.dat3, self.hdr3, self.hdu3 = None, None, None
        self.dat4, self.hdr4, self.hdu4 = None, None, None
        self.df1, self.df2 = None, None
        self.hdr1_var = tk.StringVar()

        self.add_button(self.F_1, 'Open Fits No.1', 0, 0, command=lambda: self.load_file_from_dialog(0, 1, 1))
        self.add_button(self.F_1, 'Open Fits No.2', 0, 2, command=lambda: self.load_file_from_dialog(0, 3, 2))
        self.add_button(self.F_1, 'Open Fits No.3', 1, 0, command=lambda: self.load_file_from_dialog(1, 1, 3))
        self.add_button(self.F_1, 'Open Fits No.4', 1, 2, command=lambda: self.load_file_from_dialog(1, 3, 4))

        self.var_1 = self.add_checkbox(self.F_1, 'File No.1', 3, 0)
        self.var_2 = self.add_checkbox(self.F_1, 'File No.2', 3, 1)
        self.var_3 = self.add_checkbox(self.F_1, 'File No.3', 3, 2)
        self.var_4 = self.add_checkbox(self.F_1, 'File No.4', 3, 3)

        self.add_button(self.F_1, 'Checking Data', 4, 3, command=lambda: self.generate_graph())

        self.Ax_1, self.Figure_1 = self.create_canvas(self.F_1, 5, 3, 0, 4)
        self.zoom1 = ZoomAndSelector(self.Ax_1, callback=lambda rect: self.reading(rect))

        self.combobox1 = self.create_combobox(self.F_3, 5, 0, [])
        self.combobox2 = self.create_combobox(self.F_3, 5, 1, [])
        self.combobox3 = self.create_combobox(self.F_3, 6, 0, [])
        self.combobox4 = self.create_combobox(self.F_3, 6, 1, [])

    def load_file_from_dialog(self, R, C, N):
        filetypes = [("FITS files", "*.fits"), ("CSV files", "*.csv")]
        filepath = filedialog.askopenfilename(filetypes=filetypes)
        if filepath:
            filename = os.path.basename(filepath)
            try:
                if N < 5:
                    self.load_file(filepath, N)
                    label_file = tk.Label(self.F_1, text=f"{filename}")
                    label_file.grid(row=R, column=C, padx=5, pady=5, sticky=tk.W)
                if N == 5:
                    df = self.load_file_1(filepath, N)
                    label_file = tk.Label(self.F_3, text=f"{filename}")
                    label_file.grid(row=R, column=C, padx=5, pady=5, sticky=tk.W)
                elif N == 6:
                    df = self.load_file_2(filepath, N)
                    label_file = tk.Label(self.F_3, text=f"{filename}")
                    label_file.grid(row=R, column=C, padx=5, pady=5, sticky=tk.W)
            except ValueError as e:
                messagebox.showerror("Error", str(e))

    def load_file(self, filepath: str, N):
        try:
            if N in [1, 2, 3, 4]:
                if not filepath.endswith('.fits'):
                    raise ValueError("Expected a FITS file.")
                data, header, hdu = fits.getdata(filepath), fits.getheader(filepath), fits.open(filepath)[0]
                setattr(self, f'dat{N}', data)
                setattr(self, f'hdr{N}', header)
                setattr(self, f'hdu{N}', hdu)
                logging.info("Data and header loaded successfully.")
                if N == 1:
                    self.hdr1_var.set(str(header))
        except Exception as e:
            logging.error(f"Failed to load data: {e}")
            raise ValueError(f"Failed to load data: {e}")

    @update_comboboxes_decorator('combobox1', 'combobox2')
    def load_file_1(self, filepath: str, N):
        try:
            if N == 5:
                if not filepath.endswith('.csv'):
                    raise ValueError("Expected a CSV file.")
                df = pd.read_csv(filepath)  # 读取完整的 CSV 文件
                self.df1 = df
                return df
        except Exception as e:
            logging.error(f"Failed to load data: {e}")
            raise ValueError(f"Failed to load data: {e}")

    @update_comboboxes_decorator('combobox3', 'combobox4')
    def load_file_2(self, filepath: str, N):
        try:
            if N == 6:
                if not filepath.endswith('.csv'):
                    raise ValueError("Expected a CSV file.")
                df = pd.read_csv(filepath)  # 读取完整的 CSV 文件
                self.df2 = df
                return df
        except Exception as e:
            logging.error(f"Failed to load data: {e}")
            raise ValueError(f"Failed to load data: {e}")

    def draw_data(self, data, header, checkbox_var):
        if checkbox_var.get():
            if data.ndim == 3:
                self.draw_plot(self.mom0(data, header))
            else:
                self.draw_plot(data)

    def generate_graph(self):
        data_checkbox_pairs = [
            (self.dat1, self.hdr1, self.var_1),
            (self.dat2, self.hdr2, self.var_2),
            (self.dat3, self.hdr3, self.var_3),
            (self.dat4, self.hdr4, self.var_4)
        ]
        for data, header, checkbox_var in data_checkbox_pairs:
            self.draw_data(data, header, checkbox_var)

    @staticmethod
    def mom0(data, hdr):
        return np.nansum(data, axis=0) * hdr['CDELT3'] / 1e3

    def draw_plot(self, img):
        self.Ax_1.imshow(img, origin='lower', cmap='jet')
        # self.zoom1 = ZoomAndSelector(self.Ax_1, callback=lambda rect: self.reading(rect))
        self.Figure_1.draw()

    @execute_specific_after_0('draw_spectra')
    def reading(self, rect):
        x1, y1, x2, y2 = rect
        x1, x2, y1, y2 = np.floor([x1, x2, y1, y2])
        x = np.arange(x1, x2, 1)
        y = np.arange(y1, y2, 1)
        x_mesh, y_mesh = np.meshgrid(x, y)
        x = x_mesh.ravel()
        y = y_mesh.ravel()
        self.pix_l, self.pix_b = x, y


class Widget_2(Widget_1):

    def __init__(self):

        super().__init__()
        self.BS_ran, self.RS_ran, self.Center = None, None, None
        self.Spec1, self.Spec2, self.Spec3 = None, None, None
        self.line_pos = {'1': None, '2': None, '3': None, '4': None, '5': None}
        self.line_refs = {1: None, 2: None, 3: None, 4: None, 5: None}
        self.fill_patches = {}
        self.expand = 0

        self.Ax_2, self.Figure_2 = self.create_canvas(self.F_2, 0, 3, 0, 4)
        self.zoom2 = ZoomAndSelector(self.Ax_2, callback=None)

        self.hdr1_var.trace_add('write', self.update_spinboxes)

        self.spin_1 = self.create_spinbox(self.F_2, 4, 0, 0, 0, 0, command=lambda: self.draw_vertical_lines('blue', 1))
        self.spin_2 = self.create_spinbox(self.F_2, 4, 1, 0, 0, 0, command=lambda: self.draw_vertical_lines('blue', 2))
        self.spin_3 = self.create_spinbox(self.F_2, 5, 0, 0, 0, 0, command=lambda: self.draw_vertical_lines('red', 3))
        self.spin_4 = self.create_spinbox(self.F_2, 5, 1, 0, 0, 0, command=lambda: self.draw_vertical_lines('red', 4))
        self.spin_5 = self.create_spinbox(self.F_2, 6, 0, 0, 0, 0, command=lambda: self.draw_vertical_lines('orange', 5))

        self.spin_1.config(state='disabled')
        self.spin_2.config(state='disabled')
        self.spin_3.config(state='disabled')
        self.spin_4.config(state='disabled')
        self.spin_5.config(state='disabled')

        self.add_button(self.F_2, 'Fill BS Region', 4, 2, command=lambda: self.fill_BorR(BoR=True))
        self.add_button(self.F_2, 'Fill RS Region', 5, 2, command=lambda: self.fill_BorR(BoR=False))

    def update_spinboxes(self, *args):
        if self.hdr1:
            self.T = Transition(self.hdr1)
            self.spin_1.config(from_=self.T.C2V(0), to=self.T.C2V(self.hdr1['NAXIS3']),
                               increment=self.hdr1['CDELT3'] / 1e3, state='normal')
            self.spin_2.config(from_=self.T.C2V(0), to=self.T.C2V(self.hdr1['NAXIS3']),
                               increment=self.hdr1['CDELT3'] / 1e3, state='normal')
            self.spin_3.config(from_=self.T.C2V(0), to=self.T.C2V(self.hdr1['NAXIS3']),
                               increment=self.hdr1['CDELT3'] / 1e3, state='normal')
            self.spin_4.config(from_=self.T.C2V(0), to=self.T.C2V(self.hdr1['NAXIS3']),
                               increment=self.hdr1['CDELT3'] / 1e3, state='normal')
            self.spin_5.config(from_=self.T.C2V(0), to=self.T.C2V(self.hdr1['NAXIS3']),
                               increment=self.hdr1['CDELT3'] / 1e3, state='normal')

    def draw_spectra(self):
        self.Ax_2.clear()
        x0, x1, y0, y1 = int(min(self.pix_l)), int(max(self.pix_l)), int(min(self.pix_b)), int(max(self.pix_b))

        if self.hdr1:
            cube1 = self.dat1[:, y0:y1, x0:x1]
            self.Spec1, Axe = self.Spectra(self.hdr1, cube1)
            self.Ax_2.step(Axe, self.Spec1, color='black')
        if self.hdr2:
            cube2 = self.dat2[:, y0:y1, x0:x1]
            self.Spec2, Axe = self.Spectra(self.hdr2, cube2)
            self.Ax_2.step(Axe, self.Spec2, color='green')
        if self.hdr3:
            cube3 = self.dat3[:, y0:y1, x0:x1]
            self.Spec3, Axe = self.Spectra(self.hdr3, cube3)
            self.Ax_2.step(Axe, self.Spec3, color='red')

        line1 = mlines.Line2D([], [], color='black', linewidth=1.0, label='$^{12}$CO')
        line2 = mlines.Line2D([], [], color='green', linewidth=1.0, label='$^{13}$CO')
        # line3 = mlines.Line2D([], [], color='red', linewidth=1.0, label='Fits_3')
        self.Ax_2.set_xlabel('Radial Velocity (km/s)')
        self.Ax_2.set_ylabel('Intensity (K)')
        self.Ax_2.legend(handles=[line1, line2])
        self.Ax_2.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        self.Ax_2.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        self.Ax_2.tick_params("both", which='major', length=5, width=1.5, colors='black', direction='in')
        self.Ax_2.tick_params(which='minor', length=3, width=1.0, labelsize=10, labelcolor='1', direction='in')
        self.Ax_2.grid(axis='both')
        # self.zoom2 = ZoomAndSelector(self.Ax_2, callback=None)
        self.Figure_2.draw()

    @staticmethod
    def Spectra(hdr, cube):
        Spec = []
        T = Transition(hdr)
        for i in range(cube.shape[0]):
            Velo = np.nansum(cube[i])
            Spec.append(Velo)
        Spec = np.divide(Spec, cube.shape[1] * cube.shape[2])
        Axe = np.arange(0, hdr['NAXIS3'], 1)
        Axe = T.C2V(Axe)
        return Spec, Axe

    def draw_vertical_lines(self, color, line_id):
        spin_boxes = {1: self.spin_1, 2: self.spin_2, 3: self.spin_3, 4: self.spin_4, 5: self.spin_5}

        x_pos = float(spin_boxes[line_id].get()) if spin_boxes[line_id].get() else None

        if x_pos is not None:
            if self.line_refs[line_id] is not None:
                self.line_refs[line_id].remove()
            self.line_refs[line_id] = self.Ax_2.axvline(x=self.increment(x_pos), color=color, linestyle='--', linewidth=1)
            self.line_pos[str(line_id)] = x_pos
            self.Figure_2.draw()

    def increment(self, value):
        incremented_value = self.T.C2V(0) + round((value - self.T.C2V(0)) / (self.hdr1['CDELT3'] / 1e3)) * (self.hdr1['CDELT3'] / 1e3)
        return incremented_value

    # @execute_specific_after('draw_map')
    def fill_BorR(self, BoR):
        try:
            if BoR:
                if None in [self.line_pos.get('1'), self.line_pos.get('2')]:
                    raise ValueError("BS range positions not set")
                self.BS_ran = [self.increment(float(self.line_pos['1'])),
                               self.increment(float(self.line_pos['2']))]
            else:
                if None in [self.line_pos.get('2'), self.line_pos.get('3')]:
                    raise ValueError("RS range positions not set")
                self.RS_ran = [self.increment(float(self.line_pos['3'])),
                               self.increment(float(self.line_pos['4']))]
        except Exception as e:
            messagebox.showerror("Error", f"Not valid range: {e}")

        try:
            self.Center = self.increment(float(self.line_pos['5']))
        except Exception as e:
            messagebox.showerror("Error", "Haven't define central velocity")

        if BoR:
            x0, x1 = self.BS_ran[0], self.BS_ran[1]
        else:
            x0, x1 = self.RS_ran[0], self.RS_ran[1]

        if x0 is not None and x1 is not None:
            if not x0 < x1:
                x0, x1 = x1, x0
            velo_range = np.arange(0, self.hdr1['NAXIS3'], 1)
            x0, x1 = self.T.V2C([x0, x1])

            if BoR:
                if 'first' in self.fill_patches:
                    self.fill_patches['first'].remove()

                x_fill_where1 = (velo_range > x0) & (velo_range <= x1+1)
                velo_range = self.T.C2V(velo_range)

                self.fill_patches['first'] = self.Ax_2.fill_between(velo_range, self.Spec1, where=x_fill_where1,
                                                                    color='royalblue', alpha=1, step='pre')
            else:
                if 'second' in self.fill_patches:
                    self.fill_patches['second'].remove()
                x_fill_where2 = (velo_range > x0) & (velo_range <= x1+1)
                velo_range = self.T.C2V(velo_range)

                self.fill_patches['second'] = self.Ax_2.fill_between(velo_range, self.Spec1, where=x_fill_where2,
                                                                     color='crimson', alpha=1, step='pre')

            self.Figure_2.draw()


class Widget_3(Widget_2):

    def __init__(self):

        super().__init__()
        self.Ax_3, self.Figure_3 = self.create_canvas(self.F_3, 0, 4, 0, 4)
        self.zoom3 = ZoomAndSelector(self.Ax_3, callback=lambda rect: self.output_fits(rect), reverse=False)
        self.Ax_4, self.Figure_4 = self.create_canvas(self.F_4, 0, 1, 0, 1, width=4, height=6)
        self.zoom4 = ZoomAndSelector(self.Ax_4, callback=lambda rect: self.checking(rect), reverse=True)
        self.fig_indi, self.ax_indi = plt.subplots(1, 1)
        self.zoom = ZoomAndSelector(self.ax_indi, callback=lambda rect: self.output_fits(rect), reverse=False)

        self.add_button(self.F_2, 'Create BS Contour', 4, 3, command=lambda: self.draw_contour(BoR=True))
        self.add_button(self.F_2, 'Create RS Contour', 5, 3, command=lambda: self.draw_contour(BoR=False))

        self.spin_exp = self.create_spinbox(self.F_2, 7, 0, 0, 200, 1, command=lambda: self.update_expand())
        self.add_button(self.F_2, 'Plot Background', 7, 1, command=lambda: self.draw_background())

        self.add_button(self.F_2, 'Save Figures', 7, 3, command=lambda: self.save_figure())
        self.down_drop = self.create_combobox(self.F_2, 7, 2, ['None', 'Figure_1', 'Figure_2', 'Figure_3', 'Figure_4'])

        self.add_button(self.F_2, 'Individual check', 6, 1, command=lambda: self.draw_background_individual())
        self.add_button(self.F_2, 'Red-shift', 6, 2, command=lambda: self.draw_contour_individual(BoR=False))
        self.add_button(self.F_2, 'Blue-shift', 6, 3, command=lambda: self.draw_contour_individual(BoR=True))

        self.N = 0

    @execute_specific_after_0('draw_background_individual')
    def update_expand(self):
        self.expand = int(self.spin_exp.get())

    def draw_map(self, BoR=True):
        self.Ax_3.clear()
        x0, x1, y0, y1 = int(min(self.pix_l)), int(max(self.pix_l)), int(min(self.pix_b)), int(max(self.pix_b))
        try:
            if BoR:
                if self.BS_ran is not None and self.BS_ran[0] is not None:
                    img_dat = self.dat1[self.T.V2C(self.BS_ran[0]):self.T.V2C(self.BS_ran[1]), y0-self.expand:y1+self.expand, x0-self.expand:x1+self.expand]
                    img = np.nansum(img_dat, axis=0) * self.hdr1['CDELT3'] / 1e3
                    self.Ax_3.imshow(img, origin='lower', extent=(self.T.C2L(x0-self.expand), self.T.C2L(x1+self.expand),
                                                              self.T.C2B(y0-self.expand), self.T.C2B(y1+self.expand)), cmap='GnBu')
                    self.Figure_3.draw()
            else:
                if self.RS_ran is not None and self.RS_ran[0] is not None:
                    img_dat = self.dat1[self.T.V2C(self.RS_ran[0]):self.T.V2C(self.RS_ran[1]), y0-self.expand:y1+self.expand, x0-self.expand:x1+self.expand]
                    img = np.nansum(img_dat, axis=0) * self.hdr1['CDELT3'] / 1e3
                    self.Ax_3.imshow(img, origin='lower', extent=(self.T.C2L(x0 - self.expand), self.T.C2L(x1 + self.expand),
                                                      self.T.C2B(y0 - self.expand), self.T.C2B(y1 + self.expand)), cmap='virial')
                    self.Figure_3.draw()
        except Exception as e:
            messagebox.showerror("Error", f"Not valid range: {e}")

    def draw_background(self):
        self.Ax_3.clear()
        if hasattr(self, 'cax1'):
            self.cax1.remove()

        x0, x1, y0, y1 = int(min(self.pix_l)), int(max(self.pix_l)), int(min(self.pix_b)), int(max(self.pix_b))
        img_dat = self.dat1[:, y0:y1, x0:x1]

        img = np.nansum(img_dat, axis=0) * self.hdr1['CDELT3'] / 1e3
        im = self.Ax_3.imshow(img, origin='lower', extent=(self.T.C2L(x0), self.T.C2L(x1),
                                                      self.T.C2B(y0), self.T.C2B(y1)), cmap='GnBu')
        # im = self.Ax_3.imshow(img, origin='lower', extent=(self.T.C2L(x0 - self.expand), self.T.C2L(x1 + self.expand),
        #                                               self.T.C2B(y0 - self.expand), self.T.C2B(y1 + self.expand)), cmap='GnBu')

        def format_func(value, tick_number):
            degree = int(value)
            minute = int((value * 60) % 60)
            return "{:02d}°{:02d}'".format(degree, minute)

        self.Ax_3.xaxis.set_major_formatter(mticker.FuncFormatter(format_func))
        self.Ax_3.yaxis.set_major_formatter(mticker.FuncFormatter(format_func))
        self.Ax_3.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        self.Ax_3.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        self.Ax_3.tick_params("both", which='major', length=5, width=1.5, colors='black', direction='in')
        self.Ax_3.tick_params(which='minor', length=3, width=1.0, labelsize=10, labelcolor='1', direction='in')
        divider = make_axes_locatable(self.Ax_3)
        self.cax1 = divider.append_axes("right", size="3%", pad=0.1)
        cbar = self.Figure_3.figure.colorbar(im, cax=self.cax1)
        cbar.set_label('$K{\cdot}km/s$')
        self.Ax_3.set_xlabel('Galactic Longtitude')
        self.Ax_3.set_ylabel('Galactic Latitude')
        self.Figure_3.draw()

    def draw_background_individual(self):
        self.ax_indi.clear()
        if hasattr(self, 'cax'):
            self.cax.remove()

        x0, x1, y0, y1 = int(min(self.pix_l)), int(max(self.pix_l)), int(min(self.pix_b)), int(max(self.pix_b))
        img_dat = self.dat1[:, y0 - self.expand:y1 + self.expand, x0 - self.expand:x1 + self.expand]

        img = np.nansum(img_dat, axis=0) * self.hdr1['CDELT3'] / 1e3
        im = self.ax_indi.imshow(img, origin='lower', extent=(self.T.C2L(x0 - self.expand), self.T.C2L(x1 + self.expand),
                                                      self.T.C2B(y0 - self.expand), self.T.C2B(y1 + self.expand)), cmap='GnBu')

        def format_func(value, tick_number):
            degree = int(value)
            minute = int((value * 60) % 60)
            return "{:02d}°{:02d}'".format(degree, minute)

        self.ax_indi.xaxis.set_major_formatter(mticker.FuncFormatter(format_func))
        self.ax_indi.yaxis.set_major_formatter(mticker.FuncFormatter(format_func))
        self.ax_indi.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        self.ax_indi.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        self.ax_indi.tick_params("both", which='major', length=5, width=1.5, colors='black', direction='in')
        self.ax_indi.tick_params(which='minor', length=3, width=1.0, labelsize=10, labelcolor='1', direction='in')
        divider = make_axes_locatable(self.ax_indi)
        self.cax = divider.append_axes("right", size="3%", pad=0.1)
        cbar = self.fig_indi.colorbar(im, cax=self.cax)
        cbar.set_label('$K{\cdot}km/s$')
        self.ax_indi.set_xlabel('Galactic Longtitude')
        self.ax_indi.set_ylabel('Galactic Latitude')
        self.fig_indi.canvas.draw()

    def draw_contour_individual(self, BoR=True):
        self.mx = 0
        x0, x1, y0, y1 = int(min(self.pix_l)), int(max(self.pix_l)), int(min(self.pix_b)), int(max(self.pix_b))

        try:
            if BoR:
                if None in [self.line_pos.get('1'), self.line_pos.get('2')]:
                    raise ValueError("BS range positions not set")
                self.BS_ran = [self.increment(float(self.line_pos['1'])),
                               self.increment(float(self.line_pos['2']))]
                self.cls = True
            else:
                if None in [self.line_pos.get('2'), self.line_pos.get('3')]:
                    raise ValueError("RS range positions not set")
                self.RS_ran = [self.increment(float(self.line_pos['3'])),
                               self.increment(float(self.line_pos['4']))]
                self.cls = False
        except Exception as e:
            messagebox.showerror("Error", f"Not valid range: {e}")

        try:
            if BoR:
                if self.BS_ran is not None and self.BS_ran[0] is not None:
                    img_dat = self.dat1[self.T.V2C(self.BS_ran[0]):self.T.V2C(self.BS_ran[1]),
                              y0 - self.expand:y1 + self.expand,
                              x0 - self.expand:x1 + self.expand]
                    color = 'blue'
            else:
                if self.RS_ran is not None and self.RS_ran[0] is not None:
                    img_dat = self.dat1[self.T.V2C(self.RS_ran[0]):self.T.V2C(self.RS_ran[1]),
                              y0 - self.expand:y1 + self.expand,
                              x0 - self.expand:x1 + self.expand]
                    color = 'red'

            if img_dat is not None:
                img = np.nansum(img_dat, axis=0) * self.hdr1['CDELT3'] / 1e3
                self.img_indi = img
                rms = Static.Rms(img)
                if np.isnan(rms):
                    mx = np.max(img)
                    ls = np.arange(0.6 * mx, 1 * mx, 0.1 * mx)
                    self.mx = mx
                    self.rms_indi = 0
                    print("Choosing MAX criteria, MAX is: ", self.mx)
                else:
                    ls = np.arange(3 * rms, 21 * rms, 3 * rms)
                    self.mx = 0
                    self.rms_indi = rms
                    print("Choosing RMS criteria, RMS is: ", self.rms_indi)

                if not hasattr(self, 'contour_lines'):
                    self.contours_indi = {'red': [], 'blue': []}

                for c in self.contours_indi[color]:
                    try:
                        for coll in c.collections:
                            coll.remove()
                    except Exception as e:
                        print(f"Error removing collection: {e}")
                self.contours_indi[color] = []
                self.ext_indi = (self.T.C2L(x0 - self.expand), self.T.C2L(x1 + self.expand),
                          self.T.C2B(y0 - self.expand), self.T.C2B(y1 + self.expand))
                contour = self.ax_indi.contour(img, levels=ls,
                                            extent=(self.T.C2L(x0 - self.expand), self.T.C2L(x1 + self.expand),
                                                    self.T.C2B(y0 - self.expand), self.T.C2B(y1 + self.expand)),
                                            colors=color)
                self.contours_indi[color].append(contour)
                self.ax_indi.set_xlim(self.T.C2L(x0 - self.expand), self.T.C2L(x1 + self.expand))
                self.ax_indi.set_ylim(self.T.C2B(y0 - self.expand), self.T.C2B(y1 + self.expand))
                self.fig_indi.canvas.draw()
            else:
                raise ValueError("图像数据为空。请检查范围和数据源。")
        except Exception as e:
            messagebox.showerror("错误", f"无效的范围: {e}")

    def output_fits(self, rect):
        Vb0, Vb1 = self.BS_ran
        Vr0, Vr1 = self.RS_ran
        print(self.RS_ran)
        Vc = self.increment(float(self.line_pos['5']))

        x10, y10, x20, y20 = rect
        x2, x1, y1, y2 = self.T.L2C(x10), self.T.L2C(x20), self.T.B2C(y10), self.T.B2C(y20)
        x = np.arange(x1, x2, 1)
        y = np.arange(y1, y2, 1)
        x_mesh, y_mesh = np.meshgrid(x, y)
        x = x_mesh.ravel()
        y = y_mesh.ravel()

        x_0, x_1, y_0, y_1 = int(min(x)), int(max(x)), int(min(y)), int(max(y))
        mask = self.dat1[self.T.V2C(Vb0):self.T.V2C(Vr1), y_0:y_1, x_0:x_1]

        MAX = self.mx
        Rms = self.rms_indi

        hdu = fits.PrimaryHDU(data=mask, header=self.hdr1)
        hdu.header['Vb0'] = Vb0
        hdu.header['Vb1'] = Vb1
        hdu.header['Vr0'] = Vr0
        hdu.header['Vr1'] = Vr1
        hdu.header['Vc'] = Vc
        hdu.header['MAX'] = MAX
        hdu.header['RMS'] = Rms
        hdu.header['N'] = self.N
        hdu.header['Lc'] = (x10 + x20) / 2
        hdu.header['Bc'] = (y20 + y10) / 2

        hdu.header['NAXIS1'] = mask.shape[2]
        hdu.header['NAXIS2'] = mask.shape[1]
        hdu.header['NAXIS3'] = mask.shape[0]

        hdu.header['CRPIX1'] -= x_0
        hdu.header['CRPIX2'] -= y_0
        hdu.header['CRPIX3'] -= self.T.V2C(Vb0)

        file_path = filedialog.asksaveasfilename(defaultextension=".fits",
                                                 filetypes=[("FITS files", "*.fits"),
                                                            ("All files", "*.*")])
        if file_path:
            try:
                hdu_list = fits.HDUList(hdu)
                hdu_list.writeto(file_path, overwrite=True)
                messagebox.showinfo("Success", "FITS file saved successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save FITS file: {e}")

    def draw_contour(self, BoR=True):
        self.mx = 0
        x0, x1, y0, y1 = int(min(self.pix_l)), int(max(self.pix_l)), int(min(self.pix_b)), int(max(self.pix_b))

        try:
            if BoR:
                if None in [self.line_pos.get('1'), self.line_pos.get('2')]:
                    raise ValueError("BS range positions not set")
                self.BS_ran = [self.increment(float(self.line_pos['1'])),
                               self.increment(float(self.line_pos['2']))]
            else:
                if None in [self.line_pos.get('2'), self.line_pos.get('3')]:
                    raise ValueError("RS range positions not set")
                self.RS_ran = [self.increment(float(self.line_pos['3'])),
                               self.increment(float(self.line_pos['4']))]
        except Exception as e:
            messagebox.showerror("Error", f"Not valid range: {e}")

        try:
            if BoR:
                if self.BS_ran is not None and self.BS_ran[0] is not None:
                    img_dat = self.dat1[self.T.V2C(self.BS_ran[0]):self.T.V2C(self.BS_ran[1]),
                              y0 - self.expand:y1 + self.expand,
                              x0 - self.expand:x1 + self.expand]
                    color = 'blue'
            else:
                if self.RS_ran is not None and self.RS_ran[0] is not None:
                    img_dat = self.dat1[self.T.V2C(self.RS_ran[0]):self.T.V2C(self.RS_ran[1]),
                              y0 - self.expand:y1 + self.expand,
                              x0 - self.expand:x1 + self.expand]
                    color = 'red'

            if img_dat is not None:
                img = np.nansum(img_dat, axis=0) * self.hdr1['CDELT3'] / 1e3
                mx = np.max(img)
                self.mx = mx
                ls = np.arange(0.4 * mx, 1 * mx, 0.15 * mx)

                if not hasattr(self, 'contour_lines'):
                    self.contour_lines = {'red': [], 'blue': []}

                for c in self.contour_lines[color]:
                    try:
                        for coll in c.collections:
                            coll.remove()
                    except Exception as e:
                        print(f"Error removing collection: {e}")
                self.contour_lines[color] = []
                contour = self.Ax_3.contour(img, levels=ls,
                                            extent=(self.T.C2L(x0 - self.expand), self.T.C2L(x1 + self.expand),
                                                    self.T.C2B(y0 - self.expand), self.T.C2B(y1 + self.expand)),
                                            colors=color)
                self.contour_lines[color].append(contour)
                self.Ax_3.set_xlim(self.T.C2L(x0 - self.expand), self.T.C2L(x1 + self.expand))
                self.Ax_3.set_ylim(self.T.C2B(y0 - self.expand), self.T.C2B(y1 + self.expand))
                # zoom3 = ZoomAndSelector(self.Ax_3, callback=lambda rect: self.output_fits(rect), reverse=False)
                self.Figure_3.draw()
            else:
                raise ValueError("图像数据为空。请检查范围和数据源。")
        except Exception as e:
            messagebox.showerror("错误", f"无效的范围: {e}")

    def save_figure(self):
        selected_fig = self.down_drop.get()
        figures = {
            'Figure_1': self.Figure_1,
            'Figure_2': self.Figure_2,
            'Figure_3': self.Figure_3,
            'Figure_4': self.Figure_4
        }

        if selected_fig in figures:
            fig = figures[selected_fig]
            file_path = tk.filedialog.asksaveasfilename(defaultextension=".pdf",
                                                        filetypes=[("PDF files", "*.pdf"),
                                                                   ("All files", "*.*")])
            if file_path:
                fig.figure.savefig(file_path)
        else:
            messagebox.showerror("Error", "Please select a valid figure to save.")


class Widget_4(Widget_3):

    def __init__(self):
        super().__init__()

        self.path_coor = None
        self.px, self.py = None, None
        self.arrow = None

        self.add_button(self.F_3, 'Open Csv No.1', 4, 0, command=lambda: self.load_file_from_dialog(4, 1, 5))
        self.add_button(self.F_3, 'Open Csv No.2', 4, 2, command=lambda: self.load_file_from_dialog(4, 3, 6))

        self.add_button(self.F_3, 'Select Csv No.1', 5, 2, command=lambda: self.confirm_para_1())
        self.add_button(self.F_3, 'Remove Csv No.1', 5, 3, command=lambda: self.remove_para_1())

        self.add_button(self.F_3, 'Select Csv No.2', 6, 2, command=lambda: self.confirm_para_2())
        self.add_button(self.F_3, 'Remove Csv No.2', 6, 3, command=lambda: self.remove_para_2())

        self.add_button(self.F_3, 'Remove Line', 7, 2, command=lambda: self.update_arrow())
        # self.add_button(self.F_3, 'Plot Line', 7, 1, command=lambda: self.plot_pv_line())
        self.add_button(self.F_3, 'Calculate PV', 7, 3, command=lambda: self.get_pos_pv())

        self.rms_factor_spinbox = ttk.Spinbox(self.F_3, from_=1, to=20, increment=1, command=lambda: self.update_contours())
        self.rms_factor_spinbox.set(3)
        self.rms_factor_spinbox.grid(row=7, column=0)

        self.contours = []
        self.images = {}

        self.rms_factor_spinbox_indi = ttk.Spinbox(self.F_3, from_=0.1, to=20, increment=0.1, command=lambda: self.update_contours_indi())
        self.rms_factor_spinbox_indi.set(3)
        self.rms_factor_spinbox_indi.grid(row=7, column=1)

        # 创建起始点和终点的输入框
        self.entry_start_x = tk.Entry(self.F_1, font=('Arial', 12))
        self.entry_start_x.grid(row=8, column=0, sticky='ew')
        self.entry_start_y = tk.Entry(self.F_1, font=('Arial', 12))
        self.entry_start_y.grid(row=8, column=1, sticky='ew')

        self.entry_end_x = tk.Entry(self.F_1, font=('Arial', 12))
        self.entry_end_x.grid(row=8, column=2, sticky='ew')
        self.entry_end_y = tk.Entry(self.F_1, font=('Arial', 12))
        self.entry_end_y.grid(row=8, column=3, sticky='ew')

        # 创建用于显示坐标的标签
        self.lbl_start_point = tk.Label(self.F_1, text="Start Point: ", font=('Arial', 12))
        self.lbl_start_point.grid(row=9, column=0, sticky='ew', pady=10)
        self.lbl_end_point = tk.Label(self.F_1, text="End Point: ", font=('Arial', 12))
        self.lbl_end_point.grid(row=9, column=2, sticky='ew', pady=10)

    def process_input(self):
        start_x = self.entry_start_x.get()
        start_y = self.entry_start_y.get()
        end_x = self.entry_end_x.get()
        end_y = self.entry_end_y.get()

        if not all([start_x, start_y, end_x, end_y]):
            self.path_coor = None
            return

        try:
            start_x = float(start_x)
            start_y = float(start_y)
            end_x = float(end_x)
            end_y = float(end_y)
        except ValueError:
            self.path_coor = None
            return

        if (start_x, start_y) == (end_x, end_y):
            raise ValueError("Start point and end point cannot be the same.")

        self.path_coor = [(start_x, start_y), (end_x, end_y)]

    def info_pos(self):

        self.lbl_start_point.config(text=f"Start Point: ({self.px[0]:.2f}, {self.py[0]:.2f})")
        self.lbl_end_point.config(text=f"End Point: ({self.px[1]:.2f}, {self.py[1]:.2f})")

    def confirm_para_1(self):
        c1 = self.df1.loc[:, self.combobox1.get()].values
        c2 = self.df1.loc[:, self.combobox2.get()].values
        self.scatter1 = []

        p1 = self.T.L2C(c1)
        p2 = self.T.B2C(c2)

        xlim_1 = self.Ax_1.get_xlim()
        ylim_1 = self.Ax_1.get_ylim()

        xlim_3 = self.Ax_3.get_xlim()
        ylim_3 = self.Ax_3.get_ylim()

        for i in range(len(c1)):
            scatter_0 = self.Ax_3.scatter(c1[i], c2[i], marker='*', s=15, c='red')
            scatter = self.Ax_1.scatter(p1[i], p2[i], marker='*', s=15, c='red')
            self.scatter1.append(scatter_0)

        self.Ax_1.set_xlim(xlim_1)
        self.Ax_1.set_ylim(ylim_1)
        self.Ax_3.set_xlim(xlim_3)
        self.Ax_3.set_ylim(ylim_3)

        self.Figure_1.draw()
        self.Figure_3.draw()

    def confirm_para_2(self):
        c1 = self.df2.loc[:, self.combobox3.get()].values
        c2 = self.df2.loc[:, self.combobox4.get()].values

        p1 = self.T.L2C(c1)
        p2 = self.T.B2C(c2)

        self.scatter2 = []

        xlim_1 = self.Ax_1.get_xlim()
        ylim_1 = self.Ax_1.get_ylim()

        xlim_3 = self.Ax_3.get_xlim()
        ylim_3 = self.Ax_3.get_ylim()

        for i in range(len(c1)):
            scatter_0 = self.Ax_3.scatter(c1[i], c2[i], marker='o', s=15, c='blue')
            scatter = self.Ax_1.scatter(p1[i], p2[i], marker='o', s=15, c='blue')
            self.scatter2.append(scatter_0)

        self.Ax_1.set_xlim(xlim_1)
        self.Ax_1.set_ylim(ylim_1)
        self.Ax_3.set_xlim(xlim_3)
        self.Ax_3.set_ylim(ylim_3)

        self.Figure_1.draw()
        self.Figure_3.draw()

    def remove_para_1(self):
        for scatter in self.scatter1:
            scatter.remove()
        self.Figure_1.draw()
        self.Figure_3.draw()

    def remove_para_2(self):
        for scatter in self.scatter2:
            scatter.remove()
        self.Figure_1.draw()
        self.Figure_3.draw()

    @execute_specific_after_0('draw_pv_plot')
    def get_pos_pv(self):
        self.process_input()
        if self.path_coor:
            pass
        else:
            self.path_coor = self.Figure_3.figure.ginput(2)
        self.px, self.py = zip(*self.path_coor)
        self.info_pos()

    def update_contours(self, *args):
        levels_factor = self.rms_factor_spinbox.get()
        levels_factor = float(levels_factor)
        self.draw_contours(levels_factor)

    def update_contours_indi(self, *args):
        levels_factor = self.rms_factor_spinbox_indi.get()
        levels_factor = float(levels_factor)
        self.draw_contours_indi(levels_factor)

    def draw_pv_plot(self):
        self.Ax_4.clear()
        self.contours = []
        self.images = {}

        g = Galactic(self.px * u.deg, self.py * u.deg)
        path = Path(g, width=90 * u.arcsec)

        hdu_list = [self.hdu1, self.hdu2, self.hdu3]
        for hdu in hdu_list:
            if hdu:
                img, extent = self.process_slice(hdu, path)
                self.images[hdu] = (img, extent)

        img12, ext12 = self.images.get(self.hdu1, (None, None))
        if img12 is not None:
            self.Ax_4.imshow(img12.T, origin='lower', cmap='Purples', extent=ext12, aspect='auto')

        self.Ax_4.set_ylabel('Position (deg)')
        self.Ax_4.set_xlabel('Velocity (km/s)')

        try:
            if self.arrow:
                self.arrow.remove()
                self.arrow = None

            if self.path_coor:
                self.arrow = mpatches.FancyArrowPatch(self.path_coor[0], self.path_coor[1],
                                                      mutation_scale=10, color='black', alpha=0.3)
                self.Ax_3.add_patch(self.arrow)
                self.Figure_3.draw()
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while plotting the line: {e}")

        self.Figure_4.draw()

    @execute_specific_after_0('draw_spectra')
    def checking(self, rect):
        alpha_x, y1, beta_x, y2 = rect
        alpha_y, beta_y = y1, y2
        l0, l1 = self.px
        b0, b1 = self.py
        l_del = abs(l0 - l1)
        b_del = abs(b0 - b1)
        line_length = np.sqrt(l_del ** 2 + b_del ** 2)

        if l0 > l1:
            lx0 = l0 - (l_del * y1 / line_length)
            lx1 = l0 - (l_del * y2 / line_length)
        else:
            lx0 = (l_del * y1 / line_length) + l0
            lx1 = (l_del * y2 / line_length) + l0

        if b0 > b1:
            by0 = b0 - (b_del * y1 / line_length)
            by1 = b0 - (b_del * y2 / line_length)
        else:
            by0 = (b_del * y1 / line_length) + b0
            by1 = (b_del * y2 / line_length) + b0

        x1, x2, y1, y2 = self.T.L2C(lx0), self.T.L2C(lx1), self.T.B2C(by0), self.T.B2C(by1)

        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        x = np.arange(x1, x2, 1)
        y = np.arange(y1, y2, 1)
        x_mesh, y_mesh = np.meshgrid(x, y)
        x = x_mesh.ravel()
        y = y_mesh.ravel()
        self.pix_l, self.pix_b = x, y

        re_x = max(lx0, lx1)
        re_y = min(by0, by1)
        lengths = abs(lx0 - lx1)
        widths = abs(by0 - by1)
        re_x = re_x - lengths

        alpha_x, beta_x = sorted([alpha_x, beta_x])
        alpha_y, beta_y = sorted([alpha_y, beta_y])
        L = beta_x - alpha_x
        B = beta_y - alpha_y

        re_4 = patches.Rectangle((alpha_x, alpha_y), L, B, linewidth=0.5, edgecolor='green', facecolor='none')

        re = patches.Rectangle((re_x, re_y), lengths, widths, linewidth=0.8, edgecolor='black', facecolor='none')
        self.Ax_3.add_patch(re)
        self.Ax_4.add_patch(re_4)
        self.Figure_4.draw()
        self.Figure_3.draw()

    def process_slice(self, hdu, path):
        slices = extract_pv_slice(hdu, path)
        img = slices.data
        extent = self.ax_extent(slices.header)
        return img, extent

    def draw_contours(self, levels_factor):
        # self.Figure_4.draw()
        for contour in self.contours:
            for c in contour.collections:
                c.remove()
        self.contours = []

        hdu_color_linewidth = {
            self.hdu1: ('black', 1),
            self.hdu2: ('red', 1),
            self.hdu3: ('blue', 1),
        }

        for hdu, (img, extent) in self.images.items():
            if img is not None:
                rms = Static.Rms(img)
                if rms == 0:
                    rms = np.nanstd(img)
                levels = np.arange(levels_factor * rms, 18 * rms, 6 * rms)
                color, linewidth = hdu_color_linewidth.get(hdu)
                contour = self.Ax_4.contour(img.T, levels=levels, colors=color, linewidths=linewidth, extent=extent)
                self.contours.append(contour)
        self.Figure_4.draw()

    def draw_contours_indi(self, levels_factor):
        if self.cls:
            color = 'blue'
        else:
            color = 'red'
        for c in self.contours_indi[color]:
            try:
                for coll in c.collections:
                    coll.remove()
            except Exception as e:
                print(f"Error removing collection: {e}")

        if self.rms_indi == 0:
            levels = np.arange(levels_factor * self.mx, 1 * self.mx, 0.1 * self.mx)
            self.N = levels_factor
        else:
            levels = np.arange(levels_factor * self.rms_indi, 21 * self.rms_indi, 3 * self.rms_indi)
            self.N = levels_factor
        contour = self.ax_indi.contour(self.img_indi, levels=levels, colors=color, linewidths=1, extent=self.ext_indi)
        self.contours_indi[color].append(contour)
        self.fig_indi.canvas.draw()

    @staticmethod
    def ax_extent(hdr):
        axis_1_front = (hdr['NAXIS1'] - hdr['CRPIX1']) * hdr['CDELT1']
        axis_1_back = (0 - hdr['CRPIX1']) * hdr['CDELT1']
        axis_2_front = ((hdr['NAXIS2'] - hdr['CRPIX2']) * hdr['CDELT2']) / 1e3
        axis_2_back = ((0 - hdr['CRPIX2']) * hdr['CDELT2']) / 1e3
        axis_1 = (axis_1_front, axis_1_back)
        axis_2 = (axis_2_front, axis_2_back)
        ext = (axis_2[1], axis_2[0], axis_1[1], axis_1[0])
        return ext

    def plot_pv_line(self):
        try:
            if self.arrow:
                self.arrow.remove()
                self.arrow = None

            if self.path_coor:
                self.arrow = mpatches.FancyArrowPatch(self.path_coor[0], self.path_coor[1],
                                                      mutation_scale=10, color='black', alpha=0.3)
                self.Ax_3.add_patch(self.arrow)
                self.Figure_3.draw()
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while plotting the line: {e}")

    def update_arrow(self):
        try:
            if self.arrow:
                self.arrow.remove()
                self.arrow = None
                self.Figure_3.draw()
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while updating the arrow: {e}")


class OutFlow(Widget_4):

    def __init__(self):

        super().__init__()

        self.root.mainloop()


app = OutFlow()
