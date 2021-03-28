import os

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import seaborn as sns
import scipy as sp
from statsmodels.graphics.gofplots import qqplot_2samples
import xarray as xr


class Validate:
    def __init__(self, analysis_path, quantity):

        self.analysis_path = analysis_path
        self.quantity = quantity
        self.save_path = os.path.join(analysis_path, '..', 'plots')
        self.versions = ['2.20.1', 'master-rng']
        self.all_sim_hashes = {}
        for version in self.versions:
            self.all_sim_hashes[version] = self._fetch_sim_hashes(version)

        self.colors = {
            'master-rng': '#ee7733',
            '2.20.1': '#01796f'
        }

        self.load_data()
        sns.set()

    def load_data(self):
        data = {}
        for version in self.versions:
            data[version] = self._load_data(version)
        self.data = data

# USER ACCESSIBLE FUNCTIONS

    def plot_ks_score(self):
        self._plot_ks_score(self._calc_ks_score())

    def plot_p_val(self, stest, tname, qname, qix, gs, fig):
        return self._plot_p_val(self._calc_p_val(stest), tname, qname, qix, gs, fig)

    def plot_qq(self, area, layer):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
        for pop in self.pops:
            qqplot_2samples(self.data['2.20.1'].loc[area, layer, pop].values,
                            self.data['master-rng'].loc[area, layer, pop].values,
                            ax=ax)
            plt.legend()
            plt.savefig(
                os.path.join(self.save_path, '_'.join(['qq',
                    self.quantity, area, layer, pop]) + '.pdf'))

    def plot_hist(self, area, layer):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
        for pop in self.pops:
            for version in self.versions:

                ax.hist(self.data[version].loc[area, layer, pop].values,
                        color=self.colors[version],
                        density=True,
                        histtype='step',
                        bins=20,
                        linewidth=2,
                        label=version)

            plt.legend()
            plt.savefig(
                os.path.join(self.save_path, '_'.join([
                    self.quantity, area, layer, pop]) + '.png'),
                dpi=600)


# INTERNAL FUNCTIONS

    def _fetch_sim_hashes(self, version):
        sim_hashes = os.listdir(path=os.path.join(self.analysis_path, version))
        for annoying_macos_dir in ['.DS_Store', '._.DS_Store']:
            while annoying_macos_dir in sim_hashes:
                sim_hashes.remove(annoying_macos_dir)
        return sim_hashes

    def _load_data(self, version):

        sample_data = pd.read_pickle(os.path.join(
            self.analysis_path, version,
            self.all_sim_hashes[version][0],
            '348cd785d210258c0da5cceaee62b897',
            self.quantity + '.pkl')).to_xarray()

        # Dummy aray to initalize x-array for connection probabilities
        dummy_data_array = np.zeros((np.shape(sample_data)[0],
                                     np.shape(sample_data)[1],
                                     np.shape(sample_data)[2],
                                     len(self.all_sim_hashes[version])))

        self.areas = sample_data.coords['area'].values
        self.layers = sample_data.coords['layer'].values
        self.pops = sample_data.coords['pop'].values

        data = xr.DataArray(
            dummy_data_array,
            coords={'area': self.areas,
                    'layer': self.layers,
                    'pop': self.pops,
                    'sim_hash': self.all_sim_hashes[version]},
            dims=['area',
                  'layer',
                  'pop',
                  'sim_hash'])

        for sim_hash in self.all_sim_hashes[version]:
            data.loc[:, :, :, sim_hash] = pd.read_pickle(
                os.path.join(
                    self.analysis_path,
                    version,
                    sim_hash,
                    '348cd785d210258c0da5cceaee62b897',
                    self.quantity + '.pkl')).to_xarray()
        return data

    def _calc_ks_score(self):
        # Dummy array to initalize x-array for connection probabilities
        dummy_data_array = np.zeros((len(self.areas),
                                     len(self.layers),
                                     len(self.pops)))

        ks_score = xr.DataArray(
            dummy_data_array,
            coords={'area': self.areas,
                    'layer': self.layers,
                    'pop': self.pops},
            dims=['area',
                  'layer',
                  'pop'])

        for area in self.areas:
            for layer in self.layers:
                for pop in self.pops:
                    ks_score.loc[area, layer, pop] = sp.stats.kstest(
                        self.data[self.versions[0]].loc[area, layer, pop],
                        self.data[self.versions[1]].loc[area, layer, pop]).statistic
        return ks_score

    def _calc_p_val(self, stest):
        # Dummy aray to initalize x-array for connection probabilities
        dummy_data_array = np.zeros((len(self.areas),
                                     len(self.layers),
                                     len(self.pops)))

        pval = xr.DataArray(
            dummy_data_array,
            coords={'area': self.areas,
                    'layer': self.layers,
                    'pop': self.pops},
            dims=['area',
                  'layer',
                  'pop'])

        for area in self.areas:
            for layer in self.layers:
                for pop in self.pops:
                    x = self.data[self.versions[0]].loc[area, layer, pop]
                    y = self.data[self.versions[1]].loc[area, layer, pop]
                    if np.all(np.isfinite(x)) and np.all(np.isfinite(y)):
                        try:
                            pval.loc[area, layer, pop] = stest(x, y).pvalue
                        except np.linalg.LinAlgError:
                            pval.loc[area, layer, pop] = np.nan
                    else:
                        pval.loc[area, layer, pop] = np.nan

        return pval

    def _plot_ks_score(self, ks_score):
        for pop in self.pops:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))

            sns.heatmap(ks_score.loc[:, :, pop])
            ax.set_xticklabels(self.layers)
            ax.set_yticklabels(self.areas[::2], rotation=0)
            plt.savefig(
                os.path.join(
                    self.save_path,
                    '_'.join([self.quantity, 'ks_score', pop]) + '.png'),
                dpi=600)

    def _plot_p_val(self, p_val, test_name, qname, qix, gs, fig):
        for pix, pop in enumerate(self.pops):
            ax = fig.add_subplot(gs[qix, pix])

            im = ax.pcolormesh(p_val.loc[:, :, pop].T,
                        cmap='plasma_r',
                        norm=colors.BoundaryNorm(boundaries=[0.005, 0.01, 0.05, 0.1, 1],
                                                 ncolors=256, extend='min'))
            if pix == 0:
                ax.set_yticks(0.5 + np.arange(0, len(self.layers)))
                ax.set_yticklabels(self.layers)
            else:
                ax.set_yticks([])
            if qix == 2:
                ax.set_xticks(range(0, len(self.areas), 6))
                ax.set_xticklabels(self.areas[::6], rotation=0)
            else:
                ax.set_xticks([])
            ax.set_title(f'{qname} ({pop}-pop): {test_name}-test')

        return im


if __name__ == '__main__':
    for short, long, func in [('KS', 'Kolmogorov-Smirnov', sp.stats.kstest),
                              ('ES', 'Epps-Singleton', sp.stats.epps_singleton_2samp)]:
        fig = plt.figure(figsize=(7, 8))
        fig.suptitle(f'Multi-area Model / {long} Test')
        gs = GridSpec(4, 2, width_ratios=[1, 1], height_ratios=[4, 4, 4, 1])

        for ix, quantity in enumerate(['rates', 'cv_isi', 'cc']):
            validate = Validate(
                analysis_path='./data',
                quantity=quantity)
            ax = validate.plot_p_val(func, short, quantity, ix, gs, fig)
        ax_cb = fig.add_subplot(gs[-1, :])
        plt.colorbar(ax, cax=ax_cb, orientation='horizontal', label='p-value', shrink=0.5)

        plt.savefig(os.path.join(validate.save_path, f'rngtest_mam_{short}-test.pdf'))
        plt.savefig(os.path.join(validate.save_path, f'rngtest_mam_{short}-test.png'))

    plt.show()

    # validate.plot_hist('V1','4','I')
    # validate.plot_qq('V1','4')
