import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy as sp
import xarray as xr


class Validate():
    def __init__(self, analysis_path, quantity):

        self.analysis_path = analysis_path
        self.quantity = quantity
        self.save_path = os.path.join(analysis_path, 'plots')
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

    def plot_hist(self, area, layer, pop):
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
        return os.listdir(path=os.path.join(self.analysis_path, version))

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
        # Dummy aray to initalize x-array for connection probabilities
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


for quantity in ['rates', 'cv_isi', 'cc']:
    validate = Validate(
        analysis_path='/p/project/cjinb33/albers2/data/rng',
        quantity=quantity)
    validate.plot_ks_score()
    # validate.plot_hist('V1','4','I')
