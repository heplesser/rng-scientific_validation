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
import json
import numpy as np


class Validate:
    """
    Statistical analysis of simulation results.
    
    Analysis is performed separately for each subcase, i.e. 
    area-layer-population combination. 
    
    If blocks are specified, the trials for each subcase are split
    into blocks according to the indicies provided and blocks are 
    tested against each other. Results are returned for each version-block 
    combination against all others, excluding self-combination and symmetric
    duplicates. Blocking applies to the p_val cases only.    
    """
    
    def __init__(self, analysis_path, quantity, model, versions, blocks=None):
        """
        analysis_path: directory with data for model to check
        quantity: quantity to check, eg, 'rate', 'cv_isi'; 
                  available quantities depend on model
        model: model to analyse, '4x4' or 'mam'
        versions: strings specifying code versions, see model-specific _load_data_...
                  Must be length 2!
        blocks: list of index ranges for cross-wise comparison (block vs block)
        """

        self.analysis_path = analysis_path
        self.quantity = quantity
        self.save_path = os.path.join(analysis_path, '..', '..', 'plots')
        self.model = model
        self.versions = versions
        self.blocks = blocks

        # TODO add sth similar for 4x4 if needed
        if self.model== 'mam': 
            self.colors = {
                'master-rng': '#ee7733',
                '2.20.1': '#01796f'
            }
        sns.set()

    def load_data(self):
        """Read data from directories."""

        data = {}
        for version in self.versions:
            if self.model == 'mam':
                data[version] = self._load_data_mam(version)
            elif self.model == '4x4':
                data[version] = self._load_data_4x4(version)
            else:
                raise Exception(f'No data loading implemented for {self.model}')
        self.data = data

# USER ACCESSIBLE FUNCTIONS

    def plot_ks_score(self):
        self._plot_ks_score(self._calc_ks_score())

    def plot_p_val(self, stest, tname, qname, qix, gs, fig):
        """
        Plot p-values for statistical tests performed for one quantity.
        
        Plots figures for both populations (E, I) for a given quantity.
        For each population, a heatmap shows p-values for all area-layer
        combinations.
        
        stest: test to perform, currently 'KS', 'ES' supported
        tname: test name for plot labeling
        qname: name of quantity to analyse, for plot labeling
        qix: index of quantity to analyse, for plot row
        gs: gridspec for figure
        fig: figure instance
        
        returns: axes from plotting, array of p-values for all sub-cases
        """
        
        pv = self._calc_p_val(stest)
        ax = self._plot_p_val(pv, tname, qname, qix, gs, fig)
        return ax, pv

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

    def _load_data_mam(self, version):

        sim_hashes = self._fetch_sim_hashes(version)

        sample_data = pd.read_pickle(os.path.join(
            self.analysis_path, version,
            sim_hashes[0],
            '348cd785d210258c0da5cceaee62b897',
            self.quantity + '.pkl')).to_xarray()

        # Dummy aray to initalize x-array for connection probabilities
        dummy_data_array = np.zeros((np.shape(sample_data)[0],
                                     np.shape(sample_data)[1],
                                     np.shape(sample_data)[2],
                                     len(sim_hashes)))

        self.areas = sample_data.coords['area'].values
        self.layers = sample_data.coords['layer'].values
        self.pops = sample_data.coords['pop'].values

        data = xr.DataArray(
            dummy_data_array,
            coords={'area': self.areas,
                    'layer': self.layers,
                    'pop': self.pops,
                    'sim_hash': sim_hashes},
            dims=['area',
                  'layer',
                  'pop',
                  'sim_hash'])

        for sim_hash in sim_hashes:
            data.loc[:, :, :, sim_hash] = pd.read_pickle(
                os.path.join(
                    self.analysis_path,
                    version,
                    sim_hash,
                    '348cd785d210258c0da5cceaee62b897',
                    self.quantity + '.pkl')).to_xarray()
        return data


    def _load_data_4x4(self, version):
        with open(os.path.join(
            self.analysis_path, f'seed_comparison_nest_{version}.txt')) as f:
            dic = json.load(f)

        with open(os.path.join(self.analysis_path, 'psview_dict.txt')) as f:
            psview = json.load(f)
            master_seeds = psview['seed_mesocircuit']['custom_params'][
                                  'ranges']['sim_dict']['master_seed']

        self.areas = [''] # only one exists
        self.layers = ['L23', 'L4', 'L5', 'L6']
        self.pops = ['E', 'I']
        num_samples = len(dic['FRs']['mean']['L23E']['values'])

        data_array = np.zeros(
            shape=(1, len(self.layers), len(self.pops), num_samples))

        for l, layer in enumerate(self.layers):
            for p, pop in enumerate(self.pops):
                # take mean value per population
                data_array[0, l, p] = \
                    dic[self.quantity]['mean'][layer + pop]['values']

        data = xr.DataArray(
            data_array,
            coords={'area': self.areas,
                    'layer': self.layers,
                    'pop': self.pops,
                    'master_seeds': master_seeds},
            dims=['area',
                  'layer',
                  'pop',
                  'master_seeds'])
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
                    x = self.data[self.versions[0]].loc[area, layer, pop]
                    y = self.data[self.versions[1]].loc[area, layer, pop]
                    ks_score.loc[area, layer, pop] = sp.stats.kstest(x, y).statistic
        return ks_score

    def _calc_p_val(self, stest):
        """
        Calculate p-values for given statistical test.
        
        For each sub-case (area-layer-population combination) compare
        data from each version to each other version. If blocks are
        provided, compare all version-block combination except redundant
        cases.
        
        stest: 'KS' or 'ES'
        
        returns: xarray with p-values with dimensions area-layer-population
        """
        
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
                    if self.blocks:
                        x = self.data[self.versions[0]].loc[area, layer, pop][self.blocks[0]]
                        y = self.data[self.versions[1]].loc[area, layer, pop][self.blocks[1]]
                    else:
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
        """
        Plot p-values as heatmap for a given quantity, both E and I population.
        
        p_val: xarray with p-values, dimensions area-layer-populationsupported
        test_name: test name for plot labeling
        qname: name of quantity to analyse, for plot labeling
        qix: index of quantity to analyse, selects plot row
        gs: gridspec for figure
        fig: figure instance
        
        returns: one pcolormesh instance (for later addition of colorbar)
        """

        for pix, pop in enumerate(self.pops):
            ax = fig.add_subplot(gs[qix, pix])

            im = ax.pcolormesh(p_val.loc[:, :, pop].T,
                        cmap='plasma_r',
                        norm=colors.BoundaryNorm(boundaries=[0.005, 0.01, 0.05, 0.1, 1],
                                                 ncolors=256, extend='min')
                              )
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

    model_specs = [('mam',                       # short / directory name
                    'Multi-Area Model',          # full name
                    ['2.20.1', 'master-rng'],    # code versions
                    ['rates', 'cv_isi', 'cc']),  # names of quantities
                   ('4x4', 
                    '4x4mm2 Model',
                    ['feature_cpp11-librandom_14211b8df', 'master_f1c0bcf43'],
                    ['FRs', 'LVs', 'CCs'])
                  ] 
    
    stat_specs = [('KS',                  # short name of test
                   'Kolmogorov-Smirnov',  # full name of test
                   sp.stats.kstest),      # test function
                  ('ES', 
                   'Epps-Singleton', 
                   sp.stats.epps_singleton_2samp)]
    
    # hierarchical dictionary to collect p-values by model-test-quantity
    # at innermost level, keys are strings identifying code versions
    p_all = {m: {s: {q: {(va, vb): {} 
                         for va in sorted(vers) for vb in sorted(vers) if va <= vb} 
                     for q in quants} 
                 for s, _, _ in stat_specs} 
             for m, _, vers, quants in model_specs}
    
    # run over all models
    for modshort, modlong, versions, quantities in model_specs:
        
        # for each model, perform all statistical tests
        for statshort, statlong, func in stat_specs:
            
            # all code combinations
            for ver_a in versions:
                for ver_b in versions:
                    if ver_a > ver_b: 
                        continue  # redundant B-A, A-B has been done already
                        
                    # all block combinations
                    # hardcodes that we have at least 28 trials per model-version combination
                    for bl_1 in [list(range(14)), list(range(14, 28))]:
                        for bl_2 in [list(range(14)), list(range(14, 28))]:
                            
                            if ver_a == ver_b and bl_1 >= bl_2:
                                continue   # redundant 2-1 if 1-2 has been done for same version with itself
                            
                            # graphics setup
                            fig = plt.figure(figsize=(7, 8))
                            fig.suptitle(f'{modshort} / {statshort} Test [{ver_a}/{bl_1[0]} vs {ver_b}/{bl_2[0]}]')
                            gs = GridSpec(4, 2, width_ratios=[1, 1], height_ratios=[4, 4, 4, 1])

                            # one plot in row per quantity analysed (rate, CV, CC)
                            for ix, quantity in enumerate(quantities):
                                validate = Validate(
                                    analysis_path=f'./data/{modshort}',
                                    quantity=quantity,
                                    model=modshort,
                                    versions=[ver_a, ver_b],
                                    blocks=[bl_1, bl_2])
                                validate.load_data()
                                ax, pvals = validate.plot_p_val(func, statshort, quantity, ix, gs, fig)
                                
                                # collect p-val results in hierachical dictionary
                                # mark which blocks were used by pair of first indicies in block 1 and block 2
                                # store p-values as linear sorted array of all area-layer-population combinations
                                p_all[modshort][statshort][quantity][(ver_a, ver_b)][(bl_1[0], bl_2[0])] = np.sort(pvals.to_series().dropna().values)
                                
                            # add colorbar for entire figure    
                            ax_cb = fig.add_subplot(gs[-1, :])
                            plt.colorbar(ax, cax=ax_cb, orientation='horizontal', label='p-value', shrink=0.5)

                            # store figure, PDF and PNG, mark with code version and blocks
                            figfn = f'rngtest_{modshort}_{statshort}-test_{ver_a}_{ver_b}_{bl_1[0]}_{bl_2[0]}'
                            plt.savefig(os.path.join(validate.save_path, f'{figfn}.pdf'))
                            plt.savefig(os.path.join(validate.save_path, f'{figfn}.png'))

        
        # Summary statistics for each model (MAM, 4x4)
        # For each version-block x version-block combination, plot CDF of p-values
        # Group figures in three colors: 
        #    - Old NEST vs Old NEST
        #    - New NEST vs New NEST
        #    - New NEST vs Old NEST
        #
        # Since we only have two blocks, we only have a single data set for Old-Old and New-New each,
        # and four sets for New-Old.
        #
        # Expected CDF is for a uniform distribution, i.e., the identity diagonal.
        #
        # Curves below the diagonal for low p-values indicate that small p-values, pointing to unlikely
        # results, occur less than expected, i.e., indicate that we rarely reject the null hypothesis tha
        # the data from A and B come from the same distribution.
        #
        # NOTE: SINCE DATA FOR THE VARIOUS AREAS, LAYERS, POPULATIONS COME FROM THE SAME NETWORK SIMULATION.
        #       THEY ARE NOT STATISTICALLY INDEPENDENT, SO WE CANNOT REALLY EXPECT UNIFORM DISTRIBUTIONS HERE.
        #       THEREFORE, COMPARISON WITH THE OLD-OLD AND NEW-NEW CASES ARE IMPORTANT.
        #
        # NOTE: FOR KS, DUE TO THE SMALL NUMBER OF TRIALS PER BLOCK (14), P-VALUES ARE HIGHLY DISCRETIZED.
        #
        # NOTE: FOR 4X4MM, THERE IS ONLY A SINGLE POPULATION, I.E., IN THE HEATMAPS EACH LAYERS IS SHOWN AS
        #       ONE LONG BAR REPRESENTING JUST A SINGLE VALUE. THE FEW DATA POINTS THUS AVAILABLE LEAD TO
        #       DISCRETE ES-CDFS IN THIS CASE.
        line_colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:3]
        name_mapping = {'mam': {'2.20.1': 'Old', 'master-rng': 'New'},
                        '4x4': {'feature_cpp11-librandom_14211b8df': 'New', 'master_f1c0bcf43': 'Old'}}
        stat_map = {s: l for s, l, _ in stat_specs}
        fig = plt.figure(figsize=(7, 8))
        fig.suptitle(f'{modlong} / p-value CDFs')
        gs = GridSpec(3, 2, width_ratios=[1, 1], height_ratios=[1, 1, 1])

        for ixs, (sts, stl) in enumerate(stat_map.items()):
            for ixq, q in enumerate(quantities):
                ax = fig.add_subplot(gs[ixq, ixs])
                
                for ixc, (combo, vals) in enumerate(p_all[modshort][sts][q].items()):
                    for lctr, d in enumerate(vals.values()):
                        # compute and plot CDF
                        x = np.sort(d)
                        y = np.arange(len(x)) / len(x)
                        if lctr == 0:
                            # add label only once per color
                            lbl = '{} vs {}'.format(*sorted(name_mapping[modshort][vers] for vers in combo))
                        else:
                            lbl = None
                        plt.step(x, y, label=lbl, c=line_colors[ixc], lw=4, alpha=0.7);
                plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Uniform');
                ax.set_aspect('equal');
                if ixq == 0 and ixs == 0:
                    ax.legend();
                if ixs > 0:
                    ax.set_yticks([])
                else:
                    ax.set_ylabel('Frequency of p-value')
                if ixq < len(quantities) - 1:
                    ax.set_xticks([])
                else:
                    ax.set_xlabel('Observed p-value')
                ax.set_title(f'{q}: {stl}')
                
        # Save to file, once per model.
        plt.savefig(os.path.join(validate.save_path, f'rngtest_summary_{modshort}.pdf'))
        plt.savefig(os.path.join(validate.save_path, f'rngtest_summary_{modshort}.png'))
                            
    plt.show()
