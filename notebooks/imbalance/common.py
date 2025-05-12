import itertools
from pathlib import Path
import statsmodels.api as sm
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import mistat
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from mistat.design import doe
from scipy import stats
import warnings

NREPEATS_DEFAULT = 100

FIGURES_DIR = Path.cwd() / 'figures'


def prepareCases(ncases: int, ngroups: int, n_replicate: int = 5) -> list[list[int]]:
    # ncases = 10_000
    # ngroups = 32
    fmin = -10
    cases = []
    for f in np.linspace(fmin, n_replicate, ncases):
        while True:
            case = np.rint([max(1, np.random.uniform(f, n_replicate + 0.05)) for _ in range(ngroups)])
            if np.max(case) == n_replicate:
                break
        cases.append(case)
    return [case.tolist() for case in cases]


def make_lowess(x, y, frac=0.1):
    # lowess will return our "smoothed" data with a y value for at every x-value
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lowess = sm.nonparametric.lowess(y, x, frac=frac)

    # unpack the lowess smoothed points to their values
    lowess_x = list(zip(*lowess))[0]
    lowess_y = list(zip(*lowess))[1]

    # run scipy's interpolation. There is also extrapolation I believe
    f = interp1d(lowess_x, lowess_y, bounds_error=False)

    # xnew = np.arange(min(x), max(x), (max(x) - min(x)) / 400)
    xnew = np.linspace(min(x), max(x), num=400)

    # this this generate y values for our xvalues by our interpolator
    # it will MISS values outsite of the x window (less than 3, greater than 33)
    # There might be a better approach, but you can run a for loop
    # and if the value is out of the range, use f(min(lowess_x)) or f(max(lowess_x))
    ynew = f(xnew)
    return pd.DataFrame({'x': xnew, 'y': ynew})


def preparePistonDataset(*, include_t0: bool = True, n_replicate: int = 5, seed: int = 2) -> pd.DataFrame:
    factors = {
        's': [0.01, 0.015],
        'v0': [0.00625, 0.00875],
        'k': [2000, 4000],
        # 't0': [345, 355],
    }
    defaults = {'m': 60, 'p0': 110_000, 't': 296}
    if include_t0:
        factors['t0'] = [345, 355]
    else:
        defaults['t0'] = 350
    Design = doe.central_composite(factors, alpha='r', center=[4, 4])

    simulator = mistat.PistonSimulator(**Design, **defaults,
                                       n_replicate=n_replicate, seed=seed)
    result = simulator.simulate()
    result['seconds'] = 1000 * result['seconds']
    result = result.rename({'seconds': 'milliseconds'}, axis=1)

    # transformation between factors and code levels
    factor2x = {factor: f'x{i}' for i, factor in enumerate(factors, 1)}
    x2factor = {f'x{i}': factor for i, factor in enumerate(factors, 1)}
    center = {factor: 0.5 * (max(values) + min(values))
              for factor, values in factors.items()}
    unit = {factor: 0.5 * (max(values) - min(values))
            for factor, values in factors.items()}

    # add code levels to table
    for c in factors:
        result[factor2x[c]] = (result[c] - center[c]) / unit[c]
    return result[['milliseconds', 'group', *list(x2factor)]]


def buildModel(df: pd.DataFrame, formula: str) -> sm.regression.linear_model.RegressionResultsWrapper:
    model = smf.ols(formula, data=df).fit()
    return model


def bootstrapAnalysis(df: pd.DataFrame, formula: str, nrepeats: int = NREPEATS_DEFAULT) -> pd.DataFrame:
    df = df.copy()
    return pd.DataFrame([buildModel(df.sample(frac=1, replace=True), formula).params
                         for _ in range(nrepeats)])


def befittingBootstrapAnalysis(df: pd.DataFrame, formula: str, group: list[str], nrepeats: int = NREPEATS_DEFAULT) -> pd.DataFrame:
    df = df.copy()
    return pd.DataFrame([buildModel(df.groupby(group).sample(frac=1, replace=True).reset_index(),
                                    formula).params
                         for _ in range(nrepeats)])


def parametricBootstrapAnalysis(df: pd.DataFrame, formula: str, nrepeats: int = NREPEATS_DEFAULT) -> pd.DataFrame:
    df = df.copy()
    model = buildModel(df, formula)
    residuals = model.resid
    outcome = formula.split('~')[0].strip()
    results = []
    for _ in range(nrepeats):
        df[outcome] = model.fittedvalues + \
            residuals.sample(frac=1, replace=True).reset_index(drop=True)
        results.append(buildModel(df, formula).params)
    return pd.DataFrame(results)


def wildBootstrapAnalysis(df: pd.DataFrame, formula: str, nrepeats: int = NREPEATS_DEFAULT) -> pd.DataFrame:
    df = df.copy()
    model = buildModel(df, formula)
    residuals = model.resid
    outcome = formula.split('~')[0].strip()
    results = []
    for _ in range(nrepeats):
        V = stats.norm.rvs(loc=0, scale=1, size=len(df))
        df[outcome] = model.fittedvalues + residuals * V
        results.append(buildModel(df, formula).params)
    return pd.DataFrame(results)


def createParametricBootstrapSample(df: pd.DataFrame, outcome: str, rng=None):
    df = df.copy()
    y = df[outcome]
    df[outcome] = stats.norm.rvs(loc=np.mean(y), scale=np.std(y), size=len(y), random_state=rng)
    return df


def parametricBefittingBootstrapAnalysis(df: pd.DataFrame, formula: str, group: list[str], nrepeats: int = NREPEATS_DEFAULT,
                                         seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed=seed)
    outcome = formula.split('~')[0].strip()
    return pd.DataFrame([buildModel(df.groupby(group).apply(createParametricBootstrapSample, outcome=outcome, rng=rng), formula).params
                         for _ in range(nrepeats)])


def plot_coefficients(ols_model, bba_samples, ba_samples, pbba_samples, pba_samples, wba_samples, *, ncols=6):
    terms = list(ols_model.params.index)
    if len(terms) > ncols:
        nrows = (len(terms) - 1) // ncols + 1
        fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=[20 * ncols / 6, 4 * nrows])
        axes = itertools.chain(*axes)
    else:
        fig, axes = plt.subplots(ncols=ncols, nrows=1, figsize=[20 * ncols / 6, 4])

    for ax, term in itertools.zip_longest(axes, terms):
        if term is None:
            ax.axis('off')
            continue
        if ax is None:
            continue
        ax.plot([0, 0], [ols_model.params[term] - 1.96*ols_model.bse[term],
                ols_model.params[term] + 1.96*ols_model.bse[term]])
        ax.scatter(0, ols_model.params[term], )
        ax.boxplot([bba_samples[term], ba_samples[term],
                    pbba_samples[term], pba_samples[term],
                    wba_samples[term]])

        def sample_distributions(samples, ax, offset):
            ax.plot([offset+0.2, offset+0.2],
                    samples.quantile([0.025, 0.975]), color='red')
            m = samples.mean()
            s = samples.std() * 1.96
            ax.plot([offset+0.3, offset+0.3], [m+s, m-s], color='green')
            ax.scatter(offset+0.3, m, color='green')

        sample_distributions(bba_samples[term], ax, 1)
        sample_distributions(ba_samples[term], ax, 2)
        sample_distributions(pbba_samples[term], ax, 3)
        sample_distributions(pba_samples[term], ax, 4)
        sample_distributions(wba_samples[term], ax, 5)
        sample_distributions(wba_samples[term], ax, 5)

        ax.set_title(term, fontsize=20)
        ax.get_yaxis().set_visible(False)
        ax.set_xticks([0, 1, 2, 3, 4, 5], [
                      'OLS', 'BBA', 'BA', 'pBBA', 'pBA', 'wBA'])
        labels = ax.get_xticklabels()
        for i, label in enumerate(labels):
            if i % 2 == 1:
                label.set_y(label.get_position()[1] - 0.025)  # Adjust the shift value as needed
        ax.set_xlim(-0.5, 5.5)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.axhline(0)


def plot_std_coefficients(ols_model, bba_samples, ba_samples, pbba_samples, pba_samples,
                          wba_samples, ax=None):
    std_df = pd.DataFrame({
        'LR (SE)': ols_model.bse,
        'BBA': bba_samples.agg(['mean', 'std']).transpose()['std'],
        'BA': ba_samples.agg(['mean', 'std']).transpose()['std'],
        'pBBA': pbba_samples.agg(['mean', 'std']).transpose()['std'],
        'pBA': pba_samples.agg(['mean', 'std']).transpose()['std'],
        'wBA': wba_samples.agg(['mean', 'std']).transpose()['std'],
    })
    ax = std_df.plot(style='.-', ax=ax)
    ax.set_ylabel('Standard deviation')
    ax.set_xlabel('Coefficient')
    ax.set_ylim(0, std_df.max().max() * 1.05)
    ax.set_xticks(range(len(std_df)))
    ax.set_xticklabels(std_df.index)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(8)
    return ax


def calculate_Delta(ols_model, samples):
    table = pd.DataFrame({
        'Regr.': ols_model.bse,
        'Bootstrap': samples.agg(['std']).transpose()['std'],
    })
    table['Delta'] = (100*(table['Bootstrap'] / table['Regr.'] - 1)).round(1)
    return table
