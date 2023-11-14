import itertools
import re
from pathlib import Path

import matplotlib.pyplot as plt
import mistat
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from mistat.design import doe
from scipy import stats
from PyPDF2 import PdfMerger

data_dir = Path(__file__).parent / 'data'
FIGURES_DIR = Path(__file__).parents[1] / 'figures'
FIGURES_DIR.mkdir(exist_ok=True)
TABLES_DIR = Path(__file__).parents[1] / 'tables'
TABLES_DIR.mkdir(exist_ok=True)

NREPEATS_DEFAULT = 100


def saveTable(filename, latex):
    tex_file = TABLES_DIR / filename
    tex_file.write_text(latex.replace('%', '\\%'))


def saveFigure(filename):
    pdf_file = FIGURES_DIR / filename
    if filename.endswith('png'):
        plt.savefig(pdf_file, format="png", bbox_inches="tight", dpi=600)
        return
    plt.savefig(pdf_file, format="pdf", bbox_inches="tight")
    writer = PdfMerger()
    content = pdf_file.read_bytes()
    content = re.sub(br'\/CreationDate \(D:\d+\D*\)',
                     b'/CreationDate (D:20230823000000Z)', content)
    content = re.sub(br'\/ModDate \(D:\d+\D*\)',
                     b'/ModDate (D:20230823000000Z)', content)
    pdf_file.write_bytes(content)


def preparePistonDataset() -> pd.DataFrame:
    factors = {
        's': [0.01, 0.015],
        'v0': [0.00625, 0.00875],
        'k': [2000, 4000],
        't0': [345, 355],
    }
    Design = doe.central_composite(factors, alpha='r', center=[4, 4])

    seed = 2
    # seed = 1234
    simulator = mistat.PistonSimulator(**Design, m=60, p0=110_000, t=296,
                                       n_replicate=5, seed=seed)
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

    # define helper function to convert code co-ordinates to factor co-ordinates
    def toFactor(code, codeValue):
        ''' convert code to factor co-ordinates '''
        factor = x2factor[code]
        return center[factor] + codeValue * unit[factor]

    # add code levels to table
    for c in factors:
        result[factor2x[c]] = (result[c] - center[c]) / unit[c]
    return result[['milliseconds', 'group', 'x1', 'x2', 'x3', 'x4']]


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


# def nullFactorAnalysis(df: pd.DataFrame, formula: str, nrepeats: int = 100) -> pd.DataFrame:
#     df = df.copy()
#     results = []
#     formula = f'{formula} + null'
#     for _ in range(nrepeats):
#         df['null'] = stats.norm.rvs(loc=0, scale=1, size=len(df))
#         results.append(buildModel(df, formula).params)
#     return pd.DataFrame(results)


def createParametricBootstrapSample(df: pd.DataFrame, outcome: str, rng=None):
    df = df.copy()
    y = df[outcome]
    df[outcome] = stats.norm.rvs(loc=np.mean(
        y), scale=np.std(y), size=len(y), random_state=rng)
    return df


def parametricBefittingBootstrapAnalysis(df: pd.DataFrame, formula: str, group: list[str], nrepeats: int = NREPEATS_DEFAULT) -> pd.DataFrame:
    rng = np.random.default_rng(seed=123)
    outcome = formula.split('~')[0].strip()
    return pd.DataFrame([buildModel(df.groupby(group).apply(createParametricBootstrapSample, outcome=outcome, rng=rng), formula).params
                         for _ in range(nrepeats)])


def plot_coefficients(ols_model, bba_samples, ba_samples, pbba_samples, pba_samples, wba_samples, *, ncols=6):
    terms = list(ols_model.params.index)
    if len(terms) > ncols:
        nrows = (len(terms) - 1) // ncols + 1
        fig, axes = plt.subplots(
            ncols=ncols, nrows=nrows, figsize=[16 * ncols / 6, 3 * nrows])
        axes = itertools.chain(*axes)
    else:
        fig, axes = plt.subplots(ncols=ncols, nrows=1,
                                 figsize=[16 * ncols / 6, 3])

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

        ax.set_title(term)
        ax.get_yaxis().set_visible(False)
        ax.set_xticks([0, 1, 2, 3, 4, 5], [
                      'OLS', 'BBA', 'BA', 'pBBA', 'pBA', 'wBA'])
        ax.set_xlim(-0.5, 5.5)
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
