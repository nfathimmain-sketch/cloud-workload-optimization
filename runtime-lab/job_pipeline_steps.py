import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingRegressor

# columns a job must have at prediction time (all known at submission)
INPUT_COLUMNS = ['user_id', 'submit_time', 'node_count', 'wall_time',
                 'init_priority', 'queue', 'job_mode', 'gpu_enabled', 'project_prefix']
# finished jobs used to build the user history ('job_idx' = unique job id, used to skip duplicates)
HISTORY_COLUMNS = ['job_idx', 'user_id', 'submit_time', 'end_time', 'run_time']
CATEGORICAL = ['queue', 'job_mode', 'gpu_enabled', 'project_prefix']
NAT = np.iinfo('int64').min


def to_ns(series):
    ts = pd.to_datetime(series, utc=True, errors='coerce')
    return ts.dt.as_unit('ns').astype('int64').to_numpy()


class JobInputError(ValueError):
    pass


def validate_jobs(X):
    if not isinstance(X, pd.DataFrame):
        raise JobInputError('Job input must be a pandas DataFrame.')
    if X.empty:
        raise JobInputError('Provide at least one job.')
    if not X.columns.is_unique:
        raise JobInputError('Job input contains duplicate column names.')
    # Simple checks on the raw job table. Raises JobInputError with a readable message.
    missing = [c for c in INPUT_COLUMNS if c not in X.columns]
    if missing:
        raise JobInputError(f'Missing required fields: {missing}')

    problems = []
    def check(bad, message):
        bad = np.asarray(bad)
        if bad.any():
            problems.append(f'{message} (row {list(X.index[bad][:5])})')

    nodes = pd.to_numeric(X['node_count'], errors='coerce')
    wall = pd.to_numeric(X['wall_time'], errors='coerce')
    when = pd.to_datetime(X['submit_time'], utc=True, errors='coerce')
    priority = pd.to_numeric(X['init_priority'], errors='coerce')

    check(X['user_id'].isna() | X['user_id'].astype(str).str.strip().eq(''), 'user_id is missing')
    check(~((nodes >= 1) & np.isfinite(nodes) & (nodes % 1 == 0)), 'node_count must be a whole number of at least 1')
    check(~((wall > 0) & np.isfinite(wall)), 'wall_time (minutes) must be a number greater than 0')
    check(when.isna(), 'submit_time is missing or not a valid date/time')
    check(X['init_priority'].notna() & ~np.isfinite(priority), 'init_priority must be a finite number (or empty)')
    # Finite raw values can still overflow when multiplied.
    with np.errstate(over='ignore', invalid='ignore'):
        check(~np.isfinite(wall * 60), 'wall_time is too large to convert to seconds')
        check(~np.isfinite(nodes * (wall * 60) / 3600), 'requested node-hours is too large')

    if problems:
        raise JobInputError('Invalid job input: ' + '; '.join(problems))


class JobFeatureEngineer(BaseEstimator, TransformerMixin):
    # job_id_col    : unique job id column, so the same finished job is never added to the history twice
    # mean_fallback : fallback for users with no history. None = learn it from the training data
    #                 (median of the history-mean feature); or give a fixed number, e.g. the original model's.

    def __init__(self, job_id_col='job_idx', mean_fallback=None, compute_median=False):
        self.job_id_col = job_id_col
        self.mean_fallback = mean_fallback
        self.compute_median = compute_median

    def fit(self, X, y=None, history=None):
        # history = ALL finished training jobs (any state), as in the project
        self.history_ = {}
        self.seen_ids_ = set()
        if history is None:
            raise JobInputError('Provide finished-job history when fitting this step.')
        self.add_completed_jobs(history)
        _, means, medians = self._history_features(history)
        if self.mean_fallback is None:
            self.mean_fill_ = float(np.nanmedian(means)) if np.isfinite(means).any() else 0.0
        else:
            self.mean_fill_ = float(self.mean_fallback)
        self.median_fill_ = float(np.nanmedian(medians)) if np.isfinite(medians).any() else 0.0
        return self

    def add_completed_jobs(self, jobs):
        # Add newly finished jobs to the stored history. A job whose id is already stored
        # (or repeated inside `jobs`) is skipped, so counts and averages stay correct.
        # Validate every record before updating stored history. No partial update on bad input.
        required = [self.job_id_col, 'user_id', 'submit_time', 'end_time', 'run_time']
        if not isinstance(jobs, pd.DataFrame) or not jobs.columns.is_unique:
            raise JobInputError('History must be a DataFrame with unique column names.')
        missing = [c for c in required if c not in jobs.columns]
        if missing:
            raise JobInputError(f'Missing history fields: {missing}')
        if jobs.empty:
            return self
        start, end = to_ns(jobs['submit_time']), to_ns(jobs['end_time'])
        runtime = pd.to_numeric(jobs['run_time'], errors='coerce').to_numpy(dtype=float)
        bad = ((start == NAT) | (end == NAT) | (end < start)
               | ~np.isfinite(runtime) | (runtime < 0)
               | jobs[self.job_id_col].isna().to_numpy()
               | jobs[self.job_id_col].astype(str).str.strip().eq('').to_numpy()
               | jobs['user_id'].isna().to_numpy()
               | jobs['user_id'].astype(str).str.strip().eq('').to_numpy())
        if bad.any():
            raise JobInputError(f'Invalid finished-job history (rows {list(jobs.index[bad][:5])}). Check IDs, timestamps and finite nonnegative runtimes.')
        d = pd.DataFrame({'id': jobs[self.job_id_col].to_numpy(),
                          'user_id': jobs['user_id'].to_numpy(),
                          'end': to_ns(jobs['end_time']),
                          'run': np.asarray(jobs['run_time'], dtype='float64')})
        d = d[(d['end'] != NAT) & d['run'].notna() & d['user_id'].notna()]
        d = d[~d['id'].duplicated() & ~d['id'].isin(self.seen_ids_)]
        self.seen_ids_.update(d['id'])

        for user, g in d.groupby('user_id'):
            old = self.history_.get(user)
            end = g['end'].to_numpy() if old is None else np.concatenate([old['end'], g['end']])
            run = g['run'].to_numpy() if old is None else np.concatenate([old['run'], g['run']])
            order = np.argsort(end, kind='stable')
            end, run = end[order], run[order]
            self.history_[user] = {'end': end, 'run': run,
                                   'csum': np.concatenate([[0.0], np.cumsum(run)])}
        return self

    def _history_features(self, X):
        n = len(X)
        submit = to_ns(X['submit_time'])
        counts, means, medians = np.zeros(n), np.full(n, np.nan), np.full(n, np.nan)
        for user, idx in pd.DataFrame({'u': X['user_id'].to_numpy()}).groupby('u').indices.items():
            h = self.history_.get(user)
            if h is None:
                continue
            c = np.searchsorted(h['end'], submit[idx], side='left')   # jobs finished STRICTLY BEFORE submit
            counts[idx] = c
            has = c > 0
            means[idx[has]] = h['csum'][c[has]] / c[has]
            # Only calculate the more expensive medians when the optional feature is selected.
            if self.compute_median:
                med = {k: np.median(h['run'][:k]) for k in np.unique(c[has])}
                medians[idx[has]] = [med[k] for k in c[has]]
        return counts, means, medians

    def transform(self, X):
        validate_jobs(X)
        submit = pd.to_datetime(X['submit_time'], utc=True, errors='coerce')
        out = pd.DataFrame(index=X.index)

        # numeric inputs
        out['node_count'] = pd.to_numeric(X['node_count'], errors='coerce')
        out['wall_time_seconds'] = pd.to_numeric(X['wall_time'], errors='coerce') * 60
        out['init_priority'] = pd.to_numeric(X['init_priority'], errors='coerce')

        # categorical inputs (made uniform text; missing stays missing for the imputer)
        for c in CATEGORICAL:
            out[c] = X[c].astype(str).where(X[c].notna(), np.nan).astype(object)

        # engineered features
        out['submit_hour'] = submit.dt.hour
        out['submit_dayofweek'] = submit.dt.dayofweek
        out['requested_node_hours'] = out['node_count'] * out['wall_time_seconds'] / 3600
        out['log_node_count'] = np.log1p(out['node_count'])

        counts, means, medians = self._history_features(X)
        out['user_previous_job_count'] = counts
        out['user_previous_mean_runtime'] = np.where(np.isnan(means), self.mean_fill_, means)
        out['user_previous_median_runtime'] = np.where(np.isnan(medians), self.median_fill_, medians)
        return out


class NonNegativeHGB(HistGradientBoostingRegressor):
    def predict(self, X):
        return np.maximum(super().predict(X), 0)