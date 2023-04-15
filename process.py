import pandas as pd
import numpy as np
from pandas import Timedelta

from datasets import Dataset



# fix timezones, gaps, and interpolate missing data
def preprocess(fx):    
    fx['datetime'] = pd.to_datetime(fx['datetime'], infer_datetime_format = True)
    # fx = fx[fx['datetime'].dt.year >= 2009].reset_index(drop = True)
    
    # convert to eastern time so that day light savings can be removed then back to UTC
    fx['datetime'] = fx['datetime'].dt.tz_localize(
        'US/Eastern', ambiguous = fx['datetime'].astype(bool), nonexistent='shift_forward'
    )
    fx['datetime'] = fx['datetime'].dt.tz_convert(None) - Timedelta('5 hours')
    fx = fx.drop_duplicates(subset = 'datetime', keep = 'last')
    assert not fx.duplicated(subset = 'datetime').any()
    
    # forward previous prices to fill time gaps
    fx = fx.set_index('datetime').asfreq('1min')
    fx['close'] = fx['close'].fillna(method = 'ffill')
    fx['volume'] = fx['volume'].fillna(0) # no volume in gaps (according to first rate data people)

    # fill open, high, and low with most recent close
    fx = fx.fillna({
        'open': fx['close'],
        'high': fx['close'],
        'low': fx['close']
    })
    
    # define the "trading day" as from starting on 10pm UTC of the previous day ending at 9:59pm UTC
    fx['day'] = fx.index.dayofweek 
    fx.loc[fx.between_time('22:00', '23:59').index, 'day'] += 1
    fx['day'] = fx['day'] % 7 # sunday = 6, then + 1 would be 7 but we want that roll over to monday = 0

    # ordinal trading day since start of data (just for help in preprocessing)
    fx['ordinal_day'] = (fx['day'] != fx['day'].shift()).cumsum()

    return fx


# form labels and overnight masks for training purposes
def create_training_data(fx, periods, leverage):
    # let's just arbitrary say there should be less than 400 zero volume -- shouldn't really mattter in the end
    voluminous_index = (fx['volume'] == 0.).groupby(fx['ordinal_day']).filter(lambda x: x.sum() < 400).index
    fx = fx.loc[voluminous_index]

    # calculate future deltas
    futures = {}
    for i in periods:
        futures_name = f'future_diff{i}'
        future_diff = leverage * (fx['close'].shift(-i) - fx['close']) / fx['close']
        futures[futures_name] = future_diff

    futures_df = pd.DataFrame(futures).fillna(0)
    future_cols = futures_df.columns
    fx = pd.concat((fx, futures_df), axis = 1)

    # clean up data odditity of future columns coming from the future of non-consecutive days (i.e. no overnight trading)
    overnight_masks = {}
    for i in periods:
        futures_name = f'future_diff{i}'
        no_futures = fx.between_time(f'{21 - i // 60}:{59 - i % 60}', '21:59').index
        fx.loc[no_futures, futures_name] = 0
        
        # also create a mask for overnight trades
        mask_name = f'overnight_mask{i}'
        mask = pd.Series(0, index = fx.index)
        mask.loc[no_futures] = 1
        
        # also mask out the zeros (i.e. no change) in the future columns
        mask.loc[fx[futures_name] == 0] = 1

        overnight_masks[mask_name] = mask
    
    masks_df = pd.DataFrame(overnight_masks)
    mask_cols = masks_df.columns
    fx = pd.concat((fx, masks_df), axis = 1)

    return fx, future_cols, mask_cols


# remove means from price data & volume
def demean(fx):
    # de mean prices by turning them into % change from previous minute close
    # sub and divide for numerical stability
    price_features = ['open', 'high', 'low', 'close']
    fx[price_features] = fx[price_features].sub(
        fx['close'].shift(1), axis = 0).div(
        fx['close'].shift(1), axis = 0
    )
        
    # de mean volume using 200ma (exact number doesn't matter much)
    fx["volume"] = fx['volume'] - fx['volume'].groupby(fx['ordinal_day']).rolling(200, min_periods = 0).mean().reset_index(drop = True, level = 0)

    return fx


def randomize_csv(fx):
    changes = fx.set_index('datetime')
    price_features = ['open', 'high', 'low', 'close']

    # convert open, high, low and close to percent change of previous close
    # sub and divide for numerical stability
    changes[price_features] = changes[price_features].sub(
        changes['close'].shift(1), axis = 0).div(
        changes['close'].shift(1), axis = 0
    )

    # use log returns for numerical stability
    changes[price_features] = np.log1p(changes[price_features])

    # adjust all price features so that the close log return mean is 0
    changes[price_features] = changes[price_features] - changes['close'].mean()

    # turn all closes changes positive (so we can randomize the sign later)
    negatives = changes[changes['close'] < 0].index
    changes.loc[negatives] = -changes.loc[negatives]

    # swap high and low for rows turned negative (to reflect sign change)
    changes.loc[negatives, ['high', 'low']] = changes.loc[negatives, ['low', 'high']].values

    # randomly turn changes negative with probability .5
    new_negatives = changes[np.random.rand(len(changes)) < .5].index
    changes.loc[new_negatives] = -changes.loc[new_negatives]

    # again reflect the highs and lows
    changes.loc[new_negatives, ['high', 'low']] = changes.loc[new_negatives, ['low', 'high']].values

    # convert closes to cumulative log return (cumsum of log returns is a stable log cumprod of returns)
    changes['close'] = np.cumsum(changes['close'])

    # convert OHL to log returns based on previous close log returns
    changes[['open', 'high', 'low']] = changes[['open', 'high', 'low']].add(changes['close'].shift(1), axis = 0)

    # exp to recover total change (aka price)
    changes[price_features] = np.exp(changes[price_features])

    # ensure highs and lows are correctly higher or lower than close
    assert abs((changes['high'] - changes['close']).min()) == 0
    assert abs((changes['low'] - changes['close']).max()) == 0

    # set volume back to all positive
    changes['volume'] = abs(changes['volume'])

    # drop nans and reset index so that changes looks like original csv df
    changes = changes.dropna().reset_index()

    return changes


def make_dataset(filename, periods = [5, 10, 15, 20, 30, 45, 60, 90, 120],
                 leverage = 200, return_df = False, randomize = False,
                 bins = None, stds = None, squash_factor = 4,
                 features = ['open', 'high', 'low', 'close', 'volume']):
    fx = pd.read_csv(filename)

    if randomize:
        fx = randomize_csv(fx)

    # fix timezones, gaps, and interpolate missing data
    fx = preprocess(fx)

    # form labels and overnight masks for training purposes
    fx, future_cols, mask_cols = create_training_data(fx, periods, leverage)

    # remove means from price data & volume
    fx = demean(fx)

    # get rid of first day and last day due to data incompleteness
    fx = fx.drop(fx[fx['ordinal_day'] == fx['ordinal_day'].min()].index)
    fx = fx.drop(fx[fx['ordinal_day'] == fx['ordinal_day'].max()].index)
    
    if return_df:
        return fx

    # create labels column for classification
    assert bins is not None, "bins are needed to form dataset"
    labels = {}
    for col in future_cols:
        labels[col] = pd.cut(fx[col], bins[col], labels = False)

    labels_df = pd.DataFrame(labels).astype(int)
        
    # standardize data
    assert stds is not None, "stds are needed to form dataset"
    fx[features] = fx[features].div(stds, axis = 1)
    
    # squash data (to help limit OOD)
    fx[features] = squash_factor * np.tanh(fx[features] / squash_factor)

    ohlcv = fx[features].values.reshape(-1, 1440, len(features))
    masks = fx[mask_cols].values.reshape(-1, 1440, len(periods))
    future = fx[future_cols].values.reshape(-1, 1440, len(periods))
    classes = labels_df.values.reshape(-1, 1440, len(periods))
    
    ds = Dataset.from_dict({
        "ohlcv": ohlcv, "overnight_masks": masks,
        "labels": future, "classes": classes
    })

    return ds, stds


def get_standards(leverage, features, randomize = False):
    # just using EUR/USD for now for simplicity
    curr = make_dataset("Currency_Data/OANDA/EUR_USD.csv", leverage = leverage,
                        return_df = True, randomize = randomize)
    future_cols = curr.columns[curr.columns.str.contains('future')]

    bin_map = {}
    for col in future_cols:
        no_zeros = curr[col][curr[col] != 0]
        
        # get the number of cuts to make so that the middle bin only changes by .004 (i.e. commission)
        num_cuts = int(1 / (1 - (no_zeros.abs() > .004).mean()))

        # max the number of cuts at 64 for simplicity with CE loss
        num_cuts = min(num_cuts, 64)
        
        # make the number of cuts odd so that there is a middle bin
        if num_cuts % 2 == 0:
            num_cuts -= 1
        
        cut_array = pd.qcut(no_zeros, num_cuts, labels = False, retbins = True)[1]

        # expand the first and last bins to include all values
        cut_array[0] = -np.inf
        cut_array[-1] = np.inf

        bin_map[col] = cut_array
        
    stds = curr.iloc[:int(len(curr) * .9)][features].std(axis = 0)
    
    return bin_map, stds