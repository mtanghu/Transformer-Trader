# Transformer Trading
This project tries the simple idea of applying Transformers and self-supervised learning to financial markets end-to-end, particularly the price data of Forex markets on mid-terms time scales of minutes to hours. This README will describe at a high level how this project handled risk optimization, commissions/spread, and the general model pipeline. Also discussed are issues with many popular Forex data providers, scaling properties, and profitability.

### Why Forex?
1. Forex trades 24/7 and thus has 3x more data compared to stocks, futures, and options that tend to only trade 9-5
2. In general forex tends to offer leverage which can increase the profitability on low risk strategies, especially in small-scale settings where access to leverage and subsequent liquidity for the leverage is abundant

### Why mid-term time scales?
Time scales less than minutes would be closer to the realm of High Frequency Trading which would be unsuitable for a small scale project that already has trading latency on the order of seconds.

Time scales greater than hours and into days would run into issue that most Forex brokers charge for overnight positions. Also these longer type trades may have more to do with macro-economic trends that this would be difficult to have high quality data for.

## Self-supervised learning of price data
In the context of price data, self-supervised learning refers to simply using price data as the input to the model, and also using price data to form labels. Luckily this can be combined with another paradigm of end-to-end learning where instead of trying to predict/generate future price data, we can instead have the model produce "trades" (i.e. whether to buy or sell and how much) and use how much money the trade made or loss (i.e. profitability) to calculate loss.

### Incorporating risk

A naive approach would be to simply maximize profitability by calculating the theoretical profits and losses. However this would both ignore risk and also run into a related issue that models trained to maximize profitability would not be able to "size trades" and thus not be able to calculate appropriate levels of leverage. This can be observed simply by noticing that any trade that has a positive expected value of *x* will have a leveraged expected value of *l\*x* where *l* is the leverage multiplier, and thus maximizing the expected value of leveraged trades would simply involve setting the leverage as high as possible.

This project instead opts to maximize *growth* of possible trades which is  theoretical similar to the [Kelly Criterion](https://en.wikipedia.org/wiki/Kelly_criterion). Mathematically this can be achieved my minimizing $-ln(1+p-s*c)$ where *p* is the profit of the trade as measured by percent gained or lost, *s* is the size of the trade as measured by percent of maximum leveraged capital staked , and *c* is the percent commission + approximate spread per unit of size (this project usually assumed a .002% commission and around .005% commission based on EURUSD trading costs of popular online brokers). The reasons and motivations are too low level to be discussed here, but in general this encodes "risk aversion" while being grounded in growth theory.

This can simply be used as the loss function for training neural networks (though as side note an auxilliary classification loss was also used in this project to help with underfitting). The only caveat is that because we're considering a leveraged context where losses can exceed 100% (which would cause a logarithm of negative number) losses are capped at 50% where any losses greater 50% are set to 50% while preserving the gradient. This is similar to loss clipping which has had success in [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347).

## Pipeline
### Data
This project focused on minute level OHLCV data (since that seemed to be the only widely available data even among paid services) where OHLCV stands for Open, High, Low, Close, and Volume. 87 currency pairs were considered (including some precious metals) totaling some ~.5 billion minutes worth of data.

Using pandas some key steps were taken to get the data ready for modeling:
- Interpolate any gaps (where no price data was reported for the minute) as most data providers would not report minutes where no volume/trades occurred
- Remove effects of USA daylight savings time (which would change volume)
- Normalize the data (using primarily percent change from previous minute calculations) and  apply a transformation of $4*tanh(x/4)$ to squash values between \[-4, 4\] both with the intention of decreasing the out-of-distribution (OOD) inputs

These were all packaged into convenient [HuggingFace Datasets](https://huggingface.co/docs/datasets/index) for easy loading and storing as well as an easy interface with training. The pipeline with pandas and huggingface was set up since most data providers would provide CSV type data both historically and in real-time and thus the preprocessing code wouldn't need to be adapted for real-time data in production mode.

### Modeling
To form embeddings a causal version of a convolution was applied to the OHLCV data, then these embedding were passed to a basic Transformer with a custom loss.

Many architectural variants were experimented with for various reasons:
- Linear Attention methods like [MEGA](https://arxiv.org/abs/2209.10655), [SGConv](https://arxiv.org/abs/2210.09298), [SRU++](https://arxiv.org/abs/2110.05571) were tried since the sequence length was relatively high (1440 = # minutes in a day) so that training and experimentation could happen quickly
- [FlashAttention](https://arxiv.org/abs/2205.14135) ended up being faster with less memory and had more well researched scaling properties as it's exact attention
- [SwiGLU](https://arxiv.org/abs/2002.05202v1) seemed to improve performance in a variety of settings
- Getting rid of linear biases for speed as suggest by [Geiping & Goldstein (2022)](https://arxiv.org/abs/2002.05202v1)
- Varying whether to use an embedding LayerNorm and/or a LayerNorm right before prediction which seemed to affect performance strongly ([RMSNorm](https://arxiv.org/abs/1910.07467) was also tried)
- Trying a number of novel ways to calculate a continuous output for the "size" of a trade (as opposed to just using a tanh gated output) that wouldn't necessarily have high gradient around 0
- [Rotary embeddings](https://arxiv.org/abs/2104.09864) for performance
- Robust classification losses like Lq loss from [Zhang & Sabuncu (2018)](https://arxiv.org/pdf/1805.07836.pdf) as well as Active Passive Loss from [Ma & Huang et al. (2020)](https://arxiv.org/abs/2006.13554) to help with the noisy nature of financial data and labels

fp16 mixed precision training was also used to speed up training

As a more general note on training, the hyperparameter tuning aspect of this project proved to be a significant challenge. The performance/loss of the models seemed heavily dependent on learning rate and placement of LayerNorms with no particular patterns as to the optimal settings. Also dropout and weight decay didn't seem to improve overfitting.

### Model Evaluation
The models were validated and tested on "future data" where the training data would all be historically before any validation data (trying to train the model forward in time was also an idea, but seemed to generally hurt model performance compared to more standard random sampling).

Standard metrics like loss, sharpe, profitability of trades, and average profits were measured, as well as more fine grained analysis of profitability across the trading day and long vs short preferences.

## Results

### Data provider issues
This was a key challenge point of this project given that high quality data can be very hard to come by, and diagnosing that a given dataset has issues isn't well defined. Listed are some data providers tried and this project's experience with them:
- Polygon, Alphavantage, and Twelvedata all had issues with basic data quality as their data would often not match up with larger data providers data that would match up (like Oanda, Forex.com, and broker data) on a minute to minute basis
- Tradermade only offered their API with a limited amount of *total* API requests which would be prohibitively costly for both historical downloads and real-time streaming
- broker data and Forex.com data wouldn't go far back enough historically (only went back less than a year)
- Firstratedata offered a simple historic download, however after some extensive testing it was found that for some reason the Firstratedata was *highly* predictable and thus models could achiever *extremely high levels of profit* that would not translate to other larger data providers (like Oanda, Forex.com, and broker data) even when finetuning approaches were used
- TrueFX offered tick level data though only to about a year backwards
- Oanda had the unfortunate issue of only reporting bid price data where bid prices naturally decrease in low volume periods of the day due to spread widening, but this kind of event is completely unprofitable

Overall, a recommendation would be to use tick-level data that potentially has bid and ask level data, unfortunately this kind of data may be very hard to come by without special resources.

### Scaling
This project explored the possibility of scaling to greatly improve performance of models following from neural scaling laws research that has been shown effective in a variety of contexts from originally text ([Kaplan & McCandlish et al. 2020](https://arxiv.org/pdf/2001.08361.pdf)), then in a variety of other generative contexts ([Henighan, Kaplan, Katz, et al. , 2020](https://arxiv.org/pdf/2001.08361.pdf)), and even in reinforcement learning ([Neumann & Gros 2023](https://arxiv.org/pdf/2210.00849.pdf)). However, strange and unintuitive results would happen with scaling up both in terms of data quantity and model sizes.

They can be summarized as follows:
- Using more data would decrease training loss but not validation loss regardless of model size (for a variety of metrics and architectures and hyperparameters)
- Larger models would not perform better than smaller models regardless of varying data size and would often diverge

Generally speaking this would imply that scaling laws were not applying to this specific context.

### Profitability
Unfortunately after factoring in commission and removing the effects of using only bid level data, profitable strategies were not particularly found. There may be some hope for using similar ideas for market making (where commission isn't an issue) or for liquidity taking strategies, however those are outside the scope of this project.
