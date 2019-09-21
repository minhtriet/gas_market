# Open-domain event extraction and embedding for Natural gas market prediction

## Abstract
We propose an approach to predict the natural gas price in several days using historical price data and events from news headlines. Our event extraction method detects not only the occurrence of phenomenons but also the changes in attribution and characteristics. It also serves as one of the preliminaries for a knowledge graph for real-time events. Instead of using sentences embedding as a feature, we use every word of the extracted events, encode and organize them before feeding to the learning models. Empirical results show favorable results, in term of prediction performance, money saved and scalability.

## Citation

## Installation
### Input
1. News: A CSV file with the following format, put it in 

| date       | price  |
|------------|--------|
| 02.07.2007 | 18.700 |
| 03.07.2007 | 19.510 |
| 04.07.2007 | 19.150 |
| 05.07.2007 | 21.700 |

2. Price: A CSV file with the following format

| pub_date          | info                                                       |
|-------------------|------------------------------------------------------------|
| 2011-09-05T00:00Z | China Challenges U.S. Supremacy in Shale Gas               |
| 2011-09-07T00:00Z | Flow Starts in Gas Pipeline From Russia to Germany         |
| 2011-09-08T00:00Z | European Union Seeks Power to Block Bilateral Energy Deals |
| 2011-09-14T00:00Z | Gas Flaring in North Dakota                                |

### Train
`python train_event.py`

### Inference


### Excute the strategy
`python strategy_excution_event.py --from_day 2018-01-01 --to_day 2018-10-01`
