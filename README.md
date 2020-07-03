# Open-domain event extraction and embedding for Natural gas market prediction

Our paper is accepted to CEUR Workshop, 2020 (http://ceur-ws.org/Vol-2611/paper2.pdf)

## Abstract
We propose an approach to predict the natural gas price in several days using historical price data and events from news headlines. Our event extraction method detects not only the occurrence of phenomenons but also the changes in attribution and characteristics. It also serves as one of the preliminaries for a knowledge graph for real-time events. Instead of using sentences embedding as a feature, we use every word of the extracted events, encode and organize them before feeding to the learning models. Empirical results show favorable results, in term of prediction performance, money saved and scalability.

## Citation

## Installation
We use `virtualenv` as the package management
1. Clone the repository
2. Install Python3
3. In the folder directory, run `python3 -m venv venv` to create a virtual environment
4. Run `source venv/bin/activate`
5. Run `pip install -r requirements.txt`
6. Install Spacy's English models
7. Download news headlines
### Input
1. News: A CSV file with the following format, put it in input folder. The file used to read this is `read_spot_market_v2` in `util.py`

| date       | price  |
|------------|--------|
| 02.07.2007 | 18.700 |
| 03.07.2007 | 19.510 |
| 04.07.2007 | 19.150 |
| 05.07.2007 | 21.700 |

2. Price: A CSV file with the following format. Place it in `old_log` folder

| pub_date          | info                                                       |
|-------------------|------------------------------------------------------------|
| 2011-09-05T00:00Z | China Challenges U.S. Supremacy in Shale Gas               |
| 2011-09-07T00:00Z | Flow Starts in Gas Pipeline From Russia to Germany         |
| 2011-09-08T00:00Z | European Union Seeks Power to Block Bilateral Energy Deals |
| 2011-09-14T00:00Z | Gas Flaring in North Dakota                                |

### Train
`python3 train_event.py`

### Inference
`python3 strategy_predict.py [--from_day from] [--to_day to]`

Arguments
- `--from_day from`
The starting day of the series (YYYY-MM-DD format)
- `--to_day to`
(YYYY-MM-DD format)
The ending day of the series (YYYY-MM-DD format)

### Execute / mock trading
`python3 strategy_excution_event.py [--from_day from] [--to_day to]`

Arguments
- `--from_day from`
The starting day of the series (YYYY-MM-DD format)
- `--to_day to`
(YYYY-MM-DD format)
The ending day of the series (YYYY-MM-DD format)

### Generate Venn Graphs
Note that Reverb needs an period at the end of each headlines to extract relation from them
- To get sentences that have Verbs
```cat real_reverb_result.txt | cut -f2 | uniq > reverb_indicies.txt```
- For our pipelines
```./testPlain.bash models-MUN-SC-wn30 test.txt outputFile lib/dict/index.sense```
