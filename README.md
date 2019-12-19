# Probabilistic Parsing of Phone Numbers

## Installation

1. Install Python 3.7 with Numpy, Pandas
2. Install [pytorch](https://pytorch.org/)
3. Install [pyprob](https://github.com/pyprob/pyprob)
4. Install [pyro](http://pyro.ai/)

## Running the Code

- Phone Number Parser (Works): `python infcomp_test.py --model_path=nn_model/phone_parser --number="<YOUR NUMBER>"`
- Pyro Phone Number Parser (Doesn't Work Well): `python infcomp_pyro_test.py <PATH TO CONFIG>`

### Sample Output

```sh
$ python infcomp_test.py --model_path=nn_model/phone_parser --number="+1 604 999 5585"
Warning: Empirical distributions on disk may perform slow because GNU DBM is not available. Please install and configure gdbm library for Python for better speed.
========================================================
Phone Number to Parse: +1 604 999 5585
Number of Traces / Samples: 10, 10
Model Path: nn_model/phone_parser
========================================================
Warning: different PyTorch versions (loaded network: 1.3.0, current system: 1.2.0)
Time spent  | Time remain.| Progress             | Trace | Traces/sec
0d:00:00:00 | 0d:00:00:00 | #################### | 10/10 | 26.73
('+1 604 999 5585               ', {'country': 'USA', 'country code': '1', 'area code': '604', 'line number': '999 5585'})
('+1 604 999 5585               ', {'country': 'USA', 'country code': '1', 'area code': '604', 'line number': '999 5585'})
('+1 604 999 5585               ', {'country': 'USA', 'country code': '1', 'area code': '604', 'line number': '999 5585'})
('+1 604 999 5585               ', {'country': 'Canada', 'country code': '1', 'area code': '604', 'line number': '999 5585'})
('+1 604 999 5585               ', {'country': 'Canada', 'country code': '1', 'area code': '604', 'line number': '999 5585'})
('+1 604 999 5585               ', {'country': 'Canada', 'country code': '1', 'area code': '604', 'line number': '999 5585'})
('+1 604 999 5585               ', {'country': 'Canada', 'country code': '1', 'area code': '604', 'line number': '999 5585'})
('+1 604 999 5585               ', {'country': 'Canada', 'country code': '1', 'area code': '604', 'line number': '999 5585'})
('+1 604 999 5585               ', {'country': 'USA', 'country code': '1', 'area code': '604', 'line number': '999 5585'})
('+1 604 999 5585               ', {'country': 'USA', 'country code': '1', 'area code': '604', 'line number': '999 5585'})
```

## Extra Links

- Regex Approach: https://github.com/daviddrysdale/python-phonenumbers
