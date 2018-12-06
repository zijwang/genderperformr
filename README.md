# GenderPerformr

## Intro
GenderPerformr is the model release from the paper `It’s going to be okay: Measuring Access to Support in Online Communities` by Zijian Wang and David Jurgens (in proceedings of EMNLP 2018).

It is the current state-of-the-art method that predicts gender from usernames based on a LSTM model built in PyTorch (as of Sept. 2018).

See the [project website](http://blablablab.si.umich.edu/projects/support) for full details, including contact information.
 
## Install 

### Use pip
If `pip` is installed, genderperformr could be installed directly from it:

	pip install genderperformr
### From raw
`git clone` the project and do:

	python setup.py install

### Dependencies
	python>=3.6.0
	torch>=0.4.1
	numpy
	unidecode


## Usage and Example

### `predict`
`predict` is the core method of this package, 
which takes a single username of a list of usernames, and returns a tuple of raw probabilities in `[0,1]` (0 - Male, 1 - Female), and labels (M - Male, N - Neutral, F - Female, empty string - others). 

### Simplest usage

You may directly import `genderperformr` and use the default predict method, e.g.:

    >>> import genderperformr
    >>> genderperformr.predict("AdamMcAdamson")
    (0.019139649, 'M')
    
### Construct from class
Alternatively, you may also construct the object from class, where you could customize the model path and device:
 
	>>> from genderperformr import GenderPerformr
	>>> gp = GenderPerformr()
	
	# Predict a single username
	>>> gp.predict("John")
	(0.087956183, 'M')
	
	# Predict a list of names
	>>> probs, labels = gp.predict(["BarryCA67", "pizzamagic", "KatieZ22"])
    >>> f"Raw probabilities are {probs}"
    Raw probabilities are [0.03398224 0.5439474 0.93964571]
    >>> f"Labels are {labels}"
    Labels are ['M', 'N', 'F']


More detail on how to construct the object is available in docstrings.

### Model using new data partition 
If you want to use the model described in Supplemental Material using the new data partition, you may construct the object via

    >>> gp = GenderPerformr(is_new_model=True)

All other usages remain the same.


## Citation
    @inproceedings{wang2018its,
           title={It's going to be okay: Measuring Access to Support in Online Communities},
           author={Wang, Zijian and Jurgens, David},
           booktitle={Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP)},
           year={2018}
    }
    
## Contact
Zijian Wang (zij<last_name>@stanford.edu)

David Jurgens (<last_name>@umich.edu)
