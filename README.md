# Comparing Pooled Multinomial and Dawid-Skene
## Human-in-the-Loop Machine Learning Project, MSc Artificial Intelligence, University of Amsterdam, 2023-2024 academic year


This repository contains implementations of the Pooled Multinomial and Dawid-Skene algorithms. It also contains data processing for [the Wikipedia Aggression dataset](https://figshare.com/articles/dataset/Wikipedia_Talk_Labels_Aggression/4267550).

To run the code:
```
python data_loading.py --datapoints 10000 --iterations 3 --algo mn --parse_data
```

The `algo` argument can either be `mn` or `ds` for the two algorithms. Include the `parse_data` argument to iterate through the dataset and generate the ground truths using majority voting and Dawid-Skene.
