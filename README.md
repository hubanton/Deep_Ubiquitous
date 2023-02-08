# Deep Ubiquitous and Wearable Computing - Team DAW

## Setup
- Make sure to have all required packages installed

- If you don't have the dataset, you can download it from <a href="https://archive.ics.uci.edu/ml/datasets/Activity+recognition+with+healthy+older+people+using+a+batteryless+wearable+sensor">here</a>

## Preliminary Models
- To run an initial benchmark on the dataset, simply run the 'preliminary_results.py'
- This wil store the basic classification results of various models into a csv

## Deep Learning Model
- To perform the entire tuning process simply run 'main.py'
- This will test all combinations we used for our project
- Note that creating the windows is quite slow so this will take a while

## Other scripts
- data_loading: fetches and preprocesses data
- tuning_results: visualize results from the tuning process (in main.py)
- visualize_results: plot metrics gained from preliminary models
- rnn: the modifiable RNN 
- train_and_evaluate: training loop used for all experiments