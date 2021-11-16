# Music Conformer

~["model"](./img/model_architecture.png)

Currently supports Pytorch >= 1.2.0 with Python >= 3.6  

## TODO
* Write own midi pre-processor (sustain pedal errors with jason's)
   * Support any midi file beyond Maestro
* Fixed length song generation
* Midi augmentations from paper
* Multi-GPU support

## How to run
1. Download the Maestro dataset (we used v2 but v1 should work as well). You can download the dataset [here](https://magenta.tensorflow.org/datasets/maestro). You only need the MIDI version if you're tight on space. 

2. Run `git submodule update --init --recursive` to get the MIDI pre-processor provided by jason9693 et al. (https://github.com/jason9693/midi-neural-processor), which is used to convert the MIDI file into discrete ordered message types for training and evaluating. 

3. Run `preprocess_midi.py -output_dir <path_to_save_output> <path_to_maestro_data>`, or run with `--help` for details. This will write pre-processed data into folder split into `train`, `val`, and `test` as per Maestro's recommendation.

4. To train a model, run `train.py`. Use `--help` to see the tweakable parameters. See the results section for details on model performance. 

5. After training models, you can evaluate them with `evaluate.py` and generate a MIDI piece with `generate.py`. To graph and compare results visually, use `graph_results.py`.

For the most part, you can just leave most arguments at their default values. If you are using a different dataset location or other such things, you will need to specify that in the arguments. Beyond that, the average user does not have to worry about most of the arguments.

### Training
As an example to train a model using the parameters specified in results:

```
python train.py -output_dir ourput
```
You can additonally specify both a weight and print modulus that determine what epochs to save weights and what batches to print. The weights that achieved the best loss and the best accuracy (separate) are always stored in results, regardless of weight modulus input.

### Evaluation
You can evaluate a model using;
```
python evaluate.py -model_weights rpr/results/best_acc_weights.pickle 
```

Your model's results may vary because a random sequence start position is chosen for each evaluation piece. This may be changed in the future.

### Generation
You can generate a piece with a trained model by using:
```
python generate.py -output_dir output -model_weights rpr/results/best_acc_weights.pickle 
```

The default generation method is a sampled probability distribution with the softmaxed output as the weights. You can also use beam search but this simply does not work well and is not recommended.

## Results
We trained a model with the following parameters for 300 epochs:
* **learn_rate**: None
* **ce_smoothing**: None
* **batch_size**: 2
* **max_sequence**: 2048
* **n_layers**: 6
* **num_heads**: 8
* **d_model**: 256
* **dim_feedforward**: 1024
* **dropout**: 0.1

The following graphs were generated with the command: 
```
python graph_results.py -input_dirs /results -model_names model
```
