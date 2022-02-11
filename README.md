# Makaniino

## Description
*Makaniino* builds and trains a Machine-Learning Tropical Cyclone detection model.

### Disclaimer
This software is still under heavy development and not yet ready for operational use

### Installation

Please refer to the INSTALL file.

### User Guide
Makaniino features several functionalities, broadly grouped into 4 categories:

1. Data downloading
2. Data pre-processing
3. Model training
4. Model evaluation and analysis

#### 1. Data downloading
Data can be downloaded using the following command (--help for more details)
> mk-download <download-config-file.json>

The schema of the data-download config file can be shown as follows
> mk-format-download-config --show

#### 2. Data pre-processing

Similarly to data-download, the main data-download command is invoked as follows:
> mk-preprocess <preprocess-config-file.json>
 
The pre-processing configuration file schema can be shown as follows:
> mk-format-preprocess-config --show

#### 3. Model Training

To train a model the following components need to be selected and configured:

 - *model*: the actual machine-learning model to be trained
 - *data-handler*: component that handles the training data 
 - *trainer*: component that handles the training process

To list all the available concrete components, the following 
command can be used:
> mk-format-training-config --show

Then, to inspect the configuration parameters for a specific component:  
> mk-format-training-config --show-component=<component-name>
 
At this point, a complete training configuration can be assembled:
```
    "model": {
        "name": <model-name>,
        "params": {
         ...
         },
    "data_handler": {
        "name": <data-handler-name>,
        "params": {
            ...
        }
    },         
    "trainer": {
        "name": <trainer-name>,
        "params": {
        ...
        }
```
Finally, to train the model, simply run:
> mk-train <configuration-file.json>
 
Alternatively, the user can write a file "makaniino-components.json" in the working
directory, where the desired components are listed, e.g.:
```
{
 "model": <model-name>,
 "data_handler": <data-handler-name>,
 "trainer": <trainer-name>
}
```
By invoking the tool *mk-train-from-args*, any available option (from each
component selected) can be overridden by CL arguments:
> mk-train-from-args
> --tag=&lt;simulation-tag&gt;
> --&lt;option-1&gt;=&lt;user-choice-1&gt; ...


#### 5. Workflow Examples
The folder "examples" contains several scripts that illustrate how 
to use the tools described in the sections above. Before running the 
scripts, make sure the conda env is invoked
> conda activate makaniino_env

 - 1_download.sh: Download data using *Climetlab*
 - 2_preprocess.sh: Run the default TC pre-processing
 - 3_augment.sh: Augment data by extracting crops of global fields
 - 4_train.sh: Training a TC model on augmented data
 - 5_predict.sh: Configures a TC model that reads full-fields data 
                 but uses the weights of the trained TC model

In the "examples" folder there are also scripts showing some 
functionalities for evaluating and analysing the ML model.
 
 - 6_check_dataset.sh: check dataset
 - 7_check_model.sh: check model
 - 8_evaluate_model.sh: evaluate model
 - 9_keras_model.sh: Save the model in Keras format

Finally, the following script cleans the "examples" folder itself
 - clean_examples.sh: Clean examples folder

NOTE: by default all the examples output will be placed into 
the current working directory.
 
