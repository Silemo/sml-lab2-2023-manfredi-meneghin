# Scalable Machine Learning and Deep Learning, Lab2, 2023

## About
This repository is related to the **Lab 2** activities of the course [ID2223 Scalable Machine Learning and Deep Learning](https://www.kth.se/student/kurser/kurs/ID2223?l=en) 
at [KTH](https://www.kth.se). The tasks, 1 and 2 are [here](https://github.com/Silemo/sml-lab2-2023-manfredi-meneghin/blob/d8727107096f67d3887d0f905361c93237c2b249/id2223_kth_lab2_2023.pdf) described.

In **this repository** you can find:
- [**Fine Tuning of Whisper small model in Italian**](https://github.com/Silemo/sml-lab2-2023-manfredi-meneghin/tree/main/task1) Task 1 of the Lab 2
- [**Fine Tuning of Whisper small model in Italian, divided into two different pipelines**](https://github.com/Silemo/sml-lab2-2023-manfredi-meneghin/tree/main/task2) Task 2 of the Lab 2

*Task 1* consists in the adaptation of the ["Fine-Tune Whisper For Multilingual ASR with 🤗 Transformers"](https://huggingface.co/blog/fine-tune-whisper)
blog post by *sanchit-gandhi*, applying his approach to another language, the authors' mother tongue: **italian**.

Both projects consists of **Jupyter notebooks** designed to be run in **Google CoLab**. The final application is hosted on [HuggingFace Spaces](https://huggingface.co/spaces/Silemo/whisper-it).

### The Team

* [Giovanni Manfredi](https://github.com/Silemo)
* [Sebastiano Meneghin](https://github.com/SebastianoMeneghin)

## Table of Contents
* [Introduction](#Introduction)
* [Pipelines description](#Pipelines-description)
* [Results](#Results)
* [Improving model performance](#Improving-model-performance)
* [Software used](#Software-used)

## Introduction
The project can be divided into **three main pipelines**:
- **Data preparation and Feature engineering**: the features are taken from the [Common Voice 11.0](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0) dataset, and prepared to be used for training. The dataset was saved on Google Drive, that was used as Feature Store.
- **Training and evaluation**: the training is carried out on Google Colab using a free account. Checkpoints were used to overcome the limited availability of GPUs on Colab. Finally the model was uploaded to HuggingFace, that was used as model registry.
- **Inference program**: an application was created and hosted on HuggingFace spaces using Gradio. This application allows to use and see the model's functionalities.
  - 🗣️ [*ASR Italian - Whisper small model*](https://huggingface.co/spaces/Silemo/whisper-it)

## Pipelines description
### Data preparation and Feature engineering
Here's where the **data are taken from the [Common Voice 11.0](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0) dataset**. 
When getting the train, and test set **only 10%** of the original dataset is taken, due to to the limited space, time and computing capabilities of our setup.

The next step is to **remove the attributes** that are not useful for our training, namely *accent, age, client_id, down_votes, gender, locale, path, segment and up_votes*.

The following step is to **down-sample the audio input from 48 kHz to 16 kHz**, since 16 kHz is the sampling rate expected from the Whisper model. To do this we use the Whisper Feature Extractor.

After this data preparation, we **save the dataset on Google Drive** that we use a feature store.

**Note**: this pipeline can be run on a CPU to save GPU computing resources.
### Training and evaluation
Now that we've prepared our data, we're ready to dive into the training pipeline. The **HuggingFace Trainer** will do much of the heavy lifting for us. All we have to do is:
- **Define a data collator**: the data collator takes our pre-processed data and prepares PyTorch tensors ready for the model.
- **Evaluation metrics**: during evaluation, we want to evaluate the model using the word error rate (WER) metric. We need to define a compute_metrics function that handles this computation.
- **Load a pre-trained checkpoint**: we need to load a pre-trained checkpoint and configure it correctly for training.
- **Define the training configuration**: this will be used by the HuggingFace Trainer to define the training schedule.
Once we've fine-tuned the model, we will evaluate it on the test data to verify that we have correctly trained it to transcribe speech in Italian.

For the training we have set a series of hyper-parameters. This is the summary:
- learning_rate: 1e-05
- train_batch_size: 16
- eval_batch_size: 8
- training_steps: 4000
- gradient_accumulation_steps: 2
- save_steps: 100
- eval_steps: 100

The *save_steps* were size in such a way, to allow Google Colab to run for a while, but avoid the situation where the execution time available for the virtual machine would run out, without having saved anything.
The *gradient_accumulation_steps* were increased to 2 to fit the dimension of the GPU (**NVIDIA T4 Tensor Core with 15360 MiB of VRAM**).
Other values were left as in the original [blog post](https://huggingface.co/blog/fine-tune-whisper).

At the end of training the model is pushed to HuggingFace, that acts as a model registry.

**Note**: The notebook was run 15 times, with approximately 40 min for each 100 steps of training for a total of 26.5h of training.
Keep in mind that Google Colab was available to us for no more than 4 h a day, so around 7 days were necessary for training alone.

### Inference program
The **Gradio** application consists of two interfaces:
- **Audio or Microphone**: receives as input a recording or an audio file, and outputs a written transcription
- **YouTube**: receives as input a YouTube link and outputs a written transcription.
  
Additionally to the transcription, the application was connected to [*text2tags*](https://huggingface.co/efederici/text2tags) model that given a text provides tags for the text. 
This is also outputted additionally after the transcription.

## Results
The model achieves the following results:
- Loss: 0.4549
- Wer: 200.40

The value of the *Wer* is quite low. This could be caused mainly due to the limited size of the dataset we got, that is only 10% of the original dataset. See **Improving model performance** section for the full discussion.

The intermediate values are the following:

| Run Number    | Step         | Training Loss     | Validation Loss                | Wer                        |
|:-------------:|:------------:|:-----------------:|:------------------------------:|:--------------------------:|
| 1             | 100          | 1.2396            | 1.2330                         | 176.40                     |
| 2             | 200          | 0.7389            | 0.8331                         |  80.49                     |
| 2             | 300          | 0.2951            | 0.4261                         |  70.20                     |
| 2             | 400          | 0.2703            | 0.4051                         | 101.60                     |
| 3             | 500          | 0.2491            | 0.3923                         | 112.20                     |
| 3             | 600          | 0.1700            | 0.3860                         | 107.10                     |
| 3             | 700          | 0.1603            | 0.3836                         |  90.36                     |
| 4             | 800          | 0.1607            | 0.3786                         | 135.00                     |
| 4             | 900          | 0.1540            | 0.3783                         |  99.05                     |
| 4             | 1000         | 0.1562            | 0.3667                         |  98.32                     |
| 4             | 1100         | 0.0723            | 0.3757                         | 158.90                     |
| 5             | 1200         | 0.0769            | 0.3789                         | 215.20                     |
| 5             | 1300         | 0.0814            | 0.3779                         | 170.50                     |
| 5             | 1400         | 0.0786            | 0.3770                         | 140.60                     |
| 5             | 1500         | 0.0673            | 0.3777                         | 137.10                     |
| 6             | 1600         | 0.0339            | 0.3892                         | 166.50                     |
| 7             | 1700         | 0.0324            | 0.3963                         | 170.90                     |
| 7             | 1800         | 0.0348            | 0.4004                         | 163.40                     |
| 8             | 1900         | 0.0345            | 0.4016                         | 158.60                     |
| 8             | 2000         | 0.0346            | 0.4020                         | 176.10                     |
| 8             | 2100         | 0.0317            | 0.4001                         | 134.70                     |
| 9             | 2200         | 0.0173            | 0.4141                         | 189.30                     |
| 9             | 2300         | 0.0174            | 0.4106                         | 175.00                     |
| 9             | 2400         | 0.0165            | 0.4204                         | 179.60                     |
| 10            | 2500         | 0.0172            | 0.4185                         | 186.10                     |
| 10            | 2600         | 0.0142            | 0.4175                         | 181.10                     |
| 11            | 2700         | 0.0090            | 0.4325                         | 161.70                     |
| 11            | 2800         | 0.0069            | 0.4362                         | 161.20                     |
| 11            | 2900         | 0.0093            | 0.4342                         | 157.50                     |
| 12            | 3000         | 0.0076            | 0.4352                         | 154.50                     |
| 12            | 3100         | 0.0089            | 0.4394                         | 184.30                     |
| 13            | 3200         | 0.0063            | 0.4454                         | 166.00                     |
| 13            | 3300         | 0.0059            | 0.4476                         | 179.20                     |
| 13            | 3400         | 0.0058            | 0.4490                         | 189.60                     |
| 14            | 3500         | 0.0051            | 0.4502                         | 194.20                     |
| 14            | 3600         | 0.0064            | 0.4512                         | 187.40                     |
| 14            | 3700         | 0.0053            | 0.4520                         | 190.20                     |
| 14            | 3800         | 0.0049            | 0.4545                         | 194.90                     |
| 15            | 3900         | 0.0052            | 0.4546                         | 199.60                     |
| 15            | 4000         | 0.0054            | 0.4549                         | 200.40                     |

## Improving model performance
### Model centric approaches
A series of approaches is possible, let's go over some of these options:
- **Regularization**: implementing regularization techniques such as early stopping can prevent over-fitting.
- **Hyper-parameter tuning**: tuning hyper-parameters such as learning rate and batch size can improve the performance of the model.
- **Optimization Algorithms**: adding an optimizer, such as SGD can improve the overall performance.

### Data centric approaches
The most simple approach to improve the model's performance, would be to fine-tune the Whisper model on the **whole *common_voice_11_0* dataset** instead of the 10% of it. 
This would be possible only if more computing resources (or time) and storage space are provided. For example we could make use of Colab Pro for having access to more powerful GPUs
and for more time. This would allow the training algorithm to run smoothly even on more samples, instead of having to break the computation in a high number of different runs.
Keep in mind that the 10% of the dataset occupied 16 GB of Storage in Google Drive, so going for the full dataset, might also require additional Google Drive storage.

Another approach to improve the model's performance, would be to train the model on a **larger and more recent dataset**. An example is the more recent version of 
the *common_voice* dataset by Mozilla Foundation, i.e. *common_voice_15.0*. Note that even on more recent data, if the size of the dataset remains small, the results won't improve by much.

### Notes on optimizing time and resources
Other than improving the overall performance of the model, other techniques allow the computation to use less time or resources such as computing power or storage space. 
Here we can note that the requirement for this project, i.e. checkpointing, increases training time of about 20%. We also used Gradient accumulation that allowed us to
take advantage of the full GPU, without going out of bound. These are the techniques that were taken in to consideration in this project. More techniques are available [here](https://huggingface.co/docs/transformers/perf_train_gpu_one)

------------------------------------------------
### Software used

[**Visual Studio Code**](https://code.visualstudio.com/) - main IDE

[**GitKraken**](https://www.gitkraken.com/) - git versioning

[**Google Colab**](https://colab.research.google.com/) - running environment

[**HuggingFace**](https://huggingface.co/) - dataset, model registry, GUI

[**Gradio**](https://www.gradio.app/) - GUI