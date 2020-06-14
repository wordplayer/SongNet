# SongNet
Realizing "What if machines could just hear and predict what my song sounded like?"

---From the hearts, efforts and fingers of Aisha Dantuluri, Tyler Farnan and Soumyaraj Bose (Group 51) and the mind of Robert D. Smith (https://github.com/teticio/Deej-A.I.)

Install dependencies (if not present): `pip install pandas tensorflow tensorflow-probability matplotlib seaborn`

Make sure that your version of `tensorflow` is >=2.1.0. Before running any of the training notebooks, make sure to change the kernel to the one on which `tensorflow` is installed.

Folder **Code Notebooks 0.1**: Contains notebooks for running all of the processes involved in this project

`MakeSongNet.ipynb`: Extracts songs from the file of playlists obtained from the Spotify API and creates a `.txt` file that contains a network where song IDs serve as nodes

`01_Spectrogram_Preprocessing.ipynb`: Reads the .png files of the Mel Spectrograms from the dataset used by DEEJ-A.I. and converts them into `numpy` arrays for training and analysis. Stores the `numpy` arrays in separate `.txt` files each for the training and validation set

`02_AE_dev_1.ipynb`: Template for running all the deep learning models; same template has been used and modified for the five models (A01, A02-04, B - VAE). It runs all the functions from scaling and normalizing to training the model and displaying the latent representations.

`03_Community_Detection.ipynb`: Contains the functionality from `networkx` to generate the song network; works as a template for `MakeSongNet.ipynb`

`04_Cluster_vs_Community_Analysis.ipynb`: Contains training for first model (Model A0) and all functionality for optimal cluster generation and cluster and community analysis; Works as a template for all other model notebooks

`Model Selection.ipynb`: Contains functionality involving training, validation and cluster and community analysis for all convolutional AE models A01-03 (here, A-C). The cluster and community analysis for these models is also contained in notebooks in the folder **Model Clustering Notebooks**. The model weights and the histories are stored in the folders **Model Weights** and **history pickles** respectively. To reuse a model from **Model Weights**, use `<network object>.load_weights(os.getcwd() + 'Model Weights\model_name')`

`VAE_08.ipynb`: Contains all functionality and model for the variational autoencoder; its model is stored in model_085. Reuse the VAE model using the same highlighted functionality as above
