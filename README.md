### 1. Data Acquisition and Cleaning
- **Download the Dataset**: Use `kagglehub` to download the "glasses-or-no-glasses" dataset from Kaggle.
- **Identify and Correct Labels**: Visualize the images alongside their labels to identify any that are mislabelled. You must correct the dataset as necessary. While `matplotlib` or `cv2` are commonly used for this, they are not strictly required.
### 2. Preprocessing
- **Resize Images**: Use `cv2.resize` to downscale all images from their original $1024 \times 1024$ resolution to $64 \times 64$.
- **Note**: You may adjust this resolution further depending on your available GPU resources.
### 3. Model Implementation

Implement one model for each of the following three classes for class-specific generation:
- **Variational Auto-Encoder (VAE)**
- **Generative Adversarial Networks (GAN)** 
- **Diffusion Models** 

**Goal**: Use these models to generate a total of **six images** for your submission: three people with glasses and three people without glasses.

### 4. Hyperparameter Ablation Study

Create an **ablation table** documenting at least **five changes** to hyperparameters for each model. You should choose parameters you understand, as you will be asked about them during your presentation.

Suggested hyperparameters include:

- **VAE**: Latent vector $z$ size, convolution filter sizes ($3, 5, 7$), or activation functions.
- **GAN**: Ratio of generator vs. discriminator steps, dropout rates, or minibatch configurations.
- **Diffusion**: Number of diffusion steps ($\beta_{t}$), learning rate, or batch size.
- **Common**: The number of layers is a valid hyperparameter for all models.

---
### Important Notes

- **Performance**: As long as at least one of your models generates images well, differences in performance between models will not be a major consideration.
- **Intuition**: Ensure your hyperparameter changes are intuitive (e.g., do not increase dropout if it consistently increases your loss metrics).
- **Contact**: For clarifications, contact Hans via Gmail Chat at p20252203@hyderabad.bitspilani.ac.in`.
