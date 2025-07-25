# HR 08/04/25 Toy GAN for testing, training and development
# Adapted from here: https://ricci-colasanti.github.io/Synthetic-Population-Generation/GAN.html

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from torch.utils.data import DataLoader, TensorDataset
import os
from os.path import dirname as up

DATA_DIR = os.path.join(up(__file__), 'data')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


def list_to_bins(_list):
    """Takes list and returns positions as ranges"""
    right = list(np.cumsum(_list))
    left = [0] + list(right[:-1])
    return [range(start, end) for start, end in zip(left, right)]


def lol_to_bins(lol):
    """Takes list of lists (lol) of categoricals and returns positions as ranges"""
    right = list(np.cumsum([len(el) for el in lol]))
    left = [0] + list(right[:-1])
    return [range(start, end) for start, end in zip(left, right)]


class TabularGenerator(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dim=128):
        super(TabularGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, output_dim),
            nn.Tanh()  # Using tanh to output in [-1, 1] range
        )

    def forward(self, z):
        return self.model(z)


class TabularDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(TabularDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class TabularGAN:
    def __init__(self, latent_dim=64, hidden_dim=128):
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.generator = None
        self.discriminator = None
        self.scaler = None
        self.ord = None
        self.encoder = None
        self.numerical_cols = None
        self.ordinal_cols = None
        self.categorical_cols = None
        self.encoded_categories = None
        self.feature_indices = []


    def preprocess_data(self, data, numerical_cols=None, ordinal_cols=None, categorical_cols=None):
        """Preprocesses the data by scaling numerical and encoding categorical features"""
        if numerical_cols is None:
            numerical_cols = []
        if ordinal_cols is None:
            ordinal_cols = []
        if categorical_cols is None:
            categorical_cols = []

        self.numerical_cols = numerical_cols
        self.ordinal_cols = ordinal_cols
        self.categorical_cols = categorical_cols

        # Process numerical features
        if self.numerical_cols:
            self.scaler = MinMaxScaler(feature_range=(-1, 1))
            numerical_scaled = self.scaler.fit_transform(data[self.numerical_cols])
        else:
            numerical_scaled = np.empty((len(data), 0))
        self.feature_indices.append(numerical_scaled.shape[1])

        # Process ordinal features
        if self.ordinal_cols:
            self.ord = OrdinalEncoder()
            ordinal_encoded = self.ord.fit_transform(data[self.ordinal_cols])
        else:
            ordinal_encoded = np.empty((len(data), 0))
        self.feature_indices.append(ordinal_encoded.shape[1])

        # Process categorical features
        if self.categorical_cols:
            self.encoder = OneHotEncoder(sparse_output=False)
            categorical_encoded = self.encoder.fit_transform(data[self.categorical_cols])
            self.encoded_categories = self.encoder.categories_
        else:
            categorical_encoded = np.empty((len(data), 0))
        self.feature_indices.append(categorical_encoded.shape[1])

        # Combine features
        processed_data = np.concatenate([numerical_scaled, ordinal_encoded, categorical_encoded], axis=1)
        self.feature_indices = list_to_bins(self.feature_indices)
        return processed_data.astype(np.float32)

    def postprocess_data(self, generated_data):
        """Converts generated data back to original format"""
        df = pd.DataFrame()

        if self.numerical_cols:
            numerical_data = generated_data[:, self.feature_indices[0]]

            # Inverse transform numerical
            numerical_data = self.scaler.inverse_transform(numerical_data)
            df = pd.concat([df, pd.DataFrame(numerical_data, columns=self.numerical_cols)], axis=1)

        if self.ordinal_cols:
            ordinal_data = generated_data[:, self.feature_indices[1]]
            print(ordinal_data)

            # Inverse transform ordinal
            ordinal_data = self.ord.inverse_transform(ordinal_data)
            print(ordinal_data)
            df = pd.concat([df, pd.DataFrame(ordinal_data, columns=self.ordinal_cols)], axis=1)

        if self.categorical_cols:
            categorical_data = generated_data[:, self.feature_indices[2]]

            # Inverse transform categorical
            bins = lol_to_bins(self.encoded_categories)
            for i, col in enumerate(self.categorical_cols):
                maxes = np.argmax(categorical_data[:, bins[i]], axis=1)
                categorical_data_inverse = [self.encoded_categories[i][j] for j in maxes]
                df[col] = categorical_data_inverse

        return df

    def train(self, data, numerical_cols=None, ordinal_cols=None, categorical_cols=None, epochs=1000, batch_size=32, lr=0.0002):
        """Train the GAN on tabular data"""
        # Preprocess data
        processed_data = self.preprocess_data(data, numerical_cols, ordinal_cols, categorical_cols)
        input_dim = processed_data.shape[1]

        # Initialize models
        self.generator = TabularGenerator(self.latent_dim, input_dim, self.hidden_dim)
        self.discriminator = TabularDiscriminator(input_dim, self.hidden_dim)

        # Optimizers
        optimizer_G = optim.Adam(self.generator.parameters(), lr=lr)
        optimizer_D = optim.Adam(self.discriminator.parameters(), lr=lr)
        criterion = nn.BCELoss()

        # Create dataloader
        dataset = TensorDataset(torch.from_numpy(processed_data))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Training loop
        for epoch in range(epochs):
            for i, real_data in enumerate(dataloader):
                real_data = real_data[0]
                batch_size = real_data.size(0)

                # Adversarial ground truths
                valid = torch.ones(batch_size, 1)
                fake = torch.zeros(batch_size, 1)

                # ---------------------
                #  Train Discriminator
                # ---------------------
                optimizer_D.zero_grad()

                # Real data
                real_loss = criterion(self.discriminator(real_data), valid)

                # Fake data
                z = torch.randn(batch_size, self.latent_dim)
                fake_data = self.generator(z)
                fake_loss = criterion(self.discriminator(fake_data.detach()), fake)

                # Total loss
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                optimizer_D.step()

                # -----------------
                #  Train Generator
                # -----------------
                optimizer_G.zero_grad()

                # Generate fake data
                z = torch.randn(batch_size, self.latent_dim)
                gen_data = self.generator(z)

                # Generator wants discriminator to think fake data is real
                g_loss = criterion(self.discriminator(gen_data), valid)
                g_loss.backward()
                optimizer_G.step()

            # Print progress
            if epoch % 10 == 0:
                print(f"[Epoch {epoch}/{epochs}] D loss: {d_loss.item():.4f}, G loss: {g_loss.item():.4f}", end='\r')

    def generate_samples(self, n_samples):
        """Generate synthetic samples"""
        if not self.generator:
            raise ValueError("Model not trained yet. Call train() first.")

        z = torch.randn(n_samples, self.latent_dim)
        with torch.no_grad():
            generated_data = self.generator(z).numpy()

        return self.postprocess_data(generated_data)


if __name__ == "__main__":

    # EXAMPLE 1: Random data
    # size = 100
    # data = pd.DataFrame({
    #     'age': np.random.normal(40, 15, size),
    #     'income': np.random.lognormal(8, 0.5, size),
    #     'nkids': np.random.randint(0, 12, size),
    #     'sex': np.random.choice(['M', 'F'], size),
    #     'education': np.random.choice(['None', 'GCSE', 'NVQ2+', 'Degree', 'Higher Degree', 'Apprenticeship', 'Space warrior', 'Wizard', 'Intergalactic trader', 'Chinchilla tickler'], size),
    # })
    #
    # cats = ['gender', 'education', 'nkids']
    # ords = []
    # # cats = ['gender', 'education']
    # # ords = ['nkids']
    # nums = [el for el in data.columns if el not in cats + ords]

    # EXAMPLE 2: Minos fertility data, pared down
    frac = 0.001
    data_raw = pd.read_csv(os.path.join(up(__file__), 'data', '2019_US_cohort.csv'))
    data = data_raw.sample(frac=1).reset_index(drop=True).sample(frac=frac).reset_index(drop=True)
    data['income'] = np.random.lognormal(8, 0.5, len(data))
    print('Number of individuals:', len(data))
    cats = ['sex', 'region', 'ethnicity', 'education_state']
    nums = ['age', 'income', 'nkids_ind', 'nresp']
    ords = []

    # Initialize and train GAN
    gan = TabularGAN(latent_dim=64, hidden_dim=128)
    # gan.train(data, numerical_cols=nums, categorical_cols=cats, epochs=1000, batch_size=32)
    gan.train(data, numerical_cols=nums, ordinal_cols=ords, categorical_cols=cats, epochs=100, batch_size=32)

    # Generate/cache populations for testing
    for X in (3, 4, 5, 6):
        print('Generating and caching population with 1e{} individuals...'.format(X))

        # Generate synthetic samples
        synthetic_data = gan.generate_samples(10**int(X))
        print("\nGenerated synthetic samples:")
        print(synthetic_data)

        # Some basic statistics for validation
        print('\n## Numericals, showing mean and std:')
        nums = [col for col in synthetic_data.columns if col not in cats]
        methods = {}
        for num in nums:
            real = (np.mean(data[num]), np.std(data[num]))
            synth = (np.mean(synthetic_data[num]), np.std(synthetic_data[num]))
            _all = pd.DataFrame([real, synth], columns=['Mean', 'STD']).T
            _all.columns = [[num + '_real', num + '_synth']]
            print('\n', num, _all)

        print('\n## Categoricals and ordinals, showing normalised value counts:')
        for cat in cats + ords:
            real = data[cat].value_counts(normalize=True).sort_index()
            real.name = cat + '_real'
            synth = synthetic_data[cat].value_counts(normalize=True).sort_index()
            synth.name = cat + '_synth'
            print('\n', pd.concat([real, synth], axis=1))

        outpath = DATA_DIR
        outfile = 'ganpop_1e' + str(X) + '.csv'
        synthetic_data.to_csv(os.path.join(outpath, outfile), index=False)
        print('Done!')
