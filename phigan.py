import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


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
        self.encoder = None
        self.categorical_cols = None
        self.numerical_cols = None
        self.encoded_categories = None

    def preprocess_data(self, data, categorical_cols=None):
        """Preprocesses the data by scaling numerical and encoding categorical features"""
        if categorical_cols is None:
            categorical_cols = []

        self.categorical_cols = categorical_cols
        self.numerical_cols = [col for col in data.columns if col not in categorical_cols]

        # Process numerical features
        if self.numerical_cols:
            self.scaler = MinMaxScaler(feature_range=(-1, 1))
            scaled_numerical = self.scaler.fit_transform(data[self.numerical_cols])
        else:
            scaled_numerical = np.empty((len(data), 0))

        # Process categorical features
        if self.categorical_cols:
            self.encoder = OneHotEncoder(sparse_output=False)
            encoded_categorical = self.encoder.fit_transform(data[self.categorical_cols])
            self.encoded_categories = self.encoder.categories_
        else:
            encoded_categorical = np.empty((len(data), 0))

        # Combine features
        processed_data = np.concatenate([scaled_numerical, encoded_categorical], axis=1)
        return processed_data.astype(np.float32)

    @staticmethod
    def lol_to_bins(lol):
        """Takes list of lists (lol) of categoricals and returns positions as ranges"""
        right = list(np.cumsum([len(el) for el in lol]))
        left = [0] + list(right[:-1])
        print(left, right)
        return [range(start, end) for start, end in zip(left, right)]

    def postprocess_data(self, generated_data):
        """Converts generated data back to original format"""
        if self.numerical_cols and self.categorical_cols:
            num_numerical = len(self.numerical_cols)
            numerical_data = generated_data[:, :num_numerical]
            categorical_data = generated_data[:, num_numerical:]

            # Inverse transform numerical
            numerical_data = self.scaler.inverse_transform(numerical_data)

            # Create DataFrame
            df = pd.DataFrame(numerical_data, columns=self.numerical_cols)

            # Inverse transform categorical
            bins = self.lol_to_bins(self.encoded_categories)
            for i, col in enumerate(self.categorical_cols):
                maxes = np.argmax(categorical_data[:, bins[i]], axis=1)
                categorical_data_inverse = [self.encoded_categories[i][j] for j in maxes]
                df[col] = categorical_data_inverse

        elif self.numerical_cols:
            numerical_data = self.scaler.inverse_transform(generated_data)
            df = pd.DataFrame(numerical_data, columns=self.numerical_cols)

        else:
            df = pd.DataFrame()

            # Inverse transform categorical
            bins = self.lol_to_bins(self.encoded_categories)
            for i, col in enumerate(self.categorical_cols):
                maxes = np.argmax(generated_data[:, bins[i]], axis=1)
                categorical_data_inverse = [self.encoded_categories[i][j] for j in maxes]
                df[col] = categorical_data_inverse

        return df

    def train(self, data, categorical_cols=None, epochs=1000, batch_size=32, lr=0.0002):
        """Train the GAN on tabular data"""
        # Preprocess data
        processed_data = self.preprocess_data(data, categorical_cols)
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
            if epoch % 100 == 0:
                print(f"[Epoch {epoch}/{epochs}] D loss: {d_loss.item():.4f}, G loss: {g_loss.item():.4f}")

    def generate_samples(self, n_samples):
        """Generate synthetic samples"""
        if not self.generator:
            raise ValueError("Model not trained yet. Call train() first.")

        z = torch.randn(n_samples, self.latent_dim)
        with torch.no_grad():
            generated_data = self.generator(z).numpy()

        return self.postprocess_data(generated_data)


# Example usage
if __name__ == "__main__":
    # Example with synthetic data
    size = 100
    data = pd.DataFrame({
        'age': np.random.normal(40, 15, size),
        'income': np.random.lognormal(4, 0.5, size),
        'gender': np.random.choice(['M', 'F'], size),
        'education': np.random.choice(['High School', 'College', 'Graduate'], size)
    })

    # Initialize and train GAN
    gan = TabularGAN(latent_dim=64, hidden_dim=128)
    gan.train(data, categorical_cols=['gender', 'education'], epochs=1000, batch_size=32)

    # Generate synthetic samples
    synthetic_data = gan.generate_samples(10)
    print("\nGenerated synthetic samples:")
    print(synthetic_data)
