import torch
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from torchvision import transforms
import random
from PIL import Image

class AEBSVideoDataset(Dataset):
    def __init__(self, data_dir="data/aebs", seq_len=64, image_size=32, distance_ub=60.0, random_seed=533):
        random.seed(random_seed)
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.image_size = image_size
        self.distance_ub = distance_ub
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        data_pt_file = os.path.join(data_dir, "data_seed_{}.pt".format(random_seed))
        if os.path.exists(data_pt_file):
            self.image_sequences, self.distance_sequences = torch.load(data_pt_file)
        else:
            self.build_dataset(data_dir)
            torch.save((self.image_sequences, self.distance_sequences), data_pt_file)


    def build_dataset(self, data_dir):

        for sequence_dir in os.listdir(data_dir):
            sequence_path = os.path.join(data_dir, sequence_dir)
            if not os.path.isdir(sequence_path):
                continue
            csv_path = os.path.join(sequence_path, "log.csv")
            df = pd.read_csv(csv_path, skiprows=1, header=None)
            distances = torch.tensor(df.iloc[:, 1].values)
            image_paths = df.iloc[:, -1].values

            valid_indices = ((distances <= self.distance_ub) & (distances >= 0.0)).nonzero().squeeze()
            distances = distances[valid_indices]
            image_paths = [os.path.join(sequence_path, image_path) for image_path in image_paths[valid_indices]]

            self.image_sequences = []
            self.distance_sequences = []
            # create 10 random sequences with length of 64
            for _ in range(50):
                ## first randomly choose max distance and min distance
                while True:
                    max_distance = random.uniform(30.0, 60.0)
                    min_distance = random.uniform(0.0, max_distance - 20.0)
                    valid_indices = ((distances <= max_distance) & (distances >= min_distance)).nonzero().squeeze()
                    sub_distances = distances[valid_indices]
                    sub_image_paths = [image_paths[idx] for idx in valid_indices]
                    if len(sub_distances) >= self.seq_len:
                        break
                
                while True:
                    random_idx = random.sample(range(len(sub_distances)), self.seq_len)
                    random_idx.sort()
                    seq_distances = sub_distances[random_idx].type(torch.float32)
                    diff = seq_distances[1:] - seq_distances[:-1]
                    seq_image_path = [sub_image_paths[idx] for idx in random_idx]
                    if (diff >= -3.0).all():
                        break

                image_sequence = []
                for file_path in seq_image_path:
                    assert os.path.exists(file_path), f"{file_path} does not exist"
                    image = Image.open(file_path).convert("RGB")
                    image = image.crop((250, 250, 550, 550))
                    image = self.transform(image)
                    image_sequence.append(image)
                image_sequence = torch.stack(image_sequence, dim=0).type(torch.float32)
                
                self.image_sequences.append(image_sequence)
                self.distance_sequences.append(seq_distances)

    def __len__(self):
        return len(self.distance_sequences)

    def __getitem__(self, idx):
        image_sequence = self.image_sequences[idx]
        distances = self.distance_sequences[idx] / self.distance_ub
        return image_sequence, distances

class DummyDataset(Dataset):
    def __init__(self, seq_len):
        self.seq_len = seq_len

    def __len__(self):
        return 32

    def __getitem__(self, idx):
        data = torch.randn(self.seq_len, 3, 32, 32)
        labels = torch.randn(self.seq_len)
        return data, labels

# main function
if __name__ == "__main__":
    dataset = AEBSVideoDataset()
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    iterator = iter(dataloader)
    images, distances = next(iterator)
    print(images.shape, distances.shape)
