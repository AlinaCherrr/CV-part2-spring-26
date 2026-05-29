import statistics
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def prepare_data() -> TensorDataset:
    X = torch.randn(10000, 128)
    y = torch.randint(0, 2, (10000,))

    # Шум добавляем один раз, а не на каждом батче иначе модель каждый раз видит разные данные, метрики нечестные
    noise = torch.randn(X.shape)
    X = X + noise

    dataset = TensorDataset(X, y)
    return dataset


def train():
    dataloader = DataLoader(prepare_data(), batch_size=256, shuffle=True)

    model = nn.Sequential(
        nn.Linear(128, 512), nn.ReLU(),
        nn.Linear(512, 128), nn.ReLU(),
        nn.Linear(128, 2)
    ).cuda().train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    losses_history = []
    forward_times = []
    backward_times = []

    for batch_idx, (data, target) in enumerate(dataloader):
        data = data.to('cuda')
        target = target.to('cuda')

        optimizer.zero_grad()

        # synchronize() нужен, чтобы GPU успел закончить работу до замера
        # CPU и GPU асинхронны, без этого time.time() меряет не то
        torch.cuda.synchronize()
        time_start = time.time()

        output = model(data)
        loss = criterion(output, target)

        torch.cuda.synchronize()
        time_end = time.time()
        forward_times.append(time_end - time_start)

        torch.cuda.synchronize()
        time_start_bwd = time.time()

        loss.backward()

        torch.cuda.synchronize()
        time_end_bwd = time.time()
        backward_times.append(time_end_bwd - time_start_bwd)

        optimizer.step()

        # .item() вытаскиваем число из тензора, иначе в списке копится весь граф вычислений и память утекает до OOM
        losses_history.append(loss.item())
        print(f"Batch {batch_idx} loss: {loss.item():.4f}")

        # убрала empty_cache(), он не спасает от OOM, но тормозит весь цикл

    print(f"Epoch finished, avg forward time is {statistics.mean(forward_times)}, "
          f"avg backward time is {statistics.mean(backward_times)}")

if __name__ == '__main__':
    train()