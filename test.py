import torch

timesteps = torch.randint(0, 1000, (4,), device='cpu')\
                                    .long().chunk(2)[0].repeat(2)
print(timesteps)