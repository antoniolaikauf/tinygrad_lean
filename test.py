from tinygrad import Device, Tensor, nn

# mini rete neurale con un neurone

input = Tensor.arange(1,5).numpy()
target = Tensor.arange(1,5).numpy() * -1
weight = Tensor.rand(1).numpy()
eps = 0.01
lr = 0.001
print(f"input: {input} --> output: {target}")

for _ in range(200):
    out = input * weight
    print(f"output: --> {out}")
    loss = sum((out - target)**2)
    out1 = input * (weight + eps)
    loss1 = sum((out1 - target)**2)
    grad = (loss1 - loss) / eps
    weight -= lr * grad
    print(f"weight: --> {weight}")