import json
import matplotlib.pyplot as plt
from dataset import Tokenizer, process_data, read_json, CustomDataset


def read_json(filename):
    with open(filename, "r") as f:
        return json.load(f)

def main():
    vocab_dict = read_json("./vocab.json")
    tokenizer = Tokenizer()
    tokenizer.build_vocab_from_dict(vocab_dict)

    data = read_json("./all_data.json")
    images = data["images"]
    labels = data["latex"]

    split = int(len(images)*0.3)
    train_dataset = CustomDataset(images[split:], labels[split:], 48, 128)
    valid_dataset = CustomDataset(images[:split], labels[:split], 48, 128)

    # fig, axs = plt.subplots(4, 4)
    # pos = [0, 0]
    # for i in range(16):
    #     axs[pos[0], pos[1]].imshow(train_dataset[i][0].permute(1, 2, 0))
    #     pos[1] += 1
    #     if pos[1] == 4:
    #         pos[1] = 0
    #         pos[0] += 1
    # plt.show()
    # train_loader = get_loader


if __name__ == "__main__":
    main()