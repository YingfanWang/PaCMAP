import matplotlib.pyplot as plt

def generate_figure(embedding, labels, title):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(embedding[:, 0], embedding[:, 1], s=0.5, c=labels, cmap='Spectral')
    ax.axis('off')
    ax.set_title(title)
    plt.savefig(f"./{title}.png")

def generate_combined_figure(embeddings, labels, titles, theme_title):
    len_subfigs = len(embeddings)
    assert len(labels) == len_subfigs
    assert len(titles) == len_subfigs
    n_rows = (len_subfigs + 2) // 3
    fig, axes = plt.subplots(n_rows, 3, figsize = (18, n_rows * 6))
    axes = axes.flatten()
    for i in range(len_subfigs):
        ax = axes[i]
        embedding = embeddings[i]
        label = labels[i]
        title = titles[i]
        ax.scatter(embedding[:, 0], embedding[:, 1], s=0.5, c=label, cmap='Spectral')
        ax.axis('off')
        ax.set_title(title)
    for i in range(3 * n_rows - len_subfigs):
        axes[-i - 1].axis('off')
    plt.savefig(f"./{theme_title}.png")
