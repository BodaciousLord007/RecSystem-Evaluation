import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_results(frame, output, k):
    columns = [f'precision@{k}', f'recall@{k}', f'ndcg@{k}', 'catalog_coverage']
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for column, axis in zip(columns, axes.flat):
        frame.plot.barh(x='model', y=column, ax=axis, legend=False, color='#356a99')
        axis.set_title(column)
        axis.set_ylabel('')
        axis.set_xlim(left=0)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)

