import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

res = pd.read_csv('results/fewshot_results.csv')
res


def plot_results(res: pd.DataFrame, title: str):
    # Plot f1_seen and f1_unseen vs n_epochs. 
    # Add dot for each y value to see the actual values.
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='n_epochs', y='f1_seen', data=res)
    sns.scatterplot(x='n_epochs', y='f1_unseen', data=res)
    sns.lineplot(x='n_epochs', y='f1_seen', data=res)
    sns.lineplot(x='n_epochs', y='f1_unseen', data=res)
    plt.legend(['f1_seen', 'f1_unseen'])
    plt.xlabel('n_epochs')
    plt.ylabel('F1')
    plt.title(title)
    plt.show()


def plot_results_nshots(res: pd.DataFrame, title: str, savename: str):
    # Plot f1_seen and f1_unseen for each n_shot vs n_epochs. 
    # Add dot for each y value to see the actual values.
    # Plot a line for each n_shots value.
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    for n_shots in res['n_shots'].unique():
        res_n_shots = res[res['n_shots'] == n_shots]
        # Only if there is any data in f1_unseen
        if not res_n_shots['f1_unseen'].isnull().all():
            sns.scatterplot(x='n_epochs', y='f1_unseen', data=res_n_shots)
            sns.lineplot(x='n_epochs', y='f1_unseen', data=res_n_shots, label=f'f1_unseen_{n_shots}')
        # Add untrained starting point.
        if 0 not in res_n_shots['n_epochs'].unique():
            res_n_shots = pd.concat([res_n_shots, res[res['n_epochs'] == 0]])
        sns.scatterplot(x='n_epochs', y='f1_seen', data=res_n_shots)
        sns.lineplot(x='n_epochs', y='f1_seen', data=res_n_shots, label=f'f1_seen_{n_shots}')
    # PLace legend a bit lower than the mid right part of the plot.
    plt.legend(loc='best', bbox_to_anchor=(0.5, 0., 0.5, 0.5))
    plt.xlabel('Num. Epochs')
    plt.ylabel('F1')
    plt.title(title)
    plt.savefig(f'results/{savename}.png')
    plt.show()


def plot_usp_results(res):
    '''Plot results for USP training.'''
    res1 = res[(res['freeze_zstc'] == True) & (res['freeze_usp'] == False)]
    for lr in res1['lr_usp'].unique():
        if lr == 0:
            continue
        res2 = res1[res1['lr_usp'] == lr]
        if 0 not in res2['n_epochs'].unique():
            res2 = pd.concat([res2, res[res['n_epochs'] == 0]])
        plot_results_nshots(res2, title=f'Train USP (lr={lr})', savename=f'usp_lr{lr}')


def plot_zstc_results(res):
    '''Plot results for ZSTC training.'''
    res1 = res[(res['freeze_zstc'] == False) & (res['freeze_usp'] == True)]
    for lr in res1['lr_zstc'].unique():
        if lr == 0:
            continue
        res2 = res1[res1['lr_zstc'] == lr]
        if 0 not in res2['n_epochs'].unique():
            res2 = pd.concat([res2, res[res['n_epochs'] == 0]])
        plot_results_nshots(res2, title=f'Train ZSTC (lr={lr})', savename=f'zstc_lr{lr}')


def plot_usp_plus_zstc_results(res):
    res1 = res[(res['freeze_zstc'] == False) & (res['freeze_usp'] == False)]
    for lr_z in res1['lr_zstc'].unique():
        for lr_u in res1['lr_usp'].unique():
            if lr_z == 0 or lr_u == 0:
                continue
            res2 = res1[(res1['lr_zstc'] == lr_z) & (res1['lr_usp'] == lr_u)]
            if 0 not in res2['n_epochs'].unique():
                res2 = pd.concat([res2, res[res['n_epochs'] == 0]])
            plot_results_nshots(res2,
                                title=f'Train ZSTC (lr={lr_z}) and USP (lr={lr_u})',
                                savename=f'zstc_lr{lr_z}_usp_lr{lr_u}')
