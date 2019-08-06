import matplotlib.pyplot as plt


def plot_df(subplots=False, df=None, folder='reports', title='data'):
    plt.clf()
    col = df.columns
    if subplots:
        n = len(col)
        f, ax = plt.subplots(n, 1, figsize=(15, 10), sharex=True)
        for i in range(n):
            df.plot(y=col[i], ax=ax[i])
        f.savefig('reports/%s.png'%title)
    else:
        df.plot(y=col, title=title)
        plt.savefig(folder+'/%s.png'%title)
    #plt.show()


def plot_histogram(df = None, bins = 10, folder = 'reports', title = 'data'):
    plt.clf()
    df.hist(bins=bins, figsize=(20,15))
    plt.savefig(folder + '/%s.png' % title)
    #plt.show()