import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.dates import DateFormatter


def plot_df(subplots=False, df=None, folder='reports', title='data'):
    # plot dataframe and save the figure under the folder reports
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
    # plot histogram of dataframe and save the figure under the folder reports
    plt.clf()
    df.hist(bins=bins, figsize=(20,15))
    plt.savefig(folder + '/%s.png' % title)
    #plt.show()


# Define a function for the line plot with intervals
def lineplotCI(x, y_predicted, y_history, low_CI, upper_CI, x_label, y_label, folder, title):

    # Create the plot object
    f, ax = plt.subplots()

    # Plot the data, set the linewidth, color and transparency of the
    # line, provide a label for the legend
    ax.plot(x, y_predicted, lw = 1, color = 'orange', alpha = 1, label = 'Predictions')
    ax.plot(x, y_history, lw=1, color = 'darkseagreen', alpha = 1, label = 'True')
    # Shade the confidence interval
    ax.fill_between(x, low_CI, upper_CI, color = 'orange', alpha = 0.4, label = '95% CI')
    # Label the axes and provide a title
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Display legend
    ax.legend(loc = 'best')

    plt.gcf().autofmt_xdate()
    #ax.xaxis.set_major_formatter(DateFormatter("%m/%y"))

    f.savefig(folder + '/%s.png' %title)
    #plt.show()


def vertical_plot():
    forecast_result = pd.read_csv('reports/forecast_result.csv', index_col=0)

    monthly_range = []
    monthly_range.append(forecast_result.loc['2017-11-15':'2017-11-25'].index.values)
    monthly_range.append(forecast_result.loc['2017-12-15':'2017-12-25'].index.values)
    monthly_range.append(forecast_result.loc['2018-01-15':'2018-01-25'].index.values)
    monthly_range.append(forecast_result.loc['2018-02-15':'2018-02-25'].index.values)

    plt.clf()
    fig, ax = plt.subplots()
    ax.plot(forecast_result['Forecast'])
    for i, x_span in enumerate(monthly_range):
        if i%2 == 0:
            ax.axvspan(x_span[0], x_span[-1], alpha=0.4, color='tan')
        else:
            ax.axvspan(x_span[0], x_span[-1], alpha=0.4, color='bisque')

    ax.set_title('Actuals Forecast')
    ax.set_xlabel('Forecast Date')
    ax.set_ylabel('Forecast Actuals Values')
    plt.gcf().autofmt_xdate()
    #ax.xaxis.set_major_formatter(DateFormatter("%d/%m"))
    #plt.show()
    fig.savefig('reports/forecast result.png')