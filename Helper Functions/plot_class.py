# import pandas and matplotlib
import pandas as pd
import matplotlib.pyplot as plt

def create_histogram(df):
    df.hist()
    plt.show()
    return

def create_column_chart_all_columns(df):
    df.plot.bar()
    return


def create_column_chart_2_columns(df, x_aksonas_column_name, y_aksonas_column_name):
    plt.bar(df[x_aksonas_column_name], df[y_aksonas_column_name])
    plt.xlabel(x_aksonas_column_name)
    plt.ylabel(y_aksonas_column_name)
    plt.show()
def create_box_plot_chart_for_each_column(df):
    df.plot.box()
    return
def create_box_plot_chart_for_one_column(df,column_name):
    plt.boxplot(df[column_name])
    plt.show()

def create_pie_chart(df, column_name):
    plt.pie(df[column_name],
    autopct ='% 1.1f %%', shadow = True)
    plt.show()


def create_scatter_plot(df, x_aksonas_column_name, y_aksonas_column_name):
    # scatter plot between income and age
    plt.scatter(df[x_aksonas_column_name], df[y_aksonas_column_name])
    plt.show()


def create_classsic_plot(column_x, column_y, column_name_x="x-axis",
                         column_name_y="y-axis", scatter=False):
    plt.xlabel(column_name_x)
    plt.ylabel(column_name_y)
    if (scatter == True):
        plt.scatter(column_x, column_y)
    else:
        plt.plot(column_x, column_y)