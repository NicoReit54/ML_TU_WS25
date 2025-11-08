import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def create_correlation_matrix(df: pd.DataFrame, 
                              column_names: list = None, 
                              title: str = "Correlation Matrix",
                              columns: list=None,
                              figsize: tuple = (10, 10)) -> None:
    """
    Plots a heatmap of the correlation matrix for a given DataFrame.

    Parameters:
    ----------
    column_names : list of strings
        Custom labels for the x and y axes of the heatmap. Should match the number of columns in `df`.
    df : pd.DataFrame
        The DataFrame containing (numerical!) features to compute correlations.
    title : str
        Title for the plot
    columns: list of str
        Lets the user decide which columns to choose.
    figsize : tuple
        A tuple containing the wished for size of the scatterplot

    Returns:
    -------
    None
        Displays a seaborn heatmap of the correlation matrix.
    """

    if columns:
        df = df[columns]
    
    # If no column_names are passed
    if column_names is None:
        column_names = df.columns.tolist()

    # Check if (potentially passed) names are of the same length
    elif len(column_names) != df.shape[1]:
        raise ValueError("Length of column_names must match number of DataFrame columns.")

    # Correlation matrix
    correlations = df.corr()
    # Plot figsize
    fig, ax = plt.subplots(figsize=figsize)
    # Generate Color Map
    colormap = sns.diverging_palette(220, 10, as_cmap=True)

    # Generate Heat Map, allow annotations and place floats in map
    sns.heatmap(correlations, cmap=colormap, annot=True, fmt=".2f")
    ax.set_xticklabels(
        column_names,
        rotation=45,
        horizontalalignment='right'
    )
    ax.set_yticklabels(column_names)
    ax.set_title(title)
    plt.show()
    plt.tight_layout()


def create_scatterplot_matrix(df: pd.DataFrame,
                              title: str = "Scatterplot Matrix",
                              columns: list = None,
                              figsize: tuple = (16, 18),
                              y_rotation: int = 0,
                              y_offset: int = -0.7,
                              x_rotation: int = -40) -> None:
    """
    Generates a scatterplot matrix for all numerical columns in the DataFrame.

    Parameters:
    ----------
    df : pd.DataFrame
        The DataFrame containing numerical features to visualize pairwise relationships.
    columns: list of str
        Lets the user decide which columns to choose.
    title : str
        Title for the plot
    figsize : tuple
        A tuple containing the wished for size of the scatterplot
    y_rotation: int 
        Degrees in how the y-axis labels shall be set
    y_offset: int
        How much the y labels shall be moved to the left (negative value) or 
        to the right in order to not overlap with the graph.
    x_rotation: int 
        Degrees in how the x-axis labels shall be set

    Returns:
    -------
    None
        Displays a scatterplot matrix with histograms on the diagonal.
    """

    # If the user passes columns to choose 
    if columns:
        df = df[columns]

    # Scatterplot Matrix
    sm = pd.plotting.scatter_matrix(df, figsize=figsize, diagonal='hist')
     
    # Change label rotation + may need to offset label when rotating to prevent overlap of figure
    for ax in sm.ravel():
        ax.xaxis.label.set_rotation(x_rotation)
        ax.yaxis.label.set_rotation(y_rotation)
        ax.get_yaxis().set_label_coords(y_offset, 0.5)
        
        # Hide all ticks
        ax.set_xticks(())
        ax.set_yticks(())

    print(f"Scatter matrix shape: {sm.shape}")
    # Adds a centered "super title" to the figure as this consists now of more than one axes. See docu for mroe!
    plt.suptitle(title, y = 0.9)  

    plt.show()

    fig = plt.gcf()  # Get current figure after scatter_matrix
    fig.tight_layout()


def create_countplot(
        df: pd.DataFrame,
        x_rotation: int = 45,
        figsize: tuple = (30, 30)) -> None:
    """
    Generates a countplot matrix for all categorical columns in a DataFrame.

    Parameters:
    ----------
    df : pd.DataFrame
        The DataFrame containing features to visualize. Categorical features will be extracted automatically
    figsize : tuple
        A tuple containing the wished for size of the scatterplot
    x_rotation: int 
        Degrees in how the x-axis labels shall be set

    Returns:
    -------
    None
        Displays a countplot matrix of all categorical columns in a given DataFrame.
    """
    # extract the categorical values
    df_categorical = df.select_dtypes(include = 'object').copy()

    cols = df_categorical.columns

    fig, axes = plt.subplots(int(len(cols)/4), 4 + 1, figsize=figsize)  
    axes = axes.flatten() # turn into a 1D list for indexing

    for i, col in enumerate(cols):
        sns.countplot(data=df_categorical, x=col, ax=axes[i])
        axes[i].set_title(col)
        axes[i].tick_params(axis='x', labelrotation=x_rotation)
        axes[i].set_xlabel("") # no label as in titel already

    fig.tight_layout() # for better readability