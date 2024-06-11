import pandas as pd
import numpy as np
import plotly.express as px
import plotly.colors as pc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud, ImageColorGenerator, STOPWORDS
import matplotlib
import matplotlib.pyplot as plt
from src import ukhsa_colours as uc


class Visualisation:
    def __init__(self, df):
        self.df = df

    def __apply_custom_color_scale(self, fig):
        custom_color_scale = []
        for colour, hex_code in uc.UKHSA_non_text_colours.items():
            custom_color_scale.append(hex_code)

        # custom_color_scale = ["#00AB8E", "#00A5DF", "#84BD00", "#FF7F32", "#FFB81C", "#D5CB9F"]
        for i, trace in enumerate(fig.data):
            if hasattr(trace, "marker"):
                trace.marker.color = custom_color_scale[i % len(custom_color_scale)]
            elif hasattr(trace, "line"):
                trace.line.color = custom_color_scale[i % len(custom_color_scale)]
            elif hasattr(trace, "color"):
                trace.color = custom_color_scale[i % len(custom_color_scale)]

    def describe_columns(self):
        # can i get rid of this and keep the autoeda for dataprep/sweetviz
        """
        Generate a descriptive summary for the given DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame to be summarized.

        Returns:
            pd.DataFrame: A DataFrame containing various descriptive statistics for the columns.

        Description:
            This function generates a summary of the input DataFrame, including information such as data types,
            counts, number of NaN values, number of unique values (with and without NaN), proportions of values
            (with and without NaN), mean, standard deviation, 25th percentile, 50th percentile (median),
            and 75th percentile for numeric columns. For datetime columns, the minimum and maximum values are
            also included.

            The resulting DataFrame has the following columns:
                - 'column_name': Name of each column in the input DataFrame.
                - 'data_type': Data type of each column.
                - 'count_of_entries_with_na': Total number of rows in the DataFrame (including NaN values).
                - 'count_of_entries_wo_na': Number of non-null entries for each column.
                - 'nan_count': Number of NaN values for each column.
                - 'count_of_uniques_with_na': Number of unique values (including NaN) for each column.
                - 'count_of_uniques_wo_na': Number of unique values (excluding NaN) for each column.
                - 'proportion_count_with_na': Proportion of each value (including NaN) in each column.
                - 'proportion_count_wo_na': Proportion of each value (excluding NaN) in each column.
                - 'mean': Mean value for numeric columns.
                - 'std': Standard deviation for numeric columns.
                - 'min': Minimum value for numeric and datetime columns, NaN for non-numeric columns.
                - '25%': 25th percentile for numeric columns.
                - '50%': 50th percentile (median) for numeric columns.
                - '75%': 75th percentile for numeric columns.
                - 'max': Maximum value for numeric and datetime columns, NaN for non-numeric columns.

            Note:
            - The function differentiates numeric and datetime columns from non-numeric columns based on data types.
            - The function assumes that columns with datetime data type are of type 'datetime64', 'datetime', or 'timedelta'.
        """
        column_info_dict = {
            "data_type": self.df.dtypes,
            "count_of_entries_with_na": self.df.shape[0],
            "count_of_entries_wo_na": self.df.count(),
            "nan_count": self.df.isnull().sum(),
            "count_of_uniques_with_na": self.df.nunique(dropna=False),
            "count_of_uniques_wo_na": self.df.nunique(dropna=True),
            "proportion_count_with_na": self.df.apply(
                lambda col: col.value_counts(normalize=True, dropna=False).to_dict()
            ),
            "proportion_count_wo_na": self.df.apply(
                lambda col: col.value_counts(normalize=True, dropna=True).to_dict()
            ),
            "mean": self.df.mean(numeric_only=True),
            "std": self.df.std(numeric_only=True),
            "min": self.df.apply(
                lambda col: col.min()
                if pd.api.types.is_numeric_dtype(col)
                or pd.api.types.is_datetime64_any_dtype(col)
                else pd.NA
            ),
            "25%": self.df.quantile(0.25, numeric_only=True),
            "50%": self.df.quantile(0.50, numeric_only=True),
            "75%": self.df.quantile(0.75, numeric_only=True),
            "max": self.df.apply(
                lambda col: col.max()
                if pd.api.types.is_numeric_dtype(col)
                or pd.api.types.is_datetime64_any_dtype(col)
                else pd.NA
            ),
        }
        self.column_info_df = (
            pd.DataFrame(column_info_dict)
            .reset_index()
            .rename(columns={"index": "column_name"})
        )

        return self.column_info_df

    def plot_count_and_proportion(self, columns, dropna=False):
        """
        Plot interactive bar charts showing the count and proportion of specified columns in a DataFrame using Plotly.

        Args:
            df (pd.DataFrame): The pandas DataFrame containing the data.
                A DataFrame with the data to be visualized.

            columns (list): A list of column names to plot.
                A list of column names for which the count and proportion bar charts will be generated.

            dropna (bool, optional): Whether to include NaN values in the count and proportion calculations. Default is False.
                If True, NaN values will be excluded from the calculations and the plots.
                If False, NaN values will be included in the calculations and displayed as a separate category 'missing' in the plots.

        Returns:
            None
                This function displays interactive bar charts using Plotly to visualize the count and proportion of values in the specified columns.
        """
        if not isinstance(columns, list):
            raise TypeError("columns should be a list.")

        for column in columns:
            # Copy the DataFrame to avoid modifying the original data

            data_filled = self.df.copy()

            # Calculate and sort the count and proportion values
            count_values = data_filled[column].value_counts(dropna=dropna).sort_index()
            proportion_values = (
                data_filled[column]
                .value_counts(normalize=True, dropna=dropna)
                .sort_index()
            )

            # Replace NaN / NaT with 'missing' and Convert the index to strings for non-numeric columns (to handle NaN values)
            if not dropna:
                na_fill_value = "missing"
                count_values.index = count_values.index.fillna(na_fill_value).astype(
                    str
                )
                proportion_values.index = proportion_values.index.fillna(
                    na_fill_value
                ).astype(str)

            # Create subplots with 1 row and 2 columns
            fig = make_subplots(rows=1, cols=2)

            # Add count bar chart to the first column
            fig.add_trace(
                go.Bar(
                    x=count_values.index,
                    y=count_values.values,
                    name=f"Count ({column})",
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

            # Add proportion bar chart to the second column
            fig.add_trace(
                go.Bar(
                    x=proportion_values.index,
                    y=proportion_values.values,
                    name=f"Proportion ({column})",
                    showlegend=False,
                ),
                row=1,
                col=2,
            )

            # Set subtitles for each subplot
            fig.update_xaxes(title_text=f"{column}", row=1, col=1)
            fig.update_yaxes(title_text="Count", row=1, col=1)
            fig.update_xaxes(title_text=f"{column}", row=1, col=2)
            fig.update_yaxes(title_text="Proportion", row=1, col=2)

            # Update layout
            fig.update_layout(title=f"Counts and Proportions of {column}", height=400)

            if not dropna:
                fig.update_xaxes(type="category")

            self.__apply_custom_color_scale(fig)

            fig.show()

        # Show the figure
        return

    def custom_graph(
        self,
        df,
        x_column,
        y_column_and_type,
        xaxis_type=None,
        y_axes_title=None,
        x_axes_title=None,
        barmode=None,
        z_column=None,
        graph_title=None,
        yaxis_range=None,
        xaxis_range=None,
    ):
        """
        Create a custom graph using Plotly's make_subplots with support for multiple y-axes.

        Parameters:
            dataframe (pandas.DataFrame): The DataFrame containing the data.
            x_column (str): The column name for the x-axis values.
            y_column_and_type (dict): A dictionary mapping y-column names to their chart types ('line' or 'bar').
            y_axes_title (str, optional): Title for the y-axes. If None, the y-axis titles will be set to the column names.
            barmode (str, optional): Bar mode for the bar charts. Default is 'stack'.
                Other options are 'group' (for grouped bars) and 'overlay' (for overlaid bars).

        Returns:
            None (displays the plot)
        """
        if xaxis_type == "category":
            desired_output = (
                df[x_column].sort_values(ascending=True).reset_index(drop=True)
            )
            df[x_column] = df[x_column].astype("category")

        if z_column:
            fig = go.Figure()
            for z_value in df[z_column].unique():
                z_filtered_df = df[df[z_column] == z_value]
                for y_column, type in y_column_and_type.items():
                    if type == "line":
                        fig.add_trace(
                            go.Scatter(
                                x=z_filtered_df[x_column],
                                y=z_filtered_df[y_column],
                                mode="lines",
                                name=f"{z_column}={z_value}",
                            )
                        )
                    elif type == "scatter":
                        fig.add_trace(
                            go.Scatter(
                                x=z_filtered_df[x_column],
                                y=z_filtered_df[y_column],
                                mode="markers",
                                name=f"{z_column}={z_value}",
                            )
                        )
                    elif type == "bar":
                        fig.add_trace(
                            go.Bar(
                                x=z_filtered_df[x_column],
                                y=z_filtered_df[y_column],
                                name=f"{z_column}={z_value}",
                            )
                        )

            fig.update_yaxes(title_text=y_axes_title)

            if xaxis_type == "category":
                fig.update_xaxes(categoryorder="array", categoryarray=desired_output)

        # This code plots the data for 1 type of plot category (line or bar)
        else:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            for y_column, type in y_column_and_type.items():
                if (type == "line") & (y_axes_title is None):
                    fig.add_trace(
                        go.Scatter(
                            x=df[x_column],
                            y=df[y_column],
                            name=y_column,
                            line=dict(width=2),
                        ),
                        secondary_y=True,
                    )
                    fig.update_yaxes(
                        title_text=y_column, secondary_y=True
                    )  # is this when we have bars and lines?
                    # editing y axes title
                elif (type == "line") & (y_axes_title is not None):
                    fig.add_trace(
                        go.Scatter(
                            x=df[x_column],
                            y=df[y_column],
                            name=y_column,
                            line=dict(width=2),
                        ),
                        secondary_y=False,
                    )
                    fig.update_yaxes(title_text=y_axes_title, secondary_y=False)

                elif type == "bar":
                    fig.add_trace(
                        go.Bar(x=df[x_column], y=df[y_column], name=y_column),
                        secondary_y=False,
                    )
                    fig.update_yaxes(title_text=y_column, secondary_y=False)
                    # editing y axes title
                    if (
                        y_axes_title is None
                    ):  # what does this even do? can we get rid of it!
                        fig.update_yaxes(title_text=y_column, secondary_y=False)
                    else:
                        fig.update_yaxes(title_text=y_axes_title, secondary_y=False)
                    # when using 2 bars from different columsn can choose between stacked or grouped
                    fig.update_layout(barmode=barmode)

        fig.update_xaxes(title_text=x_axes_title)
        fig.update_layout(
            title=graph_title,
            yaxis_range=yaxis_range,
            xaxis_range=xaxis_range,
            xaxis_type=xaxis_type,
        )
        self.__apply_custom_color_scale(fig)
        return fig.show()

    def create_wordcloud(self, text, remove_words):
        """
            Generate and display a word cloud from a text column in a DataFrame.

        Parameters:
            text (str): The name of the text column in the DataFrame.
            remove_words (list): A list of words to remove from the word cloud.

        Returns:
            WordCloud: A WordCloud object representing the generated word cloud.

        """

        comment_words = ""
        stopwords = set(STOPWORDS)

        # iterate through the dataframe
        for val in self.df[text]:
            # typecaste each val to string
            val = str(val)
            # split the value
            tokens = val.split()

            # Converts each token into lowercase
            for i in range(len(tokens)):
                tokens[i] = tokens[i].lower()
                comment_words += " ".join(tokens) + " "

        for j in remove_words:
            comment_words = comment_words.replace(j, "")

        wordcloud = WordCloud(
            width=600,
            height=600,
            background_color="white",
            stopwords=stopwords,
            min_font_size=10,
        ).generate(comment_words)

        # plot the WordCloud image
        plt.figure(figsize=(8, 8), facecolor=None)
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad=0)

        plt.show()
        return wordcloud
