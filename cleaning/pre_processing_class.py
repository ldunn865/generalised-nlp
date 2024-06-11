import pandas as pd
import numpy as np
import yaml

# from pandas_profiling import ProfileReport #there is a warning about deprecation
from dataprep.eda import plot, plot_correlation, create_report, plot_missing
import sweetviz as sv

with open("../config/config.yaml", "r") as f:
    config = yaml.safe_load(f)


class PreProcessing:
    def __init__(self, df):
        self.df = df

    def get_dataframe_report(self):
        # pandas profiling
        # self.profile = ProfileReport(self.df, title="Pandas Profiling Report")
        # self.profile.to_file('../outputs/output.html')

        try:
            # dataprep
            self.profile = create_report(self.df)
            # stem_path = config["output_folder"] + 'dataprep-EDA-Report'
            # report.save(filename= stem_path)
        except Exception as e:
            # sweetviz
            self.profile = sv.analyze(self.df)
            self.profile.show_notebook()
            # self.report.show_html()

        return self.profile

    def clean_column_names(self):
        self.df.columns = self.df.columns.str.lower().str.strip().str.replace(" ", "_")
        return self.df

    def remove_duplicates(self):
        number_of_duplicates = self.df.duplicated().sum()
        self.df = self.df.drop_duplicates()

        # tell user how many duplicates were removed
        return print(f"Number of duplicates dropped: {number_of_duplicates}\n")

    def lowercase_strip_rows(self, columns_to_clean):
        """Transforms the entire column by lowercasing and removing trailing or leading whitespace

        Args:
            df (pandas DataFrame): a pandas dataframe.
            columns_to_lower (list): A list of columns the user wants to lowercase.


        Returns:
            A dataframe with transformed columns.

        """
        if isinstance(columns_to_clean, pd.core.indexes.base.Index):
            columns_to_clean = columns_to_clean.tolist()
        elif isinstance(columns_to_clean, list):
            pass
        else:
            print(f"{columns_to_clean} should be either a list or an Index")

        for x in columns_to_clean:
            try:
                self.df[x] = self.df[x].str.lower().str.strip()
            except:
                print(f"Unable to convert {x} to str.lower")

        return self.df

    def fill_or_remove_missing_values(self, replacement_dict: dict):
        """
        Returns a dataframe with removed or replaced missing values.

        Args:
            replacement_columns (dictionary): A dictionary containing information about whether the user
                                            wants to remove NaN or replace with a value. The key represent
                                            the column names the user and the keys represent either 'remove'
                                            if user wants to remove nas in that column or the replacement
                                            string on the NaN.
                                            Example dict replacement_columns = {'col_name' : 'missing',
                                                                                'col_name_1' : 0,
                                                                                    'col_name_2' : 'remove_row',
                                                                                    'col_name_3' : 'remove_col'})

        Returns:
            A transformed dataframe.
        """

        for column, solution in replacement_dict.items():
            if solution == "remove_row":
                try:
                    self.df = self.df.dropna(subset=column)
                except KeyError as e:
                    print(f"{e} not found in dataframe")

            elif solution == "remove_col":
                try:
                    self.df = self.df.drop(columns=[column])
                except KeyError as e:
                    print(f"{e} not found in dataframe")

            else:
                if column in self.df.columns:
                    self.df[column] = self.df[column].fillna(solution)
                else:
                    print(f"{column} not found in dataframe")

        return self.df

    def convert_datatype(self, column_types: dict):
        """This function transforms specific columns datatypes depending on whether the user wants to keep NaN or not.

        Args:
            column_types (dict): A dictionary with columns as keys and datatype as values.
            datatype needs to be in quotes if user wants to keep NaN values.
            Example: {'col_name': 'integer', 'col_name_2': 'int64', 'col_name' : str}



        Returns:
            A DataFrame with updated data types.



        """

        if not isinstance(column_types, dict):
            raise TypeError(
                "column_types should be a dictionary. Example - {'col_name' : 'integer','col_name_2' : int64}"
            )

        for column, datatype in column_types.items():
            if column in self.df.columns:
                try:
                    self.df[column] = pd.to_numeric(
                        self.df[column], downcast=datatype, errors="coerce"
                    )  # if we want to keep na for integer/float columns
                except ValueError:
                    self.df[column] = self.df[column].astype(
                        datatype
                    )  # deals with converting to string or to boolean

            # create a warning to let user know there is NaTs
            Number_of_NaN = self.df[column].isnull().sum()
            if Number_of_NaN > 1:
                print(
                    f"\033[91mWARNING\033[0m there are {Number_of_NaN} NaT in the {column} column"
                )
            else:
                print(f"{column} not found in dataframe")

        return self.df

    def convert_to_datetime(self, replacement_dict: dict):
        """This returns a dataframe with the correct date format.

        Args:
            df (pd.DataFrame): A pandas DataFrame.
            replacement_columns (dict): A replacement dictionary where keys represent column name and values represent either the format of the string or the units of the integer.
                                        Example dict replacement_columns={'date': '%Y-%m-%d %H:%M:%S',
                                                                            'date_int': 'ns'}

        Returns
            A pandas DataFrame.


        """

        for key, value in replacement_dict.items():
            if value in ["D", "s", "ms", "us", "ns"]:
                self.df[key] = pd.to_datetime(self.df[key], unit=value, errors="ignore")
            else:
                self.df[key] = pd.to_datetime(
                    self.df[key], format=value, errors="ignore"
                )

            # Check if the column is now in datetime format
            if not pd.api.types.is_datetime64_any_dtype(self.df[key]):
                print(f"Conversion to datetime format failed for the column '{key}'.")

            # create a warning to let user know there is NaTs
            Number_of_NaT = (
                self.df[key].isnull().sum()
            )  # look at what happens if they cant convert 'missing'
            if Number_of_NaT > 1:
                print(
                    f"\033[91mWARNING\033[0m there are {Number_of_NaT} NaT in the {key} column"
                )

        return self.df

    def extract_date_info(self, date_column: list, replacement_dict: dict):
        """Returns a dataframe with new columns representing different time stamps

        Args:
        date_column (list): The date column/columns in the form of a list.
        options (dict): Dictionary of options for date components to extract.
        The dict should be in the format options = {"date": bool,
                                                    "year": bool,
                                                    "quarter": bool,
                                                    "month": bool,
                                                    "day": bool,
                                                    "day_name": bool,
                                                    "time": bool,
                                                    "hour": bool,
                                                    "strftime": bool,
                                                    "custom": str}

        Returns:
        A dataframe

        """
        if not isinstance(date_column, list):
            raise TypeError("date_column should be a list.")

        for col in date_column:
            date_info = pd.to_datetime(self.df[col])
            for component, extract in replacement_dict.items():
                if component == "strftime":
                    self.df[f"{col}_{component}"] = date_info.dt.strftime("%p")
                elif component == "day_name":
                    self.df[
                        f"{col}_{component}"
                    ] = date_info.dt.day_name()  # does this need to be here?
                elif component == "custom":
                    self.df[f"{col}_{component}"] = date_info.dt.strftime(extract)
                else:
                    extracted_component = getattr(date_info.dt, component)
                    self.df[f"{col}_{component}"] = extracted_component

        return self.df
