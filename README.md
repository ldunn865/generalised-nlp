# Review/Survey Data Analysis Project

This project analyzes review/survey data, cleans the data using a cleaning class, and performs data analysis using an analysis class. It also includes a notebook section for conducting topic modeling.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Testing](#testing)
- [Version-Control](#Version-Control)



## Installation

Once you have cloned the repository to your local machine, install all packages by running:

```poetry

#Type into terminal
    poetry install #install dependencies
    poetry init #activates the environmnt


#more information on poetry can be found here https://python-poetry.org/docs/basic-usage/

```





## Usage

```python
from your_project import Visualisation, PreProcessing

### Load your dataset into a DataFrame (replace 'your_data.csv' with your data file)
df = pd.read_csv('your_data.csv')

### Initialize the Visualisation and PreProcessing classes
visualizer = Visualisation(df)
preprocessor = PreProcessing(df)

### Generate a data profiling report
data_report = preprocessor.get_dataframe_report()

### Remove duplicates
preprocessor.remove_duplicates()

### Clean column names
cleaned_df = preprocessor.clean_column_names()

### Fill or remove missing values based on a replacement dictionary
replacement_dict = {'column1': 'missing', 'column2': 'remove_row', 'column3': 0}
preprocessed_df = preprocessor.fill_or_remove_missing_values(replacement_dict)

### Convert data types for specific columns
column_types = {'numeric_column': 'integer', 'boolean_column': bool}
preprocessed_df = preprocessor.convert_datatype(column_types)

### Convert columns to datetime format
date_conversion_dict = {'date_column1': '%Y-%m-%d', 'date_column2': 'ns'}
preprocessed_df = preprocessor.convert_to_datetime(date_conversion_dict)

### Extract date-related information from date columns
date_column_to_extract = ['date_column1']
date_extraction_options = {'year': True, 'month': True, 'day': True, 'day_name':  True}
preprocessed_df = preprocessor.extract_date_info(date_column_to_extract,  date_extraction_options)

### Generate a descriptive summary of the columns
column_summary = visualizer.describe_columns()

### Plot count and proportion charts for specific columns
columns_to_plot = ['column1', 'column2', 'column3']
visualizer.plot_count_and_proportion(columns_to_plot, dropna=False)

### Create a custom graph with multiple y-axes
custom_graph_settings = {
    'x_column': 'date',
    'y_column_and_type': {'value1': 'line', 'value2': 'bar'},
    'xaxis_type': 'category',
    'y_axes_title': 'Values',
    'x_axes_title': 'Date',
    'barmode': 'group',
    'graph_title': 'Custom Graph',
    'yaxis_range': [0, 100],
    'xaxis_range': ['2022-01-01', '2022-12-31']
}
visualizer.custom_graph(df, **custom_graph_settings)
```

### Configuration 

Create config folder on top level of directory
Create a `config/config.yaml` file:

```
data_raw_folder: "../data/raw/"
data_processed_folder: "../data/processed/"
output_folder: "../outputs/"
data_folder: "../data/"
```



## Testing

We don't currently have tests for this repo. But the following is a guideline for when we do.

For consistency the tests are run using pytest. To run the tests, run python and pytest in the conda environment:

conda activate doc_extract cd /path/to/pdf-table-extractor python -m pytest


## Version-Control

An unchecked git push from a laptop will go into the git history on GitHub, with any issues only being flagged when the GitHub actions run on a PR or commit on GitHub.

nbstripout should be installed on your laptop and pre-commit should be installed for this repository.

These will be installed if you create the environment from spec-file.txt as above and run pre-commit install as below.

### pre-commit
This project uses pre-commit to run a series of checks on the code before it is committed to the repository.

If you create the environment from spec-file.txt as above, then pre-commit is installed, but also needs installing for this specific repository. The environment needs to be activated at all times when working on this repository. To install for the repository, run:

```bash
cd /path/to/pdf-table-extractor
pre-commit install
```


### nbstripout
nbstripout installs a git hook to clear jupyter notebook output cells on commit or on demand. If you are working locally on a laptop you must have nbstripout installed on your laptop and share a screenshot with your line manager.

If you create and activate the environment from spec-file.txt as above, then nbstripout is installed. The environment needs to be activated at all times when working on this repository. To configure, run:

```bash
mkdir -p~/.config/git # This folder may not exist
nbstripout --install --global --attributes=~/.config/git/attributes

# These set your name and email address so your commits can be tracked
git config --global user.name "YOUR_FIRST_NAME_LAST_NAME"
git config --global user.email "YOUR__EMAIL_ADDRESS"

```

Then take a screenshot of the output of:

```bash
git config --list
```
and share with your line manager.
