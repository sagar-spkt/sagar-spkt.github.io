---
title: 'Defeating The Size: Working with Large Tabular Data on AWS S3 using Snowpark'
date: 2023-09-10
permalink: /posts/2023/09/s3-snowflake-snowpark/
excerpt: "We'll explore the synergy between AWS S3, Snowpark, and Snowflake for efficient handling of large tabular data. By combining these tools, you can seamlessly process and analyze extensive datasets stored in AWS S3."
tags:
  - AWS S3
  - Snowflake
  - Snowpark
  - Big Data
  - ETL
---

Welcome to this blog post where we'll dive into a powerful combination of tools for working with large tabular data: AWS S3, Snowpark and Snowflake. These tools, when used in tandem, enable seamless processing and analysis of large datasets stored in AWS S3. In this blog, we'll walk through a sample code that leverages these technologies to work with the classic Iris dataset. This code can be easily adapted to handle much larger datasets, making it a valuable addition to your data engineering and machine learning toolkit.

## Setting the Stage

Before we get started, let's set up our environment. We'll be using Python along with some essential libraries.

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
```

The first block of code imports the necessary libraries. `numpy` and `pandas` are widely used for numerical computations and data manipulation, while `load_iris` is a convenient function for loading the Iris dataset.

## Loading and Preparing the Data

Next, for demo purpose, we load the Iris dataset and convert it into a Pandas DataFrame.

```python
iris = load_iris()
df = pd.DataFrame(iris["data"], columns=iris["feature_names"])
df["target"] = iris["target"]
df.head()
```
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
      <th>petal length (cm)</th>
      <th>petal width (cm)</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>

Here, we've converted the Iris data into a structured DataFrame, making it easier to work with. This DataFrame includes features like sepal length, sepal width, petal length, petal width, and a target variable.

## Uploading Data to AWS S3

Now, let's upload our DataFrame to AWS S3.

```python
import boto3

# You can use your custom bucket and prefixes
bucket, key = "sagemaker-lal-stage", "iris.csv"
s3_client = boto3.client("s3")
s3_client.put_object(
    Bucket=bucket,
    Key=key,
    Body=df.to_csv(index=False),
    ContentType="text/csv",
)
```

By executing this code, we're placing the data in an S3 bucket, making it accessible for further processing. You can upload your data of any size to S3 any method. Just note the bucket name and prefixes of the data files.

## Establishing a Connection with Snowflake

The next step involves connecting to Snowflake using Snowpark, a powerful tool for data processing in Snowflake.

```python
from snowflake.snowpark import Session

connection_parameters = {
    "account": "####SF_ACCOUNT_NAME######",
    "user": "#####USER#######",
    "password": "#####PASSWORD#######",
    "warehouse": "#####WAREHOUSE########",
    "database": "#####DATABASE########",
    "schema": "#####SCHEMA(PUBLIC)########",
}

session = Session.builder.configs(connection_parameters).create()
```

We establish a connection using session instantiated with the specified parameters. This allows us to interact with the Snowflake database.

## Creating a Temporary Table

Now, let's create a temporary table in Snowflake where we'll load our data from S3.

```python
table_name = "iris_dataset"
session.sql(f"""create temporary table {table_name} (
    SEPAL_LENGTH float,
    SEPAL_WIDTH float,
    PETAL_LENGTH float,
    PETAL_WIDTH float,
    TARGET integer
)"""
).collect()
```

This code creates a temporary table in Snowflake with the same structure as our DataFrame. This is where we'll be loading our data. Temporary table are destroyed when our session is terminated. In my experience, temporary table are really fast in comparison to standard table or transient table available in snowflake. So, it is preferable to work on temporary table if we don't need data persisting on Snowflake.

## Copying Data from S3 to Snowflake

With our temporary table in place, let's copy the data from S3 into Snowflake.

```python
session.sql(f"""copy into {table_name}
from 's3://{bucket}/{key}'
credentials=( AWS_KEY_ID='#######AWS_KEY_ID#######' AWS_SECRET_KEY='#######AWS_SECRET_KEY#######')
file_format=(TYPE=CSV COMPRESSION=NONE SKIP_HEADER=1)
"""
).collect()
```

This command transfers the data from S3 to our Snowflake table using the specified credentials(which have permission to get, put and delete object in AWS S3). All files with prefix `{key}` in bucket `{bucket}` are processed by the above command. Note that while uploading data to s3, we uploaded with header. So, we tell Snowflake to ignore header with `SKIP_HEADER=1`. Also, no compression was done in the uploaded data. You can mention compression type if your data is compressed in any way.

## Analyzing Data with Snowpark

Now that our data is in Snowflake, we can perform any operation we like. We'll be using Snowpark's DataFrame capabilities for this.

```python
sdf = session.table(table_name)
```

This line creates a Snowpark DataFrame from our Snowflake table.

We can display the table by changing the Snowpark DataFrame to Pandas Dataframe.
```python
sdf.to_pandas()
```
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SEPAL_LENGTH</th>
      <th>SEPAL_WIDTH</th>
      <th>PETAL_LENGTH</th>
      <th>PETAL_WIDTH</th>
      <th>TARGET</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>145</th>
      <td>6.7</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>146</th>
      <td>6.3</td>
      <td>2.5</td>
      <td>5.0</td>
      <td>1.9</td>
      <td>2</td>
    </tr>
    <tr>
      <th>147</th>
      <td>6.5</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>148</th>
      <td>6.2</td>
      <td>3.4</td>
      <td>5.4</td>
      <td>2.3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>149</th>
      <td>5.9</td>
      <td>3.0</td>
      <td>5.1</td>
      <td>1.8</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>150 rows × 5 columns</p>
</div>

## Computing Pairwise Correlations

For demo, let's start by computing pairwise correlations between sepal length and sepal width. For reference, the formula of Pearson Correlation is:
$$r=\frac{\sum ( x_{i} -\overline{x})( y_{i} -\overline{y})}{\sqrt{\sum ( x_{i} -\overline{x})^{2}\sum ( y_{i} -\overline{y})^{2}}}$$

```python
from snowflake.snowpark import DataFrame, Column
from snowflake.snowpark import functions as spf

def pair_correlation(df: DataFrame, x: str, y: str) -> DataFrame:
    # broadcast mean using `over`
    x_diff = df[x] - spf.mean(df[x]).over()
    y_diff = df[y] - spf.mean(df[y]).over()
    
    # Store results in columns
    df = df.with_columns(
        ["x_diff", "y_diff"],
        [x_diff, y_diff],
    )
    
    numerator = spf.sum(df["x_diff"]*df["y_diff"])
    denominator = spf.sqrt(spf.sum(spf.pow(df["x_diff"], 2))*spf.sum(spf.pow(df["y_diff"], 2)))
    
    # prepare dataframe with pair values in first two columns and correlation value in last column
    return df.select(
        spf.lit(x).alias("FEAT1"),
        spf.lit(y).alias("FEAT2"),
        (numerator/denominator).alias("VALUE"),
    )
```

Here, we define a function `pair_correlation` that accepts the Snowpark Dataframe and two columns name whose correlation is to be determined, and it will return a new dataframe with results.  This function leverages Snowpark's powerful functions for data manipulation.


```python
pair_correlation(sdf, "SEPAL_LENGTH", "SEPAL_WIDTH").to_pandas()
```
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FEAT1</th>
      <th>FEAT2</th>
      <th>VALUE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>SEPAL_LENGTH</td>
      <td>SEPAL_WIDTH</td>
      <td>-0.11757</td>
    </tr>
  </tbody>
</table>
</div>

You can continue with other analysis as you require.

## Wrapping Up

With the analysis complete, we close our Snowflake session.

```python
session.close()
```

Additionally, we clean up our S3 bucket by deleting the uploaded data.

```python
s3_client.delete_object(
    Bucket=bucket,
    Key=key,
)
```

And there you have it! We've walked through the steps of loading data into Snowflake from AWS S3, conducting various correlation analyses, and finally, cleaning up our environment.

This powerful combination of Snowpark and Snowflake opens up a world of possibilities for handling large tabular datasets. Whether you're a data scientist or a machine learning engineer, having these tools in your arsenal can significantly enhance your data processing capabilities. Experiment with your own datasets and unlock valuable insights!

Happy coding!