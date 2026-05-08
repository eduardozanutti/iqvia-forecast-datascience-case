# Case: Weekly Demand Forecasting

## Problem Description

You have received a dataset containing the units of pharmaceutical products sold weekly. Your goal is to build a predictive model to forecast product demand by distributor and geographic region of the stores responsible for sales.

The data provided for this case will likely fit in your computer's memory. However, it would be interesting if you developed your solution seeking efficiency in terms of memory and processing.

## Delivery Format

The development artifacts of the solution can be delivered in a compressed file, or shared via link to a code repository (i.e. GitHub, GitLab, Bitbucket, AzureDevops, etc). In case of sharing a repository link, ensure that the project is public so we can access your solution. You can develop your solution in notebooks (.ipynb) or scripts (.py).

The presentation of results must be delivered in the form of a slide presentation saved in .ppt format.

## Tasks

### 1. Data Preparation
- Load and verify data integrity.
- Handle missing values and data cleaning, if necessary.
- Encode categorical variables.
- Normalize or standardize numerical variables, if necessary.
- Create new variables that may be useful for the model.

### 2. Exploratory Analysis
- Descriptive analysis to understand the main characteristics of the data.
- Create visualizations to identify patterns and seasonal trends in demand.
- Describe any relevant insights obtained during the analysis.

### 3. Model Building and Evaluation
- Variable selection.
- Cross-validation.
- Choose and justify the selection of a modeling technique.
- Choose and justify the selection of an evaluation metric.

## Data Dictionary

- **week_dt**: date corresponding to the first day of the week.
- **dsupp_id**: distributor identification code for the product.
- **product_id**: product identification code.
- **region_nm**: IBGE macro-region where the store, in which the product was sold, is located.
- **units_qty**: quantity of units sold in the week.
- **product_attr_1, product_attr_2, and product_attr_3**: specific product attributes.