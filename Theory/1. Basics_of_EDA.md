# Exploratory Data Analysis

It is a set of processes used to filter the data from redundancies such as missing values, outliers etc. It also helps us to understand the relationship between variables and gives us a wider perspective to build on it.

The following are the steps in EDA:

* **Understand the Data:** Comprehend the structure of the data.
* **Clean the Data:** Remove outliers and fill the missing values.
* **Analyse the reationship between variables:**

## Handling Outliers

Determination of Outlier comes before Missing value since, the outliers can negetively influence the value during Missing Value Imputation. Outliers mean they are below or above 3 standard deviations from the mean. An effective tool to visualize outliers is Box Plot.

**Box Plot:**

![title](https://www.isixsigma.com/wp-content/uploads/2018/11/basic_box_plot-400x190.png)

The Line in the middle of the Box plot shows the median. The median is important as it is not affected by the outliers. 50% of the Data lies above the Median and the other 50% lies below it. The lower Quartile represents 25% of the Data below median and the Upper Quartile represents 25% of the data above median. The lines or whiskers at the end shows exhibits the rest of the data.

![title](https://www.itl.nist.gov/div898/handbook/eda/gif/boxplot0.gif)

The Outliers are represented in the form of dots beyond the box plot. However, not all outliers are bad and hence, care must be taken.

## Missing Value Imputation

Many real-world datasets may contain missing values for various reasons. They are often encoded as NaNs, blanks or any other placeholders. Training a model with a dataset that has a lot of missing values can drastically impact the machine learning model’s quality. One way to handle this problem is to get rid of the observations that have missing data. However, you will risk losing data points with valuable information. A better strategy would be to impute or estimate the missing values such that it does not interfere with the statistics of our dataset. The following are some of the techniques to impute missing value.

* **Imputation using Mean, Median:** This works by calculating the `Mean`/`Median` of the non-missing values in a column and then replacing the missing values within each column separately and independently from the others. It can only be used with numeric non-categorical data. BoxPlot can assist to decide between `Mean` and `Median`. If the data is skewed and there are outliers than mean would give an erroneous measure of central tendency, interfering with the overall distribution of the data. Thus, `mean` should be used only when the data is symmetrically distributed otherwise `Median` gives better result. However, this procedure does not account for correlation between features or probability of uncertainity in determination of the feature.
* **Imputation using Mode:** Imputation is `Mode` wourks best for categorical data. However, it may disturb the distribution of the data.
* **Imputation using Constant:** This technique is almost never a good idea. It drastically effects all the parameters of the distribution.
* **Imputation Using k-NN:** The k nearest neighbours is an algorithm that is used for simple classification. The algorithm uses `feature similarity` to predict the values of any new data points. This means that the new point is assigned a value based on how closely it resembles the points in the training set. This can be very useful in making predictions about the missing values by finding the k’s closest neighbours to the observation with missing data and then imputing them based on the non-missing values in the neighbourhood. It works by creating a basic mean impute then using the resulting complete list to construct a KDTree. Then, it uses the resulting KDTree to compute nearest neighbours (NN). After it finds the k-NNs, it takes the weighted average of them. It can be much more accurate than the mean, median or mode however, it is computationally very expensive because kNN works by storing the whole training dataset in the Memory. Moreover, it is quite sensitive to outliers.
* **Imputation Using Multivariate Imputation by Chained Equation (MICE):** This type of imputation works by filling the missing data multiple times. Multiple Imputations (MIs) are much better than a single imputation as it measures the uncertainty of the missing values in a better way. The chained equations approach is also very flexible and can handle different variables of different data types (ie., continuous or binary) as well as complexities such as bounds or survey skip patterns. Library: `from impyute.imputation.cs import mice`.
* **Imputation Using Deep Learning:** This method works very well with categorical and non-numerical features. It is a library that learns Machine Learning models using Deep Neural Networks to impute missing values in a dataframe. Library: `import datawig`
* **Stochastic regression imputation:** It is quite similar to regression imputation which tries to predict the missing values by regressing it from other related variables in the same dataset plus some random residual value.
* **Extrapolation and Interpolation:** It tries to estimate values from other observations within the range of a discrete set of known data points.
* **Hot Deck Imputation:** Works by randomly choosing the missing value from a set of related and similar variables.