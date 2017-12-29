# Introducing pydqc

pydqc is an automatic data quality check package written in Python. It acts as
an easy-to-use data summary framework, which could, to some extent, relieve us
from the pain of writing tedious codes for general data understanding.

### How does it work?

To make it simple, pydqc accepts a data table as input and provides [data
summary
report](https://github.com/SauceCat/pydqc/blob/master/test/output/data_summary_properties_2016.xlsx)
or [data compare
report](https://github.com/SauceCat/pydqc/blob/master/test/output/data_compare_properties_2016.xlsx).
It also allows you to transform the full process of data summary as well as data
compare into jupyter notebooks. ([data summary
notebook](https://github.com/SauceCat/pydqc/blob/master/test/output/data_summary_notebook_properties_2016.ipynb),
[data compare
notebook](https://github.com/SauceCat/pydqc/blob/master/test/output/data_compare_notebook_properties_2016.ipynb))
The graph below summarizes the workflow of pydqc.

![Process of pydqc](https://cdn-images-1.medium.com/max/800/1*23LPThGFMk3EhLaa3_MSFg.jpeg)

#### Step 1: data schema

To implement data_summary or data_compare, we need to first have **data
schema**, which contains names and data types of columns to be checked. It is
necessary because different types of data require different strategies to
generate meaningful statistical summary. For instance, in terms of categorical
columns, you may want to check the value counts, while for numeric columns you
prefer plotting out the distribution graphs.

> pydqc recognizes four basic data types, including **‘key’, ‘str’, ‘date’, ‘numeric’**.   
> **‘key’:** column that doesn’t have concrete meaning itself, but acts as ‘key’ to link with other tables.  
> **‘str’:** categorical column  
> **‘date’:** datetime column  
> **‘numeric’:** numeric column  

However, the automatic infer_schema function is not yet perfect enough to do
everything correctly. For example, infer_schema function (at least this version)
can’t infer columns with ‘key’ type, columns that don’t have concrete meaning
themselves, but act as ‘key’ to link with other tables. And sometimes it would
infer ‘str’ type as ‘numeric’ type when the categorical column represents its
categories using numeric value. Therefore, the automatically generated data
schema still needs human modification to ensure its correctness. You can easily
do the modification by selecting from a drop down list.

![modify data type with excel drop-down list](https://cdn-images-1.medium.com/max/800/1*uS9NrZWfJyjUQEsItqYTSA.png)  

Examples: [raw
schema](https://github.com/SauceCat/pydqc/blob/master/test/output/data_schema_properties_2016.xlsx)**,**
[modified
schema](https://github.com/SauceCat/pydqc/blob/master/test/output/data_schema_properties_2016_mdf.xlsx).

#### Step 2 (option 1): data summary  

Data summary, as it is named, is to summary useful statistical information for a
data table. As mentioned previously, pydqc classifies data columns into four
types. For each of these four types, data_summary function provides varied
statistical information.

**‘key’ and ‘str’:** sample value, rate of nan values, number of unique values,
top 10 value count.

![example data summary output for ‘key’ and ‘str’ type](https://cdn-images-1.medium.com/max/800/1*Eqdc0LtPIOfAJs20H5rn-w.png)

**‘date’:** sample value, rate of nan values, number of unique values, minimum
numeric value, mean numeric value, median numeric value, maximum numeric value,
maximum date value, minimum date value, distribution graph for numeric values. (
*numeric value for ‘date’ column is calculated as the time difference between
the date value and today in months.*)

![example data summary output for ‘date’ type](https://cdn-images-1.medium.com/max/800/1*EfJ6NnrJYrhv5VQ6_L5G_A.png)

**‘numeric’:** sample value, rate of nan values, number of unique values,
minimum value, mean value, median value, maximum value, distribution graph
(log10 scale automatically when absolute maximum value is larger than 1M).

![example data summary output for ‘numeric’ type](https://cdn-images-1.medium.com/max/800/1*DFIM1SNyNqUJ67VbrmGzJw.png)

![example data summary output for ‘numeric’ type with log10 scale](https://cdn-images-1.medium.com/max/800/1*BS54T3BiiFrBDeceGnFVCg.png)

Examples: [data summary
report](https://github.com/SauceCat/pydqc/blob/master/test/output/data_summary_properties_2016.xlsx),
[data summary
notebook](https://github.com/SauceCat/pydqc/blob/master/test/output/data_summary_notebook_properties_2016.ipynb)

#### Step 2 (option 2): data compare

Sometimes we may want to see whether the test set is different from the training
set and how they are different from each other. And for data set with different
snapshots, it might be useful to check whether data is consistent through
different snapshots. This kind of use cases gives rise to the data_compare
function that compares statistical characteristics of the same columns between
two different tables. The same as data_summary, columns with different types are
compared with varied methods.

**‘key’:** compare sample value, rate of nan values, number of unique values,
size of intersection set, size of set only in table1, size of set only in
table2, venn graph. **(upper value for table1, lower one for table2)**

![example data compare output for ‘key’ type](https://cdn-images-1.medium.com/max/800/1*WQJwHKbXmsfyMTgTVD-pVw.png)

**‘str’:** compare sample value, rate of nan values, number of unique values,
size of intersection set, size of set only in table1, size of set only in
table2, top 10 value counts. **(upper value and blue bar for table1, lower one
and orange bar for table2.)**

![example data compare output for ‘str’ type](https://cdn-images-1.medium.com/max/800/1*wE7q6WaporZK5wA3GPXUjQ.png)

**‘date’:** compare sample value, rate of nan values, number of unique values,
minimum numeric value, mean numeric value, median numeric value, maximum numeric
value, maximum date value, minimum date value, distribution graph for numeric
values. **(upper value and blue line for table1, lower one and orange line for
table2.)**

![example data compare output for ‘date’ type](https://cdn-images-1.medium.com/max/800/1*N1lt-05AZlMq_E6LbpRTpw.png)

**‘numeric’:** compare sample value, rate of nan values, number of unique
values, minimum value, mean value, median value, maximum value,distribution
graph. **(upper value and blue line for table1, lower one and orange line for
table2.)**

![example data compare output for ‘numeric’ type](https://cdn-images-1.medium.com/max/800/1*JGaI9xif785boLyIwQJkjA.png)

Examples: [data compare
report](https://github.com/SauceCat/pydqc/blob/master/test/output/data_compare_properties_2016.xlsx),
[data compare
notebook](https://github.com/SauceCat/pydqc/blob/master/test/output/data_compare_notebook_properties_2016.ipynb)

### GitHub Repo

For more details about **pydqc** package, please refer to
[https://github.com/SauceCat/pydqc](https://github.com/SauceCat/pydqc).
