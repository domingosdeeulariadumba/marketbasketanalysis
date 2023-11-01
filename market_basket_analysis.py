# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 11:58:33 2023

@author: domingosdeeularia
"""

import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sb
import warnings
warnings.filterwarnings('ignore')
# %%


# %%

    ''' EXPLORATORY DATA ANALYSIS'''
    
    
    
    '''
    Importing the dataset
    '''
df_ptn = pd.read_csv('Groceries data.csv')


    '''
    Checking the dataset info, statistical summary, first and last five rows
    '''
df_ptn.info()

df_ptn.describe(include = 'all')

df_ptn.head()

df_ptn.tail()


    '''
    We want to understand which products have the highest orders amounts. As it
    is easier to assess visually, we first create a list and then a new dataframe
    with these items.    
    '''
# Creating the top ten items list in terms of order amount
top10item_list = list(df_ptn['itemDescription'
                             ].value_counts()[:10].index)


# Creating the respective dataframe and passing the 'month' column name 
#(slicing until the third letter)
df_top10items = df_ptn[df_ptn['itemDescription'
                              ].isin(top10item_list)]

df_top10items ['month'] = pd.to_datetime(df_top10items['month'],
                                 format='%m').dt.month_name().str[:3]

# Creating the pivot tables for month and year analysis
pivot_table_year = pd.crosstab(df_top10items['itemDescription'], 
                               df_top10items['year'])

pivot_table_month = pd.crosstab(df_top10items['itemDescription'],
                                df_top10items['month'])

    '''
    Displaying the top ten items stacked bar plots
    '''
# Creating a figure with two subplots in y axis
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 11), dpi = 400)
# Plotting the yearly orders stacked bar subplot
pivot_table_year.plot(kind='bar', stacked=True, ax=axes[0])
axes[0].set_title('Top 10 Items Orders', fontsize = 16,
                  fontweight = 'bold')
axes[0].set_xticklabels([])
axes[0].set_xlabel('')
# Plotting the monthly orders stacked bar subplot
pivot_table_month.plot(kind='bar', stacked=True, ax=axes[1])
axes[1].set_xticklabels(pivot_table_month.index, rotation=55, fontsize = 13, 
                        fontweight = 'bold')
axes[1].set_xlabel('')
sb.set_palette('Paired')
# Adjusting the layout to prevent possible overlapping
plt.tight_layout()
# Show the combined figure
plt.show()

    '''
    From the figure above it is noticed that 'whole milk' was the most ordered
    item throughout the two years. Its higher orders amount was in 2015. It is
    also clear that, depending on the month of each year, the total orders
    was around the same quantity    
    '''
# %%

    '''FREQUENT PATTERNS ANALYSIS'''


    '''
    We'll now look for frequent items combinations. It will be created a 
    function to automatize the whole process. That is:
        1. We'll first group the 'Member_number' (which counts as Invoice
        Number) and 'itemDescription' attributes. We'll then unstack this table
        have a relation of all the items by Invoices. The item that was not 
        ordered by a determined client is passed as 'NaN', on the opposite case
        it is passed as 1.
        
        2. We'll next fill the 'NaN' with '0', indicating that this product was
        not ordered by the client associated with the 'Member_number'.
        
        3.The last step will consist on presenting the frequent items 
        combinations for the chosen association rule (we consider here the
        FPGrowth algorithm, but this function is set for any other), given the
        principal parameters according to the business context and rules. 
        
        *In our case, it is hypothesized that the managers looks only for 
        products with at least 35% of representation (min_support) on the whole
        items set. These items need to have a minimum combinations probability 
        of 90% ('Confidence') and at least three more chances of being bought 
        together everytime the antecedent(s) product(s) is/are ordered than 
        when they are independent. 
        To address this request, we'll set the minimum suport and for 4% and
        30% respectively, as we also wish to make a report of our findings to
        the managers even if eventually, no association attends the business
        conditions.
    '''




def freq_items(algorithm, data, InvoicesNo_ColumnName, Items_ColumnName, 
               min_support, confidence):
     
    
    df_items = data.groupby([f'{InvoicesNo_ColumnName}', 
                             f'{Items_ColumnName}'])[f'{Items_ColumnName}'].count(
                                 ).unstack().reset_index(
                                     ).set_index(f'{InvoicesNo_ColumnName}')
        
    # Creating binary entries for items and member number association
    df_Ass_bin = df_items.applymap(lambda x: 0 if x != 1 else 1).astype(int)
        
    # Defining the min_support for the chosen algorithm
    df_Ass = algorithm(df_Ass_bin, min_support = min_support, 
                       use_colnames = True)
    
    # Inserting the confidence treshold and finding the combinations
    df_Ass_metrics_confidence = association_rules(df_Ass,
                                                  metric = 'confidence', 
                                                  min_threshold = confidence)
    
    # Sorting the associations relation in descending order, first by 
    # confidence and then by 'lift'
    output = df_Ass_metrics_confidence.sort_values(['confidence','lift'],
                                             ascending = [False, False])
    
    # Filtering the output metrics just for confidence and lift
    filt_output = output[['antecedents', 'consequents',
                                    'confidence', 'lift']]
    
    
    # Checking the top ten association for the given rule
    top10_filt_output = filt_output.head(10)
    
        
    return top10_filt_output


display('Results:', freq_items(fpgrowth, df_ptn, 'Member_number',
                               'itemDescription', 0.04, 0.30))

      '''
      The results above shows that the found associations may occur more 
      due to comonnality as the highest lift is about 1.15 with a 
      confidence of 37%. Therefore, no cobination of items attended the 
      business conditions to define a sales strategy on the products involved.
      '''
___________________________________END_________________________________________























j = freq_items(fpgrowth, df_ptn, 'Member_number', 'itemDescription', 0.04, 0.30)