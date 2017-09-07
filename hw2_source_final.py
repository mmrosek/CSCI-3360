from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
import re
from bs4.element import NavigableString, Tag
import datetime
import urllib
import requests
import scipy.stats as stats
import math
import matplotlib.pyplot as plt

# Function to scrape strings from html 
def scrape_string(restaurant_attribute):
    targets = []
    for target in restaurant_attribute.children:
        if isinstance(target, NavigableString):
            targets.append(target)
        if isinstance(target, Tag):
            targets.extend(scrape_string(target))
    return targets


def scrape_temperatures(fromMonth, fromDay, fromYear, toMonth, toDay, toYear):

    url = 'http://www.georgiaweather.net/index.php?variable=HI&site=WATHORT'
    values = {'fromMonth' : str(fromMonth),
              'fromDay' : str(fromDay),
              'fromYear' : str(fromYear),
              'toMonth' : str(toMonth),
              'toDay' : str(toDay),
              'toYear': str(toYear)}
    data = urllib.parse.urlencode(values)
    data = data.encode('ascii')
    test = urllib.request.Request(url, data)
    final = urlopen(test,data=data)
    
    soup = BeautifulSoup(final, 'lxml')
            
    # Returns the first table in html_content
    table = soup.find_all("table")[1]       
    
    # Returns the column headings
    column_headings = table.find_all("td", attrs={'class':'tdClassAlternate'})
    
    column_heading_list = []
    # Start at 1 b/c first element is not a column heading
    for i in range(1,len(column_headings)):
        column_heading_list.append(scrape_string(column_headings[i])[0])
    
    # Returns number of columns in table
    num_columns = len(column_heading_list)
    
    # Initializes new dataframe
    new_df = pd.DataFrame(columns=range(num_columns), index=[0])
    
    new_df.columns = column_heading_list
    
    # Returns all of the temperature data as well as 5 preceding values that are not needed
    all_data = table.findAll('h5')
    
    # Parses through table and puts stats into dataframe going across each row
    column_marker = 0
    row_marker = 0
    for i in range(5,len(all_data)):
        new_df.ix[row_marker,column_marker] = all_data[i].get_text()
        column_marker += 1
        if column_marker == num_columns:
            row_marker += 1
            column_marker = 0
    
    return(new_df)

def paired_ttest(X,Y,alpha):
    difference = np.array(X) - np.array(Y)
    diff_mean = np.mean(difference)
    diff_std = np.std(difference, ddof=1)
    t_stat = diff_mean/(diff_std/math.sqrt(len(X)))
    cv = stats.t.ppf(0.975, len(X)-1)
    p_val = 1-stats.t.cdf(t_stat, len(X)-1)
    print('T-statistic: ' + str(t_stat) + "\n" +
            "P value: " + str(p_val) + "\n"
            "Critical Value: " + str(cv))
 
y = scrape_temperatures('January', 1, 2016, 'February', 1, 2016)

x = scrape_temperatures('January', 1, 2017, 'February', 1, 2017)

max_temps_2017 = np.array(x.ix[:,1].astype(float))

max_temps_2016 = np.array(y.ix[:,1].astype(float))
 
paired_ttest(max_temps_2017,max_temps_2016,.05)  


########################################################################

### Part 3 ###

# Problem 1 #

def results_by_sites(startYear,endYear):

    for year in range(int(startYear),int(endYear)+1):
        
        html_content = urlopen("http://www.cfbstats.com/{}/team/257/index.html".format(str(year)))
        
        # Create html object
        soup = BeautifulSoup(html_content, "lxml")
        
        # Returns the first table in html_content
        table = soup.find_all("table")[1]
        
        # Returns column headings
        column_headings = table.find_all("th")
        
        # Creates a list containing the column names
        col_names=[]
        for th in column_headings:
            col_names.append(th.get_text())
        
        # Returns number of columns in table
        num_of_columns = len(col_names)
            
        # Initializes new dataframe
        new_df = pd.DataFrame(columns=range(num_of_columns), index=[0])
        
        new_df.columns = col_names
                
        row_marker = 0
        # Parses through table and puts stats into dataframe going across each row
        for row in table.find_all('tr'):
            column_marker = 0
            columns = row.find_all('td')
            for column in columns:
                new_df.ix[row_marker,column_marker] = column.get_text()
                column_marker += 1
                if column_marker == num_of_columns:
                    row_marker += 1
        
        if '@ : Away' in new_df.ix[len(new_df)-1,0]:
            new_df = new_df.ix[:len(new_df)-2,:]
            
        if year == int(startYear):
            cont_table = np.zeros(shape=[2,3])
        
        for row in range(len(new_df)):
            if "+" in new_df.ix[row,'Opponent']:
                if "W" in new_df.ix[row,'Result']:
                    cont_table[0,1]+=1
                elif "L" in new_df.ix[row, "Result"]:
                    cont_table[1,1]+=1
            elif "@" in new_df.ix[row,'Opponent']:
                if "W" in new_df.ix[row,'Result']:
                    cont_table[0,2]+=1
                elif "L" in new_df.ix[row, "Result"]:
                    cont_table[1,2]+=1
            else:
                if "W" in new_df.ix[row,'Result']:
                    cont_table[0,0]+=1
                elif "L" in new_df.ix[row, "Result"]:
                    cont_table[1,0]+=1
        
        test_statistic=0
        for row in range(cont_table.shape[0]):
            for col in range(cont_table.shape[1]):
                test_statistic += (cont_table[row,col] - (np.sum(cont_table[row,:])*np.sum(cont_table[:,col])/np.sum(cont_table)))**2 / (np.sum(cont_table[row,:])*np.sum(cont_table[:,col])/np.sum(cont_table))

        p_value = 1 - stats.chi2.cdf(test_statistic, (cont_table.shape[0]-1)*(cont_table.shape[1]-1))
    
    return({'Contingency Table:':cont_table,'P-value:':p_value})

#a).

results = results_by_sites(2012,2016)

results

#b).

cont_table = results['Contingency Table:']

# Data to plot
labels = 'Home', 'Neutral', 'Away'
sizes = [cont_table[0,0],cont_table[0,1],cont_table[0,2]]
colors = ['lightskyblue', 'yellowgreen', 'lightcoral']
patches, percents, texts  = plt.pie(sizes, colors=colors, shadow=True, startangle=90,autopct='%1.1f%%')
plt.legend(patches, labels, loc="best")
plt.title("Percentage of Wins by Location")
plt.axis('equal')
plt.show()

#c).

degrees_of_freedom = (cont_table.shape[0]-1)*(cont_table.shape[1]-1)

print('Degrees of Freedom: ' + str(degrees_of_freedom))

critical_value = stats.chi2.ppf(0.95, degrees_of_freedom)

print('Critical Value: ' + str(critical_value))

test_statistic=0
for row in range(cont_table.shape[0]):
    for col in range(cont_table.shape[1]):
        test_statistic += (cont_table[row,col] - (np.sum(cont_table[row,:])*np.sum(cont_table[:,col])/np.sum(cont_table)))**2 / (np.sum(cont_table[row,:])*np.sum(cont_table[:,col])/np.sum(cont_table))

p_value = 1 - stats.chi2.cdf(test_statistic, degrees_of_freedom)

print('P-Value: ' + str(p_value))

# Because the P-value is much larger than 0.05, I would not reject the null hypothesis that game results are independent of game sites.
