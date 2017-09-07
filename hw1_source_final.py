### Part 2 ###

from bs4 import BeautifulSoup
from bs4.element import NavigableString, Tag
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import collections
import plotly.plotly as py

# Function to scrape strings from html 
def scrape_string(restaurant_attribute):
    targets = []
    for target in restaurant_attribute.children:
        if isinstance(target, NavigableString):
            targets.append(target)
        if isinstance(target, Tag):
            targets.extend(scrape_string(target))
    return targets
    

def scrape_web_page(soup):

    ############################################
    
    # Scrapes prices
    
    # Finds the price of each restaurant
    prices = soup.findAll('span', attrs={'class':'price_range'})
    
    # Creates a list of prices
    price_list = []
    for price in prices:
        price_list.append(scrape_string(price)[1])
    
    ############################################
    
    # Scrapes the rating 
    
    ratings = soup.findAll('img', attrs={'class':'sprite-ratings'})
    
    ratings_list = []
    for child in ratings:
        if isinstance(child, Tag):
            ratings_list.append(child.get('alt', ''))
        else:
            ratings_list.append(child.strip())
    
    #############################################
    
    # Scrapes number of reviews
    
    reviews = soup.find_all('span', attrs={'class':'reviewCount'})
    
    review_list = []
    for review in reviews:
        review_list.append(int(scrape_string(review)[1].strip().split()[0]))
    
    #############################################
    
    # Scrapes names
    
    names = soup.find_all('h3', attrs={"class":'title'})
    
    name_list = []
    for name in names:
        name_list.append(scrape_string(name)[1].strip())
    
    #############################################  
    
    # Scrapes photos
    
    photos = soup.find_all("div", attrs={"onclick" : lambda L: L and L.startswith("ta.trackEventOnPage")})
    
    photo_list = []
    for i in range(len(photos)):
        if i == 0:
            continue
        photo_list.append(int(scrape_string(photos[i])[0].strip().split()[2].strip("()")))
       
    ###############################################
    
    # Scrapes cuisine
    
    cuisines = soup.find_all('div', attrs={'class':'cuisines'})
    
    cuisine_list =[]
    for i in range(len(cuisines)):
        if len(scrape_string(cuisines[i])) == 5:
            cuisine_list.append(scrape_string(cuisines[i])[3])
        elif len(scrape_string(cuisines[i])) == 7:
            cuisine_list.append(scrape_string(cuisines[i])[3] + ", " + scrape_string(cuisines[i])[5])
        elif len(scrape_string(cuisines[i])) == 9:
            cuisine_list.append(scrape_string(cuisines[i])[3] + ", " + scrape_string(cuisines[i])[5] + ", " + scrape_string(cuisines[i])[7])
        elif len(scrape_string(cuisines[i])) == 11:
            cuisine_list.append(scrape_string(cuisines[i])[3] + ", " + scrape_string(cuisines[i])[5] + ", " + scrape_string(cuisines[i])[7] + ", " + scrape_string(cuisines[i])[9])
        elif len(scrape_string(cuisines[i])) == 13:
            cuisine_list.append(scrape_string(cuisines[i])[3] + ", " + scrape_string(cuisines[i])[5] + ", " + scrape_string(cuisines[i])[7] + ", " + scrape_string(cuisines[i])[9] + ", " + scrape_string(cuisines[i])[11])
        else:
            cuisine_list.append(scrape_string(cuisines[i])[3] + ", " + scrape_string(cuisines[i])[5] + ", " + scrape_string(cuisines[i])[7] + ", " + scrape_string(cuisines[i])[9] + ", " + scrape_string(cuisines[i])[11] + ", " + scrape_string(cuisines[i])[13])
         
    ##############################################
      
    return(cuisine_list,photo_list,name_list,ratings_list,price_list,review_list)

final_cuisine_list=[]
final_photo_list=[]
final_name_list=[]
final_ratings_list=[]
final_price_list=[]
final_review_list=[]

for i in range(3):
    if i == 0:
        html_content = urlopen("https://www.tripadvisor.com/Restaurants-g29209-Athens_Georgia.html")
        b_soup = BeautifulSoup(html_content, 'lxml')
        scraped_contents = scrape_web_page(b_soup)
        final_cuisine_list.extend(scraped_contents[0])
        final_photo_list.extend(scraped_contents[1])
        final_name_list.extend(scraped_contents[2])
        final_ratings_list.extend(scraped_contents[3])
        final_price_list.extend(scraped_contents[4])
        final_review_list.extend(scraped_contents[5])
    if i == 1:
        html_content = urlopen("https://www.tripadvisor.com/Restaurants-g29209-oa30-Athens_Georgia.html")
        b_soup = BeautifulSoup(html_content, 'lxml')
        scraped_contents = scrape_web_page(b_soup)
        final_cuisine_list.extend(scraped_contents[0])
        final_photo_list.extend(scraped_contents[1])
        final_name_list.extend(scraped_contents[2])
        final_ratings_list.extend(scraped_contents[3])
        final_price_list.extend(scraped_contents[4])
        final_review_list.extend(scraped_contents[5])
    if i == 2:
        html_content = urlopen("https://www.tripadvisor.com/Restaurants-g29209-oa60-Athens_Georgia.html")
        b_soup = BeautifulSoup(html_content, 'lxml')
        scraped_contents = scrape_web_page(b_soup)
        final_cuisine_list.extend(scraped_contents[0])
        final_photo_list.extend(scraped_contents[1])
        final_name_list.extend(scraped_contents[2])
        final_ratings_list.extend(scraped_contents[3])
        final_price_list.extend(scraped_contents[4])
        final_review_list.extend(scraped_contents[5])

### Missing Values

# Price: The National Restaurant

# Photos: Larry's Giant Subs, Chick-fil-A

# Removes unwanted description
numerical_ratings = []
for rating in final_ratings_list:
    numerical_ratings.append(float(rating.split()[0]))
    
# Converts $ into numerical value
numerical_prices = []
for price in final_price_list:
    num_price = (price.count('$'))
    if num_price > 4:
        numerical_prices.append(num_price/2)
    else:
        numerical_prices.append(num_price)

food_df = pd.DataFrame(columns=['Name','Cuisine','Number_of_Photos','Price','Rating','Number_of_Reviews'],index=list(range(len(final_name_list))))

food_df['Name']=np.array(final_name_list)
food_df['Cuisine']=np.array(final_cuisine_list)
food_df['Rating']=np.array(numerical_ratings)
food_df['Number_of_Reviews']=np.array(final_review_list)

price_index = 0
for row in range(len(food_df)):
    if food_df.ix[row,'Name']=='The National Restaurant':
        continue
    else:
        food_df.ix[row,'Price'] = numerical_prices[price_index]
        price_index += 1
        
photo_index = 0
for row in range(len(food_df)):
    if (food_df.ix[row,'Name'] == "Larry's Giant Subs") | ((food_df.ix[row,'Name'] == 'Chick-fil-A') & (row < 60)):
        food_df.ix[row,'Number_of_Photos'] = 0
    else:
        food_df.ix[row,'Number_of_Photos'] = final_photo_list[photo_index]
        photo_index += 1


print(food_df)

### Problem 1

#A). Max number of reviews
food_df.Name[np.argmax(food_df.Number_of_Reviews)]

#B). 5 Number Summary

from IPython.display import display

# Reviews
display(food_df.Number_of_Reviews.describe())

# Ratings
display(food_df.Rating.describe())

#C).

plt.hist(food_df.Number_of_Photos, normed=True)
plt.title('Number of Photos')
plt.show()

plt.hist(food_df.Number_of_Reviews, normed=True)
plt.title('Number of Reviews')
plt.show()

plt.hist(food_df.Rating, normed=True)
plt.title('Rating')
plt.show()

#D). Pie Chart

import plotly.plotly as py
from plotly.graph_objs import *
py.sign_in('iLuvPython', 'w7ELl48xtF7Z7uAPIaJb')

# Extracts list of unique cuisine types
cuisine_list = []
for cuisine in set(food_df.Cuisine.unique()):
    for i in range(len(cuisine.split(','))):
        if cuisine.split(',')[i].strip() not in cuisine_list:
            cuisine_list.append(cuisine.split(',')[i].strip())

# Creates dictionary with key-value pairs corresponding to cuisine types and number of occurrences
cuisine_dict = collections.defaultdict(int)
for cuisine in cuisine_list:
    for row in range(len(food_df)):
        if cuisine in str(food_df.ix[row,'Cuisine']):
            cuisine_dict[cuisine] += 1

# Creates usable lists for pie chart generating function
cuisine_pie_keys = []
for key in cuisine_dict.keys():
    cuisine_pie_keys.append(key)

cuisine_pie_values = []
for value in cuisine_dict.values():
    cuisine_pie_values.append(value)
    
data = {
    'data': [{'labels': cuisine_pie_keys,
              'values': cuisine_pie_values,
              'type': 'pie'}],
    'layout': {'title': "Cuisine Types of Athens' Restaurants"}
     }

py.iplot(data, filename = 'basic-line')

#E). QQ Plots (answer in next cell)

import scipy.stats as stats

stats.probplot(food_df.Rating, dist="norm", plot=plt)
plt.title('QQ Plot of Rating')
plt.show()

stats.probplot(food_df.Number_of_Reviews, dist="norm", plot=plt)
plt.title('QQ Plot of Reviews')
plt.show()

stats.probplot(food_df.Number_of_Photos, dist="norm", plot=plt)
plt.title('QQ Plot of Photos')
plt.show()

#F. Correlation

# Pre-processing
food_df.Number_of_Photos = food_df.Number_of_Photos.astype(int)
food_df.Number_of_Reviews = food_df.Number_of_Reviews.astype(int)

display('Correlation between Rating and Number of Photos:',food_df.Rating.corr(food_df.Number_of_Photos))

display('Correlation between Number of Reviews and Number of Photos:',food_df.Number_of_Photos.corr(food_df.Number_of_Reviews))

display('Correlation between Rating and Number of Reviews:',food_df.Number_of_Reviews.corr(food_df.Rating))

### Part 3 ###

from scipy import stats  
import numpy as np  
import matplotlib.pyplot as plt

### Creates function that samples randomly from Chi-Sq distribution and plots histogram with overlayed gaussian
def sample_and_plot(n):
    
    # Creates random samples and calculates means
    i=0
    sample_mean_list = []
    while i < 1000:
        sample_mean_list.append(np.mean(np.random.chisquare(2,n)))
        i+=1
        
    sample_mean_array = np.array(sample_mean_list)
    
    # Plots histogram
    plt.hist(sample_mean_array, bins=50, normed=True)
    
    # Finds minimum and maximum of ticks to calculate interval
    ticks = plt.xticks()[0]  
    mintick, maxtick = min(ticks), max(ticks)  
    interval = np.linspace(mintick, maxtick, len(sample_mean_array))
    
    # Returns mean and std. deviation
    mean, stdev = stats.norm.fit(sample_mean_array) 
    norm_pdf = stats.norm.pdf(interval, mean, stdev)   
    plt.title(r'$\mathrm{Histogram\ of\ Chi-Square\ Samples,\ n=%s,}\ \mu=%.3f,\ \sigma=%.3f$' %(n,mean, stdev))
    plt.plot(interval, norm_pdf, label="Norm")
    plt.show()
