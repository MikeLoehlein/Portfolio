
# coding: utf-8

# In[ ]:


'''
Introduction

This Python script scrapes the beer advocate website and downloads selected information for the beers listed on the first pages 
(current 4) of each style. Each style page has about 50 links, for an estimated 200 (max) links per beer style. 

For each of the beers scraped, it saves the picture of the beer, the beer ratings, beer information, and names/company which 
produces the beer. The script then produces high level summaries of the information which it scrapes. 
'''


# In[24]:


#Import Libraries
from urllib.request import Request, urlopen, urlretrieve, URLopener
from urllib.error import HTTPError
from urllib.error import URLError
from bs4 import BeautifulSoup
import re
import pandas as pd
import csv

import logging
import traceback
import os


# In[2]:


#User Defined Functions:

#Get Style Links
def Beer_Style_Links():
    '''
    This function scrapes all of the 'beer style' links from Beer Advocate's style page. It then saves those 
    links to a list which is returned. 
    
    This function takes no parameters.
    '''
    try:
        #Open the site and load the BeautifulSoup object
        site= "https://www.beeradvocate.com/beer/style/"
        hdr = {'User-Agent': 'Mozilla/5.0'}
        req = Request(site,headers=hdr)
        html = urlopen(req)
        bsObj = BeautifulSoup(html.read(), "lxml")
    except Exception as e:
        #Save any errors to the log
        logger.error(str(e))
        logger.error(str(site))
    else:
        #Process the BeautifulSoup object and create a list of beer sylte links
        names = bsObj.find("div", {"id":"ba-content"}).findAll("a", href=re.compile("^(/beer/style/)((?!:).)*$"))
        links = []

        for i in range(len(names)-1):
            links.append(names[i].attrs["href"])

        return links
    
#Get Beer Links
def Beer_Links(link, depth, beer_links):
    '''
    This function takes a beer style link and saves all of the beer profile links from the page to a list. it is set
    so that it will go to the 'next page' of the beers for that style up to the 'depth' number of pages. 
    If depth = 3 then it will scrape the first three pages of the beer style. The function returns a list
    of beer links.
    
    Parameters:
        link = is the beer style link which should be scraped
        depth = is the number of pages the fucntion should scrape for beer links
        beer_links = is the list which function should append the scrapped beer links to.
    '''
    page = link
    j = 0
    
    while j < depth:
        try:
            #Open the site and load the BeautifulSoup object
            site= "https://www.beeradvocate.com" + str(page)
            hdr = {'User-Agent': 'Mozilla/5.0'}
            req = Request(site,headers=hdr)
            html = urlopen(req)
            bsObj = BeautifulSoup(html.read(), "lxml")
        except Exception as e:
            #Save any errors to the log
            logger.error(str(e))
            logger.error(str(site))
        else:
            #Process the BeautifulSoup object and create append
            # the beer links to the list
            names = bsObj.find("div", {"id":"ba-content"}).findAll("a", href=re.compile("^(/beer/profile/[0-9]+/[0-9]+)((?!:).)*$"))

            for i in range(len(names)-1):
                beer_links.append(names[i].attrs["href"])

        #get the next page link(s) from the page
        next_link = bsObj.find("div", {"id":"ba-content"}).findAll("a")

        next_page = []
        for h in range(len(next_link)-1):
            if next_link[h].getText() == 'next':
                next_page.append(next_link[h])

        #if there are multiple links with the text next, take the first one.
        if len(next_page) > 1:
            next_page = next_page[0].attrs["href"]
        page = next_page
        j += 1

#Get Beer Information
def Beer_Info(page, writer):
    '''
    This function scrapes the selected information for a specific beer profile and saves it to a .csv file. It also
    saves the image to a BeerImages folder which the user needs to create prior to running the code. 
    
    The function scrapes the:
        - beer name
        - beer stats
        - beer information
        - beer score
    sections of the beer profile.
    
    Parameters
        - page = the beer page to be scraped
        - writer = the writer object which the output should be saved to.
    '''
    try:
        #Open the site and load the BeautifulSoup object
        site= "https://www.beeradvocate.com" + page
        hdr = {'User-Agent': 'Mozilla/5.0'}
        req = Request(site,headers=hdr)
        html = urlopen(req)
        bsObj = BeautifulSoup(html.read(), "lxml")
    except Exception as e:
        #Save any errors to the log
        logger.error(str(e))
        logger.error(str(site))
    else:
        #Process the BeautifulSoup object
        try:
            #Save the image to disk
            image = bsObj.find("div", {"id":"info_box"}).find('img')['src']
            name = bsObj.find("div", {"id":"info_box"}).find('img')['alt']
            name = re.sub(r'[^a-zA-Z0-9]+','', name)

            req = Request(image,headers=hdr)
            resource = urlopen(req)
            output = open("BeerImages/" + name +".jpg","wb")
            output.write(resource.read())
            output.close() 
        except Exception as e:
            #Save any errors to the log
            logger.error(str(e))
            logger.error(str(image))

        #Scrape the selecte data from the webpage
        item = {}
        item['name'] = re.sub('\n|\t', '||', bsObj.find("h1").getText())
        item['score'] = re.sub('\n|\t', '||', bsObj.find("div", {"id":"score_box"}).getText())  
        item['stats'] = re.sub('\n|\t', '||', bsObj.find("div", {"id":"stats_box"}).getText())  
        item['info'] =  re.sub('\n|\t', '||', bsObj.find("div", {"id":"info_box"}).getText())
        dta = pd.Series(item, name='item')
        dta = dta.str.encode('utf-8') 
        
        writer.writerow(dta)

#Error Logging
def error_log():
    '''
    This function creates an error logger which is used to track the exceptions caught by the function. 
    It saves a text file in the folder which contains the source code.
    '''
    global logger
    logger = logging.getLogger("File Log")
    logger.setLevel(logging.ERROR)
    handler = logging.FileHandler("log.txt", mode='a', encoding=None, delay=False)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)        
        
#Execution Function
def BeerAdvocateScrap(limit=True):
    '''
    This function executes the BeerAdvocate.com scrapping job.
    
    Paramters:
        Limit = Should the function only scrape the first three style links to test (if True) or all styles (if False). 
                Default = True.
    '''
    
    #Create directory for beer images
    file_path = "/BeerImages"
    directory = os.path.dirname(file_path)

    try:
        os.stat(directory)
    except:
        os.mkdir(directory) 

    
    #Create error log
    error_log()
    
    #Get all style links
    style_links = Beer_Style_Links()
    print(str(len(style_links)) + " Total Style Links") #print total styles
    
    #Get beer links
    if limit:
        beer_links = []
        print("Limited to first 3 styles")
        for i in range(4):
            Beer_Links(style_links[i], 4, beer_links)
    else:
        beer_links = []
        print("Collecting all Styles")
        for i in range(len(style_links)-1):
            Beer_Links(style_links[i], 4, beer_links)

    print(str(len(beer_links)) + " Total Beer Links")

    #Create csv file to save scrapped data in.
    csvFile = open("BeerInformation.csv",'w', newline='')
    #Scrape beer links
    try:
        writer = csv.writer(csvFile)
        for i in range(len(beer_links)-1):
            if i % 50 == 0:
                print("Link Number: " + str(i))
            Beer_Info(beer_links[i], writer)    
    finally:
        #Close csv file.
        csvFile.close()



# In[4]:


#Execute the Scrapping job
BeerAdvocateScrap(limit=False)


# In[ ]:


'''Below is the data cleaning process along with high level data summarization'''


# In[3]:


####Import the data
data = pd.read_csv('BeerInformation.csv', names=(1,2,3,4))
print(type(data))

print()
print(data.head())

print()
print(data.columns)

#Subset each column for indiviual processing
c1 = data.loc[:,1]
c2 = data.loc[:,2]
c3 = data.loc[:,3]
c4 = data.loc[:,4]

#Print top five row of each series
print()
print(c1[:5])

print()
print(c2[:5])

print()
print(c3[:5])

print()
print(c4[:5])


# In[33]:


#Clean C1: Beer Info

#Standardize deliminters
regex_pat = re.compile(r'([||]+)') 
c1_1 = c1.str.replace(regex_pat,'||') 

#Split the text  by delimiter into columns
c1_2 = [p.split('||') for p in c1_1.values]

#Cast as a dataframe
df = pd.DataFrame(c1_2)

#select only required columns
df1 = df[[3,5,6,7,9]]
print(df1.columns)

#Clean each column
regex_pat = re.compile(r'(Style:)')
df1.loc[:,5] = df1.loc[:,5].str.replace(regex_pat,'')

regex_pat = re.compile(r'([Alcohol by volume (ABV): %])')
df1.loc[:,6] = df1.loc[:,6].str.replace(regex_pat,'')

regex_pat = re.compile(r'(Availability:)')
df1.loc[:,7] = df1.loc[:,7].str.replace(regex_pat,'')

#rename columns
df1.columns = ['company', 'style', 'abv', 'availability', 'notes']
print(df1.head(2))


# In[5]:


#Clean C2: Beer Name

#Standardize deliminters
regex_pat = re.compile(r'([||]+)')
c2_1 = c2.str.replace(regex_pat,'||')

#Clean data
regex_pat = re.compile(r'(b\')')
c2_1 = c2_1.str.replace(regex_pat,'')

regex_pat = re.compile(r'(")')
c2_1 = c2_1.str.replace(regex_pat,'')

#Split the text  by delimiter into columns

c2_1 = [p.split('||') for p in c2_1.values]

#Cast as a dataframe
df2 = pd.DataFrame(c2_1)

#select only required columns
df2 = df2[[0,1]]

#rename columns
df2.columns = ['beer_name', 'brewery']

print(df2.head())


# In[6]:


#Clean C3: Beer Rating

#Standardize deliminters
regex_pat = re.compile(r'([||]+)')
c3_1 = c3.str.replace(regex_pat,'||')

#Clean each column
regex_pat = re.compile(r'(b\')')
c3_1 = c3_1.str.replace(regex_pat,'')

regex_pat = re.compile(r'([/5])')
c3_1 = c3_1.str.replace(regex_pat,'')

regex_pat = re.compile(r'(Ratings)')
c3_1 = c3_1.str.replace(regex_pat,'')

regex_pat = re.compile(r'([,])')
c3_1 = c3_1.str.replace(regex_pat,'')

#Split the text  by delimiter into columns
c3_1 = [p.split('||') for p in c3_1.values]

#Cast as a dataframe
df3 = pd.DataFrame(c3_1)

#select only required columns
df3 = df3[[2,3,4]]

#rename columns
df3.columns = ['rating','rating_cat', 'number_rating']

print(df3.head())


# In[7]:


#Clean C2: Beer Stats

#Standardize deliminters
regex_pat = re.compile(r'([||]+)')
c4_1 = c4.str.replace(regex_pat,'||')

#Clean each column
regex_pat = re.compile(r'(b\')')
c4_1 = c4_1.str.replace(regex_pat,'')

regex_pat = re.compile(r'([#,%])')
c4_1 = c4_1.str.replace(regex_pat,'')

#Split the text  by delimiter into columns
c4_1 = [p.split('||') for p in c4_1.values]

#Cast as a dataframe
df4 = pd.DataFrame(c4_1)

#select only required columns
df4 = df4[[3,5,7,9,11,15,17,19]]

#rename columns
df4.columns = ['ranking','reviews', 'ratings','pdev','bro_score','wants','gots','trade']

print(df4.head())


# In[8]:


#Merge the four dataframes and cast columns to numeric as appropriate
df = pd.concat([df1, df2, df3, df4], axis=1)

#Cast numeric columns to numeric from string
df['abv'] = pd.to_numeric(df['abv'], errors='coerce')
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df['number_rating'] = pd.to_numeric(df['number_rating'], errors='coerce')
df['ranking'] = pd.to_numeric(df['ranking'], errors='coerce')
df['reviews'] = pd.to_numeric(df['reviews'], errors='coerce')
df['ratings'] = pd.to_numeric(df['ratings'], errors='coerce')
df['pdev'] = pd.to_numeric(df['pdev'], errors='coerce')
df['bro_score'] = pd.to_numeric(df['bro_score'], errors='coerce')
df['wants'] = pd.to_numeric(df['wants'], errors='coerce')
df['gots'] = pd.to_numeric(df['gots'], errors='coerce')
df['trade'] = pd.to_numeric(df['trade'], errors='coerce')


print(df.head(2))
print()

print(df.columns)
print()

print(df.shape)
print()

print(df.dtypes)


# In[9]:


'''Summaries'''


# In[11]:


#Counts by Style
counts = df['style'].value_counts()

print("top 20 beer styles by beer count")
print(counts[:20])

print()
print("bottom 20 beer styles by beer count")
print(counts[-20:])

'''
By stype, the top beers appear to all have the same general count. This is expected as the web scrapper only took 
the first four pages of the style. The depth of the scrape would need to be set deeper to determine the beer with the 
most beer styles.

For the beers with the least styles however, just scrapping the first four pages did provide enough information. 
The beer styles with the least beers listed include :
 - Faro, Happoshu, Sahti, Black and Tan, and Eisbock
'''


# In[13]:


#Counts by Company
counts = df['company'].value_counts()

print('Top 20 companies by beer count')
print(counts[:20])

'''
Of the beers listed on the first four pages of the styles, Bostom Beer Company has the most listed.
'''

print()
print('Bottom 20 companies by beer count')
print(counts[-20:])

'''
There are many companies with only one beer listed.
'''


# In[14]:


#Counts by Availability
counts = df['availability'].value_counts()

print('Beer Availability')
print(counts)

'''
Most beers scrapped are offered year round. A large number of beers are offered on a rotating basis.
'''


# In[18]:


#Counts by Rating Category
counts = df['rating_cat'].value_counts()

print("Beer Ratings (categorical)")
print(counts)

counts = df['rating_cat'].value_counts(normalize=True)

'''
Only 204 (1%) beers are reated as world class. 50% of beers are rated as good/very good.
'''

print()
print("Beer Ratings (categorical)")
print(counts)


# In[37]:


#Summary stats by abv
print("Summary of Alcohol By Volume")
print(df['abv'].describe())


# In[19]:


#Summary stats by rating
print("Summary by Rating (out of 5)")
print(df['rating'].describe())

'''
The mean rating is 3.6 and the median is 3.74 out of 5.
'''


# In[20]:


#Summary stats by number of ratings
print('Summary of the number of ratings')
print(df['number_rating'].describe())

'''
The mean number of ratings is 226 per beer while the median is only 38. The number of ratings is very scewed with the 
beer with the max number of ratings having over 16K ratings.
'''


# In[21]:


#Summary stats by abv
print("Summary of number of Reviews")
print(df['reviews'].describe())

'''
The number of reviews is right skewed. The mean is higher than the median. Most beers get only a handful of reviews (17) or
less though a handful get a lot (over 100 or even 1000)
'''


# In[20]:


#Summary stats by rating
print("Summary of Ratings")
print(df['ratings'].describe())

'''
The number of reviews is right skewed. The mean is higher than the median. Most beers get only a handful of reviews (55) or
less though a handful get a lot (over 200 or even 1000)
'''


# In[22]:


#Weighted Average Rating by Style
grouped = df.groupby('style')

def wavg(group):
    d = group['rating']
    w = group['ratings']
    return (d * w).sum() / w.sum()

wa = grouped.apply(wavg).sort_values(ascending=False)

print("Top 20 Styles")
print(wa[:20])

print()
print("Bottom 20 Styles")
print(wa[-20:])

'''
The top rated styles have on average a score of 4 or more. While the lowest rated styles have a rating of 3 or less.

The top rated styles are Imperial Stout/Gueze, Imperial IPA, and Wild Ale.
'''


# In[23]:


#Weighted Average Rating by Company
grouped = df.groupby('company')

def wavg(group):
    d = group['rating']
    w = group['ratings']+.00000000000000000000000001
    return (d * w).sum() / w.sum()

wa = grouped.apply(wavg).sort_values(ascending=False)

print("Top 20 Company")
print(wa[:20])

print()
print("Bottom 20 Company")
print(wa[-20:])

'''
The top rate companies all appear to be craft breweries as non of the main brands appear in the top 20. For the 
lowest rated companies, they all have 0 ratings.
'''

