---
title: "How to work with Google Analytics API"
date: 2017-10-19T09:45:12-04:00
draft: false
---


In this notebook, we would use the Google Analytics API to fetch the data from our Google Analytics account. The resultant csv can subsequnetly be written to a preferred database.

For getting data from google analytics, we need to set up a project in our google developers account
and obtain a keyfile (in json format).

The module below builds a service object and takes various parameters such as api_name, api_version, scope, key_file_location,
and service_account_email




```python
def get_service(api_name, api_version, scope, key_file_location,service_account_email):
    credentials = ServiceAccountCredentials.from_json_keyfile_name(key_file_location,scope)
    http = credentials.authorize(httplib2.Http())
    #Build service object
    service = build(api_name, api_version, http=http)
    return service
```

The module below receives several parameters which are the conditions for fetching our data


```python
def ga_api_reporting(service,ga_id,begin_date,finish_date,\
metrics_list,dim_list,seg_con,sort_con,filer_con,index,max_results):
    return service.data().ga().get(
        ids=ga_id,
        start_date=begin_date,
        end_date=finish_date,
        metrics=metrics_list,
        dimensions=dim_list,
        segment=seg_con,
        sort=sort_con,
        filters=filer_con,
        start_index=index,
        max_results=max_results
    ).execute()
```

In this example, we would extract data related to screenviews. As parameter we would be passing the following-

1. service: The service object created above
2. ga_id: Your Google analytics account id
3. begin_date: The start date from which the data we want to extract
4. finish_data: The end date till which we want to obtain our data
5. metrics_list: The list of metrics for which we want to get our data. In this case (ga:screenviews)
6. dim_list: The list of dimensions. We are fetching the screenviews with dates and screenname as the dimensions
7. seg_con: Any specific segment defined for which we want to get data. For eg: Mobile only users
8. sort_con: The metrics on which we want to sort our data. '-ga:screenviews' gives us the data in decreasing order of the screenviews.
9. filer_con: Various filter conditions can be defined here. We dont want the screens without any views. So 'ga:screenViews!=0'
10. index: Starting index of the results. We are looping the results by providing the starting index in multiples of 1000, i.e 1000 results everytime.
11. max_results: Total row limit can be specified here


```python
def app_screen_views(service,start_dt,end_dt):
    for i in range(1,10000,1000):
        screen_views_data = ga_api_reporting(service,'ga:999999',\
        start_dt,end_dt,'ga:screenviews','ga:date,ga:screenName',None,\
        '-ga:screenviews','ga:screenViews!=0',str(i),str(1000))
        extract_rows(screen_views_data)
```

We have to write the results obtained into a csv file. We check the results for rows and write the output to the 'screenviews.csv'.



```python
def extract_rows(results):

    output_file = open('screenviews.csv', 'ab')
    output_file_obj = csv.writer(output_file)
    if results.get('rows', []):
        for row in results.get('rows'):
            output = []
            for cell in row:
                output.append(cell)
            output_file_obj.writerows([output])
```

Now the next module contains the main function in which we pass the following-

1. scope: Analytics v3 of Google Analytics
2. service_account_email: The email we obtain after creating a project in Google Developer console
3. key_file_location: The key file in JSON format obtained from the Google Developer console.
4. date: In this example we extract for last 10 days.


```python

def main():
    scope = ['https://www.googleapis.com/auth/analytics.readonly']
    service_account_email = 'id-mg-app-reporting@my-project-1470392440259.iam.gserviceaccount.com'
    key_file_location = 'key_file.json'
    service = get_service('analytics', 'v3', scope, key_file_location,service_account_email)
    date_format = str(10)+"daysAgo"
    app_screen_views(service,str(date_format),str(date_format))


```

#### Complete Code



```python
## Importing Libraries
import argparse
from apiclient.discovery import build
from oauth2client.service_account import ServiceAccountCredentials
import httplib2
from oauth2client import client
from oauth2client import file
from oauth2client import tools
import csv
import os
import datetime

def get_service(api_name, api_version, scope, key_file_location,service_account_email):
    credentials = ServiceAccountCredentials.from_json_keyfile_name(key_file_location,scope)
    http = credentials.authorize(httplib2.Http())
    #Build service object
    service = build(api_name, api_version, http=http)
    return service

def ga_api_reporting(service,ga_id,begin_date,finish_date,metrics_list,\
dim_list,seg_con,sort_con,filer_con,index,max_results):
    return service.data().ga().get(
        ids=ga_id,
        start_date=begin_date,
        end_date=finish_date,
        metrics=metrics_list,
        dimensions=dim_list,
        segment=seg_con,
        sort=sort_con,
        filters=filer_con,
        start_index=index,
        max_results=max_results
    ).execute()

## Need to change the GA account 
def app_screen_views(service,start_dt,end_dt):
    for i in range(1,10000,1000):
        screen_views_data = ga_api_reporting(service,'ga:999999',start_dt,\
        end_dt,'ga:screenviews','ga:date,ga:screenName',None,'-ga:screenviews'\
        ,'ga:screenViews!=0',str(i),str(1000))
        extract_rows(screen_views_data)
        
def extract_rows(results):

    output_file = open('screenviews.csv', 'ab')
    output_file_obj = csv.writer(output_file)
    if results.get('rows', []):
        for row in results.get('rows'):
            output = []
            for cell in row:
                output.append(cell)
            output_file_obj.writerows([output])
            
## Need to change the service_account_email and other params
def main():
    scope = ['https://www.googleapis.com/auth/analytics.readonly']
    service_account_email = 'id-mg-app-reporting@my-project-9999999.iam.gserviceaccount.com'
    key_file_location = 'key_file.json'
    service = get_service('analytics', 'v3', scope, key_file_location,service_account_email)
    date_format = str(10)+"daysAgo"
    app_screen_views(service,str(date_format),str(date_format))



if __name__ == '__main__':
    main()


```
