---
title: "Google analytics on Google Spreadsheets"
date: 2017-09-04T08:30:14-04:00
draft: false
---

Don’t you love excel or google spreadsheet for that matter. The formulas and flexibilities. Too much freedom to enjoy. What if we can link our google analytics data with our very own spreadsheet and our sheets would be updated automatically. Every morning, one glance and we are straight into the zone. No clutter! Read on to know how you can get that under your belt.

I hope you are familiar with the terms such as dimensions and metrics which you can leverage to get the data in your very own sheets.  If not check out my [first post](#) to get a fair idea about these.

So lets move into a fresh spreadsheet and install a add on that lets you do half the task. Head over to the Add on tabs in your spreadsheet and install the Google Analytics add on into your account.

___

![Excel Add-on](https://raw.githubusercontent.com/siddhantmaharana/siddhantmaharana.github.io/master/img/excel_add_on.png "Excel Add-on")
___

And don’t forget to grant access to the same if it prompt you with a message. Restart the browser to let google ingest the changes. Now if you hover over to the add ons on your spreadsheet you will be able to find the google analytics tab.  

Christen your report with a name, input the account information and type in some metrics and dimensions relevant to your problem (which we discussed in our previous post).
Lets try a simple task to generating a report to check the daily split of our new and returning users in our website.
First exercise, get the relevant dimensions and metrics pertinent to our problems. 

> Dimensions: Date<br>
> Metrics: New Users, Users

Now that we have all these , lets add **last 30 days** in our date range columns and  hit **Create Report**. And there it is. A sheet named Report Configuration appears from nowhere and has the same input fields we saw in our beta query explorer. Use the same Add on tab to find the **Run Report** button residing inside the Google analytics option.
And there it is. The report is generated in a new sheet. It beautifully lists down the data we need under the results section and we will head over to another sheet and use our excel expertise to get the data we need.

In the new sheet lets input our data from the report generated sheet and use vlookups to get the other fields populated. Flex your excel muscles to populate other metrics and in a jiffy we are ready with our own customized report in spreadsheet. 

Let’s automate it so that it shows the latest data from out Google analytics. Head over to google analytics tab and schedule the report to run everyday. 
Simple.. Aint it? And beautiful indeed.

