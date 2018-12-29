---
title: "Using the Query Explorer for Google Analytics"
date: 2017-08-03T15:40:00-04:00
draft: false
---

Well, you must have heard people talk about this free tool called Google Analytics(GA) and how you can leverage it to gain some useful insights into your business. But how do you wrap your head around around the overwhelming Google Analytics interface and the sea of numbers?

In this blog, we would close that intimidating and confusing UI and get some basics figured about the lingua franca of analytics and data used in GA and try out the simple Query Explorer. Let’s start with **dimensions** and **facts**. 

> **Dimension** is a collection of reference information about a measurable event. 
> And **Facts** are the measurements and metrics.

So every problem around Google analytics can be broken into its constituent dimensions and facts(metrics).
___
Let’s take an example. _“How many people have visited my website over the last few days”_ 

That sounds simple. Right? 
> **Users** can be a relevant metric/fact and **date** can be a dimension

Allright, now lets take another one _“What do the people from Citysville really read in  my blog”_

> If you think hard enough, you can figure out that **City** and **Page Name** are the dimensions and **Screenviews** can be a metric. 

Now that’s simple if I had the list of the entire available dimensions and metrics and I could really search what I need. Great news is that google have already a list with a search bar for you to explore.  Check it out [here](https://developers.google.com/analytics/devguides/reporting/core/dimsmets).
___

![Dimensions and Facts](https://raw.githubusercontent.com/siddhantmaharana/siddhantmaharana.github.io/master/img/dimensions_metrics.PNG "Dimensions and Facts")

___

You can even drill down to check all the extensive list of dimensions and facts.


#### Query Explorer

The Query Explorer is a developer tool that you can leverage to link with your Google Analytics account and get the report right away!
Check it out [here](https://ga-dev-tools.appspot.com/query-explorer/)
___

![Query Explorer](https://raw.githubusercontent.com/siddhantmaharana/siddhantmaharana.github.io/master/img/query_explorer_GA.PNG "Query Explorer")

___

Looks like a simple report generator right? Yes, exactly that’s what it is. First things first. You need to grant access to the tool to access your google analytics account. 

Select the account and the view for which you want to track the metrics. That’s fairly simple. Right?

So lets try to accomplish the above task through this tool

To view our metrics on a daily level we will first enter the start and end date. 
Now if you can decode the task and jot down the metrics and dimensions involved, then you are already halfway! So we need users on a daily level.

> Dimensions: Date<br>
> Metrics: Users

Simple, right?
Let’s leave the other optional fields blank and hit the ‘Run Query’ button. And Voila, the report is ready. 


If you are still unsure about the metrics and dimensions and what do they mean exactly, check out the documentation by google and then you can play around!


