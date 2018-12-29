---
title: "Getting started in Hugo"
date: 2017-06-19T09:45:12-04:00
draft: false
---


Hugo makes life much easier when it comes to scaffolding a website. Plus the amazing themes. Thanks to the devs out there. Here we will go ahead to create one for us and host it in our favorite Github pages. Follow along and you will have an amazing website to yourself.

#### Create repos in Github
1. Let's call the first one 'blog' - This repo is for building the hugo website and experimenting with various themes.
2. This has to be of the form 'username.github.io' - We would be deploying the final content for the website here.

#### Setting up Hugo in your local system

First let's install Hugo from the official website.

Create a folder in your local system. Let's name it 'blog'. This is the folder we would use for our experimentation in our local computer. So create a directory and navigate inside the folder 

```shell
mkdir blog
cd blog
```

Inside the folder, let's create a hugo site and intiatilze with git init

```shell
hugo new site .
git init
```

Next step is to add our git repo 'blog' and fetch its contents

```shell
git remote add origin https://github.com/siddhantmaharana/blog.git
git pull origin master
```

We won't be needing the 'public' folder. So we will add it in the gitignore for now.

```shell
vim .gitignore
public\
```

Let's push the changes in the exisitng repo and see if it works so far.

```shell
git add .
git status
git commit -m "initial commit"
git push -u origin master
```

#### Adding themes and some more hugo love

Now let's add a theme for our website. The [site](https://themes.gohugo.io/) contains a great deal of curated themes. Let's pick one. Change the directory to the themes and clone the files into this. For my site, I went ahead with [Hugo Geo](https://github.com/alexurquhart/hugo-geo/)

To do that, we will move to the themes directory and clone the theme.

```shell
cd themes
git clone https://github.com/alexurquhart/hugo-geo.git
```

Now let's modify certain lines in the config.toml file. We will update our config file. We will just copy from [here](https://github.com/alexurquhart/hugo-geo/blob/master/exampleSite/config.toml) to our config.toml

#### Creating post and pushing changes

Let's create a new post with the following command

```shell
hugo new post/hello.md
```

Lets also change the base url in the config.toml file to point it at our git website (username.github.io)
We can view the existing changes and website using the following command.

```shell
hugo server -w
```

Next we would clone the other repository into another folder and update it.

```shell
cd ..
git clone https://github.com/siddhantmaharana/siddhantmaharana.github.io.git
git pull origin master
```
Let's copy all contents from 'blog' to the new repository

```shell
hugo -d ../siddhantmaharana.github.io/
```


Now let's update the changes in the other repo

```shell
cd siddhantmaharana.github.io
git add --all
git commit -m "initial commit"
git push origin master
```

And voila.. you have a site hosted in github for free!
