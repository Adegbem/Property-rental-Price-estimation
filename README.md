# Streamlit-based Web Application

## 1) Overview

![Streamlit](resources/imgs/streamlit.png) 

#### 1.1) What is Streamlit?

[![What is an API](resources/imgs/what-is-streamlit.png)](https://youtu.be/R2nr1uZ8ffc?list=PLgkF0qak9G49QlteBtxUIPapT8TzfPuB8)

If you've ever had the misfortune of having to deploy a model as an API (as was required in the Regression Sprint), you'd know that to even get basic functionality can be a tricky ordeal. Extending this framework even further to act as a web server with dynamic visuals, multiple responsive pages, and robust deployment of your models... can be a nightmare. That's where Streamlit comes along to save the day! :star:

In its own words:
> Streamlit ... is the easiest way for data scientists and machine learning engineers to create beautiful, performant apps in only a few hours!  All in pure Python. All for free.

> It’s a simple and powerful app model that lets you build rich UIs incredibly quickly.

Streamlit takes away much of the background work needed in order to get a platform which can deploy your models to clients and end users. Meaning that you get to focus on the important stuff (related to the data), and can largely ignore the rest. This will allow you to become a lot more productive.  

##### Description of files

For this repository instruction, we are only concerned with a single file:

| File Name              | Description                       |
| :--------------------- | :--------------------             |
| `property_rental_app.py`          | Streamlit application definition. |

## 2) Usage Instructions

#### 2.1) Running the Streamlit web app on your local machine

As a first step to becoming familiar with our web app's functioning, we recommend setting up a running instance on your own local machine.

To do this, follow the steps below by running the given commands within a Git bash (Windows), or terminal (Mac/Linux):

 1. Ensure that you have the prerequisite Python libraries installed on your local machine:

 ```bash
 pip install -U streamlit numpy pandas scikit-learn
 ```

 2. Clone the *forked* repo to your local machine.

 ```bash
 git clone https://github.com/{github username}/Property-rental-Price-estimation.git
 ```  

 3. Navigate to the base of the cloned repo, and start the Streamlit app.

 ```bash
 cd classification-predict-streamlit-template/
 streamlit run property_rental_app.py
 ```

 If the web server was able to initialise successfully, the following message should be displayed within your bash/terminal session:

```
  You can now view your Streamlit app in your browser.

    Local URL: http://localhost:8501
    Network URL: http://192.168.43.41:8501
```

You should also be automatically directed to the base page of your web app. This should look something like:

![Streamlit base page](resources/imgs/streamlit-base-splash-screen.png)

Congratulations! You've now officially deployed the web application!

While we leave the exploration of the web app up to you
