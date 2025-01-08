# Football-Analyst-Using-Deep-Learning

## Introduction
The goal of this project is detect players, goalkeepers, referees and ball in a video using YOLO. We will train model on custom data to improve performance. 
Additionally, we will assign players to teams based on the colors of their t-shirts using Kmeans for pixel segmentation and clustering. With this information, we can measure a team's ball acquisition percentage in a match.

## Data 

Football Analyst Using Deep Learning
Data
We use data from SoccerNet. The dataset consists of numerous 30-second labeled video clips from football matches. You can find more information about the dataset [here](https://www.soccer-net.org/tasks/game-state-reconstruction).

To download the data, you can run the following command:
```
from SoccerNet.Downloader import SoccerNetDownloader
mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory="data/SoccerNetGS")
mySoccerNetDownloader.downloadDataTask(task="gamestate-2024",
                                        split=["train", "valid", "test", "challenge"])
```


After running this command, you need to unzip the directory. The data structure will look like this:

```
data/
   SoccerNetGS/
      train/
      valid/
      test/
      challenge/
```

Once the data is downloaded, you need to run the [process_data/process_data.ipynb](process_data/process_data.ipynb) notebook to perform sampling and extract the labels necessary for the project.


## Train Model 

Chúng tôi sử dụng model YOLOv8x để train 