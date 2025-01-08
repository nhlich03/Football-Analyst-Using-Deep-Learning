# Football-Analyst-Using-Deep-Learning

## Data 

Football Analyst Using Deep Learning
Data
We use data from SoccerNet. The dataset consists of numerous 30-second labeled video clips from football matches. You can find more information about the dataset [here](https://www.soccer-net.org/tasks/game-state-reconstruction).

To download the data, you can run the following command:
'''
from SoccerNet.Downloader import SoccerNetDownloader
mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory="data/SoccerNetGS")
mySoccerNetDownloader.downloadDataTask(task="gamestate-2024",
                                        split=["train", "valid", "test", "challenge"])
'''


After running this command, you need to unzip the directory. The data structure will look like this:
'''
data/
   SoccerNetGS/
      train/
      valid/
      test/
      challenge/
'''

Once the data is downloaded, you need to run the process_data/process_data.ipynb notebook to perform sampling and extract the labels necessary for the project.
