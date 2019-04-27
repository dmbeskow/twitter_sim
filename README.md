# twitter_sim

This is a simple Agent Based Model (ABM) for modeling disinformation in Twitter.  It attempt to explicitly model the structure and rules of Twitter as well as the behavior of typical Twitter Users and then Measure the change in beliefs over time.  This type of ABM allows user to insert Bots/Trolls and explore the various disinformation forms of maneuver that we observe empirically.  

### Installation

In order to install the package, unzip the zip file of the package, navigate to the package parent directory (should contain a file called `setup.py`), and then run the following Command:

```
pip install --user .
```

### Running from `twitter_sim` package

Once installed, the user can run a single run of the simulation with the Command

```
# Run one iteration of the simulation with
# Normal means random following
# Polarized means two polarized communities
import twitter_sim
all_beliefs,total_tweets, G = twitter_sim.run(size = 100,
                                  perc_bots = 0.12,
                                  strategy = 'normal',
                                  polarized = 'polarized')
```

Both `twitter_sim_eda.py` and `twitter_sim_eda.ipynb` are provided as templates for running and exploring small simulations.  The following commands can provide visualization of the results:


```
# Draw network Diagram
%matplotlib inline
twitter_sim.draw_simulation(G)

# Area Plot of Type of Tweets over Time
twitter_sim.draw_tweet_timeline(total_tweets)

# Bar Plot of Type of Tweets
twitter_sim.draw_tweet_bar(total_tweets)

# Mean Belief over Time
twitter_sim.draw_beliefs(all_beliefs)
```

### Running from Command Line

In order to execute multiple runs from the command line with larger networks, use the file `run_twitter_sim.py`.  An example command to run from command line interface is

```
# Commands for backing with random and targeted
python3 run_sim.py -size 1000 -perc 0.05 -runs 12 -strategy normal -polarized normal
python3 run_sim.py -size 1000 -perc 0.05 -runs 12 -strategy targeted -polarized normal

# Commands for bridging with random and targeted
python3 run_sim.py -size 1000 -perc 0.05 -runs 12 -strategy normal -polarized polarized
python3 run_sim.py -size 1000 -perc 0.05 -runs 12 -strategy targeted -polarized polarized
```
