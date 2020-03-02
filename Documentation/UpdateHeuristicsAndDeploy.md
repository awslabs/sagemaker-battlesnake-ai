# STEP 2 - Customize the AI heuristics

Using your customised model or using the provided pretrained model, you can add some code to change the AI's movement decision. 

For example, you can calculate if the move will make you collide into a snake body or head with a longer body (in both case you die).

Another one will be detect that you may be able to kill another shorter snake colliding head to head.

_If you do clever things, your pull request is welcome!_

## Architecture

![Heuristic Dev Architecture](images/ArchitectureSagemakerBattleSnakeHeuristics.png "Heuristic Dev Architecture")

> __Estimated cost__ : This environment does not add any cost to the STEP 1 environment. The free tiers include 250 hours per month of this notebook instance on the first two months.
> After the free tiers the charge will be $0.269 per hour for the notebook instance ($6.5 per 24 hour period).
> __Saving tip__ : Once you have finished working you can stop your notebook instance to stop consuming free tiers or occuring charge. You can restart them later to continue your work.

## How to develop your own heuristic algorithms

- Open and run the notebook in `HeuristicDeveloper.ipynb` and ensure that you have a functioning model (if you have altered the inputs model, you may need to configure the inference step in `get_action(*args)`).
- Edit `run` in the class `MyBattlesnakeHeuristics` in `battlesnake_heuristics.py` with your 


## Deploy your own custom snake
TODO