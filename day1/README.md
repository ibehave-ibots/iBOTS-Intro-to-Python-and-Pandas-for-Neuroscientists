# Day 1: Exploring the Experiment Design with Pandas

In this session we will focus on getting a better understanding of the experiment and the data that was recorded as part of the [Steinmetz et al, 2019](https://www.nature.com/articles/s41586-019-1787-x) Nature paper.

- [**Notebook 1**](1%20Running%20Data%20Pipelines%20With%20Jupyter.ipynb) focuses on downloading the data and packaging it into an easy-to-analyze format
- [**Notebook 2**](2%20Exploring%20Experiment%20Design%20With%20Pandas.ipynb) focuses on exploring the downloaded data using Python tools such as Pandas and Seaborn

---
### Variable description

The data we'll be working with today contains the following variables:

* `'mouse'`: mouse name.
* `'session_date'`: when a session was performed.
* `'session_id'`: unique id for a session.
* `'stim_onset'`: when the stimulus appeared on the screen.
* `'contrast_right'`: contrast level for the right stimulus, which is always contralateral to the recorded brain areas.
* `'contrast_left'`: contrast level for left stimulus.
* `'gocue_time'`: when the go cue sound was played.
* `'response_time'`: when the response was registered, which has to be after the go cue.
* `'response_type'`: which side the response was (`-1`, `0`, `1`). When the right-side stimulus had higher contrast, the correct choice was `-1`. `0` is a no go response.
* `'reaction_type'`: direction of the wheel movement.
* `'reaction_time'`: reaction time computed from the wheel movement.
* `'feedback_type'`: if the feedback was positive (`+1`, reward) or negative (`-1`, white noise burst).
* `'feedback_time'`: when feedback was provided.
* `'active_trials'`: `True` for active trials and `False` for passive trials (i.e. when the mouse was no longer engaged and stopped making responses).