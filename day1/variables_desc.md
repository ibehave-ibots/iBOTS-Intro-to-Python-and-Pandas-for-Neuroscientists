The data contains 39 sessions from 10 mice, data from [Steinmetz et al, 2019](https://www.nature.com/articles/s41586-019-1787-x). Time bins for all measurements are 10ms, starting 500ms before stimulus onset. The mouse had to determine which side has the highest contrast. The data contains the following variables:

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