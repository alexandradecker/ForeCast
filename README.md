# forecast
Data for each participant is saved in each folder for either behavioral, eye tracking and pupil

##behavioralData
behavioral contains behavioral data for the simon task. This is data from the encoding session, as well as a column for whether each image was remembered or forgotten in the surprise memory task. Column names represent the following:



participant - participant number; each participant has a unique ID

trial_index - trial number during the encoding phase; should range from 1 to 228

screenposition - screen position that stimulus was presented on; either left or right side of screen

stimulus - name of the stimulu sthat was presented.

trialtype- reflects whether the trial was a congruent or incongruent trials.

classifyenc - reflects whether the image was of something natural or manmade

rt - response time of participant on this trial

response - whether participant pressed f or j

responsecorrect - the correct response that the participant should have pressed

responsecorrrrect_side - the side of the screen that the image was presented on

screenLocation - location of image on the screen

stimnum - stimulus number (can ignore)

acc - accuracy; 1 - participant made a correct response; 2 - participant made an incorrect response

rtResidual - response time with the effect of trial number removed

rtResidualC - response time with the effect of trial number removed, mean centered within partipant so that each participants' response time is zero

rt_rec - response time when the participant saw the image at recogniiton and said it was either old (they had seen it before) or new (they hadn't seen it before)

trialno_rec - the trial number that this stimulus was presented in during the recogniton memory phase (anywhere between 1 and 456).

ratingConfidence - how confident the participant was in making a memory judgement (old vs. new). 1 is guessing and 4 is 100% confident

acc_rec - accuracy at recogniition in mkaing an old/new judgement. Since these are all old images, participants ge a 0 (incorrect) if they said the image was new, and 1 if they said the image was old

memoryHits.hc - whether or not participans made a high confidence correct response during recognition; 1 represents images that the participant was very confident that the image was old (a very good memory was formed); the confidence rating here is either a 3 or 4. 


##pupilData

pupil contains the following columns:

Subject - the participant ID; you can change this to 'participant' if/when you need to merge the files with behavioral data

Trial - the trial number

Time - Time relative to when a stimulus was presented; for example, -3 would measure what happened 3 seconds before the simulus came on the screen. 

pupil - pupil size at a given sample


#eyeTrackingData


participant - participant number

trial_index - trial number

right_fix_index - the number of fixations they have made (so far) in a given trial; e.g. if a row has a value for right_fix_index = 2, then the participant has made 2 fixations up to this point in the current trial.

right_gaze_x - the x coordinate of where the participant is looking on the screen

right_gaze_y - the y coordinate of where the participant is looking on the screen

right_in_blink - whether the participant is  blinking during that sample

right_in_saccade - whether the participant is saccading (moving their eyes) on that trial

right_pupil_size - pupil size

right_saccade_index - the number of times the participant has saccaded up until that point in a given trial

sample_index - the sample number (1000 samples per second)

sample_message - contains messages, such as when fixation, or stimulus is on the screen, and when the participant made a key press.

The rest of the columns are the same columns as the behavioral data.







