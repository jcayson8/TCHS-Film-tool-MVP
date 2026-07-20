# TCHS Football Film Breakdown Tool

This build includes:

- Persistent MP4 storage under `/data/clips` for Render persistent disks.
- A Games table on the Label Queue page.
- Direct **Label**, **Reset Labels**, and **Delete Game** buttons for every game.
- **Reset Labels** keeps the uploaded videos, removes labels and AI predictions for that game, removes its training events, rebuilds learned counts from the remaining labeled film, and returns the clips to the labeling queue.
- **Delete Game** permanently removes the game's videos, clips, labels, AI predictions, training records, and game statistics.

Deploy this ZIP to the existing Render service so the current PostgreSQL database and persistent disk remain attached.

## Optional AI pre-labeling on upload

The Upload Film page includes an **Automatically pre-label clips with AI** checkbox.

- Checked: the current AI prediction is copied into the editable label fields after upload.
- Unchecked: the clip is uploaded with blank coach-label fields.
- AI-pre-labeled clips remain in the Label Queue and do not count as coach-approved labels or training corrections until a coach reviews and saves them.
- The choice is sent with each upload as `autoLabel=true|false`.

## Upload Film dashboard update
- Dark commercial dashboard styling matching the approved mockup.
- Inline playback for uncertain team-identity clips.
- Marking a clip Offense or Defense automatically advances to the next clip in the active review queue.
- Failed file retry and clip reprocessing controls remain available.


## Upload performance update

- Uploads up to four MP4 clips concurrently instead of waiting for each clip sequentially.
- Overall progress is calculated from actual uploaded bytes across all active clips.
- Failed, timed-out, or interrupted clips are preserved in the Retry Failed Uploads queue.
- Successful clips are not uploaded again when retrying failures.
- The polished Mark as Offense, Mark as Defense, Retry Uploads, and Retry Processing actions remain included.

The concurrency value can be changed in `backend/public/index.html` using `UPLOAD_CONCURRENCY`. Start at 4 on Render; increase to 6 only after confirming the web service remains stable.

## Dynamic Formation Matchups

This build adds a coach-defined formation workflow:

- Offensive and defensive formation fields use team-specific autocomplete suggestions.
- Suggestions are built only from coach-labeled clips; no generic formations are hard-coded.
- New formation names appear automatically after a clip is saved.
- Team Breakdown includes an Offense vs Defense Matchups dashboard showing defensive front, coverage, and man/zone response percentages for every offensive formation.
- Formation Library supports team-specific rename, merge, and delete operations and automatically rebuilds reports.
- Matchup data can be filtered by team and game.


## Offense-only Label Queue

The Label Queue now loads only clips where `film_side = 'offense'` and `status = 'needs_labeling'`. Defensive and unsure clips remain available in Upload Film → Team Identity Review. Correcting an unsure clip to Offense makes it eligible for the Label Queue automatically.

## Label Queue classification gate

The default Label Queue is now **Verified Offense**. Automatically sorted clips do not enter this queue until a coach marks the clip as Offense in Team Identity Review. The queue also includes filters for All Offense, Defense, and All Classified clips.
