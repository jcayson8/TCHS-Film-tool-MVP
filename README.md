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
