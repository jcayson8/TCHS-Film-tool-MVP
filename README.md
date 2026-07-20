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
