# TCHS Football Film Breakdown Tool

This build includes:

- Persistent MP4 storage under `/data/clips` for Render persistent disks.
- A Games table on the Label Queue page.
- Direct **Label**, **Reset Labels**, and **Delete Game** buttons for every game.
- **Reset Labels** keeps the uploaded videos, removes labels and AI predictions for that game, removes its training events, rebuilds learned counts from the remaining labeled film, and returns the clips to the labeling queue.
- **Delete Game** permanently removes the game's videos, clips, labels, AI predictions, training records, and game statistics.

Deploy this ZIP to the existing Render service so the current PostgreSQL database and persistent disk remain attached.
