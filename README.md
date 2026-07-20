# TCHS Film Breakdown Complete Repo

## Persistent storage
Uploaded MP4 files are stored on the Render persistent disk mounted at `/data`. The database is PostgreSQL. Code redeploys do not erase either one.

## Game deletion
The labeling page includes **Delete Whole Game**. Select a team and a specific game, then confirm deletion. This removes all clips, labels, predictions, game-specific training records, and video files for that game.
