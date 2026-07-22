TCHS Football Film Breakdown Tool — V1.0 Combined Update

Replace these ROOT-LEVEL files in:
jcayson8/TCHS-Film-tool-MVP

1. server.js
2. index.html

Included:
- Team Review to Label Queue handoff fix
- Offense, Defense, or both queue selection
- One-clip-at-a-time reliable batch uploading
- Per-file upload status and Retry Failed
- Refresh/close warning during active uploads
- Upload form and previous-session restoration
- Removed separate Offensive Tendencies tab
- Removed separate Defensive Tendencies tab
- Removed separate Formation Matchups tab
- Merged all analytics into Team Breakdown
- Team Breakdown views: Overview, Offensive Breakdown,
  Defensive Breakdown, Formation Matchups, Situational Breakdown
- Opponent and game filters inside Team Breakdown
- Formation analytics populated from saved coach labels
- Dark mode preserved

Suggested commit message:
Merge V1 analytics and fix upload queue workflow

After committing, wait for Render to redeploy.
