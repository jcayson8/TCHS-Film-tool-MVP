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

- Uploads MP4 clips sequentially so each completed clip is saved independently.
- Overall progress is calculated from the current clip and completed uploads.
- Failed, timed-out, or interrupted clips are preserved in the Retry Failed Uploads queue.
- Successful clips are not uploaded again when retrying failures.
- The polished Mark as Offense, Mark as Defense, Retry Uploads, and Retry Processing actions remain included.

The browser frontend is the root-level `index.html`. Docker copies it into `/app/public`, while local development serves it directly from the repository root.

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

## Local development

### Prerequisites

- Node.js 20 or newer
- PostgreSQL
- FFmpeg (used by Team Identity processing)

On macOS, these can be installed with Homebrew:

```bash
brew install node@20 postgresql@16 ffmpeg
brew services start postgresql@16
```

On Windows, install Node.js 20, PostgreSQL, and FFmpeg using your preferred package manager or their official installers. Confirm that `node`, `npm`, `psql`, and `ffmpeg` are available in PowerShell before continuing.

### Configure and run

On macOS, create an empty local database using your preferred PostgreSQL role, then install the existing project dependencies:

```bash
createdb tchs_film
npm install
```

Create the ignored local media directory and set the environment variables for the current terminal:

```bash
mkdir -p .local-data/clips
export DATABASE_URL="postgresql://localhost:5432/tchs_film"
export DATA_DIR="$PWD/.local-data"
export PORT=8080
export NODE_ENV=development
```

Start normally or use Node 20's watch mode:

```bash
npm start
# Or: npm run dev
```

In another terminal, verify the API and open the app:

```bash
curl -i http://localhost:8080/api/health
open http://localhost:8080
```

On Windows PowerShell, run the equivalent setup from the repository directory:

```powershell
createdb tchs_film
npm install
New-Item -ItemType Directory -Force .local-data\clips | Out-Null
$env:DATABASE_URL = "postgresql://localhost:5432/tchs_film"
$env:DATA_DIR = (Join-Path (Get-Location) ".local-data")
$env:PORT = "8080"
$env:NODE_ENV = "development"
npm start
# Or: npm run dev
```

Then verify the API and open the app from another PowerShell window:

```powershell
Invoke-WebRequest http://localhost:8080/api/health
Start-Process http://localhost:8080
```

You can copy `.env.example` as a reference, but this project does not load `.env` files automatically. Export the variables in your shell or supply them through your process manager. When `DATA_DIR` is omitted, the server uses `/data` if that directory exists and is writable; otherwise it creates `.local-data/clips` automatically.
