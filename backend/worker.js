// backend/worker.js

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import chokidar from 'chokidar';
import { Client } from 'pg';
import { analyzeClip } from './analysis.js';

// Pre-deploy guard (must come *after* imports)
if (process.env.RENDER_PRE_DEPLOY) {
  console.log('⚡️ Pre-deploy check, exiting 0');
  process.exit(0);
}

// Wrap everything in a single async function
async function main() {
  const DATA_DIR     = process.env.DATA_DIR     || '/data';
  const DATABASE_URL = process.env.DATABASE_URL;
  if (!DATABASE_URL) {
    console.error('❌ DATABASE_URL not set');
    process.exit(1);
  }

  // Connect to Postgres
  const db = new Client({ connectionString: DATABASE_URL });
  try {
    await db.connect();
    console.log('✅ Connected to Postgres');
  } catch (err) {
    console.error('❌ Failed to connect to Postgres:', err);
    process.exit(1);
  }

  // Ensure /data exists
  if (!fs.existsSync(DATA_DIR)) {
    fs.mkdirSync(DATA_DIR, { recursive: true });
  }

  // DEBUG: list current files in DATA_DIR
  try {
    const files = fs.readdirSync(DATA_DIR);
    console.log('🔍 Files currently in DATA_DIR:', files);
  } catch (e) {
    console.error('⚠️ Could not read DATA_DIR:', e);
  }

  // Watch for new MP4s
  const watcher = chokidar.watch(DATA_DIR, {
    ignoreInitial: true,
    depth: 0,
    awaitWriteFinish: { stabilityThreshold: 2000 }
  });

  watcher.on('add', async (filePath) => {
    if (!filePath.toLowerCase().endsWith('.mp4')) return;
    console.log(`▶ Detected new clip: ${filePath}`);

    try {
      const plays = await analyzeClip(filePath);
      console.log(`↳ Extracted ${plays.length} plays`);

      for (const p of plays) {
        await db.query(
          `INSERT INTO plays
             (team, clip, start_time, end_time, offense_formation,
              defense_formation, blitz, coverage, run_direction,
              pass_type, completed)
           VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)`,
          [
            p.team,
            path.basename(filePath),
            p.startTime,
            p.endTime,
            p.offenseFormation,
            p.defenseFormation,
            p.blitz,
            p.coverage,
            p.runDirection,
            p.passType,
            p.completed
          ]
        );
      }
      console.log(`☑ Wrote ${plays.length} rows to DB`);

      // Move processed clip
      const doneDir = path.join(DATA_DIR, 'processed');
      if (!fs.existsSync(doneDir)) fs.mkdirSync(doneDir, { recursive: true });
      fs.renameSync(filePath, path.join(doneDir, path.basename(filePath)));
      console.log(`✔ Moved clip to /data/processed`);
    } catch (err) {
      console.error('❌ Error processing clip:', err);
    }
  });

  watcher.on('error', err => console.error('Watcher error:', err));

  console.log(`🎬 Worker watching for new clips in ${DATA_DIR}`);

  // Keep the process alive indefinitely
  setInterval(() => {}, 1000 * 60 * 60);
}

// Start the worker
main().catch(err => {
  console.error('Fatal error in worker:', err);
  process.exit(1);
});
