import express from 'express';
import multer from 'multer';
import path from 'path';
import fs from 'fs';
import cors from 'cors';
import { fileURLToPath } from 'url';
import { Pool } from 'pg';
import {
  learnFromCorrection,
  predictClip,
  savePrediction,
  rebuildModelCounts
} from './analysis.js';
import { classifyPossession } from './teamIdentity.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const app = express();

const frontendCandidates = [
  path.join(__dirname, 'public'),
  __dirname
];
const publicDir = frontendCandidates.find((directory) =>
  fs.existsSync(path.join(directory, 'index.html'))
);
if (!publicDir) {
  console.error(
    `Frontend entry file not found. Expected index.html in ${frontendCandidates.join(' or ')}`
  );
  process.exit(1);
}

app.use(cors());
app.use(express.json({ limit: '4mb' }));

function defaultDataDir() {
  if (process.env.DATA_DIR) return path.resolve(process.env.DATA_DIR);
  if (process.platform !== 'win32') {
    const persistentDataDir = path.join(path.parse(__dirname).root, 'data');
    try {
      if (fs.statSync(persistentDataDir).isDirectory()) {
        fs.accessSync(persistentDataDir, fs.constants.W_OK);
        return persistentDataDir;
      }
    } catch (_error) {
      // Fall back to repository-local storage when /data is unavailable.
    }
  }
  return path.join(__dirname, '.local-data');
}

const DATA_DIR = defaultDataDir();
const CLIP_DIR = path.join(DATA_DIR, 'clips');
fs.mkdirSync(CLIP_DIR, { recursive: true });

const DATABASE_URL = process.env.DATABASE_URL;
if (!DATABASE_URL) {
  console.error('DATABASE_URL is required');
  process.exit(1);
}

const db = new Pool({
  connectionString: DATABASE_URL,
  ssl: process.env.NODE_ENV === 'production' ? { rejectUnauthorized: false } : undefined
});

const FILM_SIDES = new Set(['offense', 'defense', 'needs_review']);
const CLIP_STATUSES = new Set(['needs_labeling', 'labeled', 'skipped']);
const QUEUE_STATUSES = new Set(['not_queued', 'queued']);
const FORMATION_SIDES = new Set(['offense', 'defense']);
const AI_SERVICE_URL = String(process.env.AI_SERVICE_URL || 'http://127.0.0.1:8000').replace(/\/+$/, '');
const configuredDetectionTimeout = Number(process.env.AI_DETECTION_TIMEOUT_MS);
const AI_DETECTION_TIMEOUT_MS = Number.isFinite(configuredDetectionTimeout)
  ? Math.min(120000, Math.max(1000, Math.round(configuredDetectionTimeout)))
  : 60000;
const DEFENSIVE_OBJECT_CLASSES = Object.freeze([
  'defensive_end',
  'defensive_tackle',
  'middle_linebacker',
  'inside_linebacker',
  'outside_linebacker',
  'cornerback',
  'safety',
  'football',
  'official'
]);
const DATASET_STATUSES = new Set(['draft', 'active', 'archived', 'ready']);
const REVIEW_STATUSES = new Set(['draft', 'reviewed', 'verified']);
const ANNOTATION_READINESS_TARGETS = Object.freeze({
  annotatedFrames: 500,
  verifiedFrames: 250,
  boxesPerClass: 100
});

const text = (value) => String(value ?? '').trim();
const lower = (value) => text(value).toLowerCase();
const nullableText = (value) => {
  const cleaned = text(value);
  return cleaned || null;
};
const safeName = (name) => String(name || 'clip.mp4').replace(/[^a-zA-Z0-9._-]/g, '_');
const pct = (numerator, denominator) => denominator
  ? Math.round((Number(numerator) / Number(denominator)) * 1000) / 10
  : 0;
const normalizeBoolean = (value) => {
  if (typeof value === 'boolean') return value;
  if (value === null || value === undefined || value === '') return null;
  const normalized = lower(value);
  if (['true', '1', 'yes', 'on'].includes(normalized)) return true;
  if (['false', '0', 'no', 'off'].includes(normalized)) return false;
  return null;
};
const normalizeInteger = (value) => {
  if (value === '' || value === null || value === undefined) return null;
  const number = Number(value);
  return Number.isInteger(number) ? number : null;
};
const normalizeDate = (value) => {
  const candidate = text(value);
  return /^\d{4}-\d{2}-\d{2}$/.test(candidate) ? candidate : null;
};
const positiveId = (value, label = 'ID') => {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed <= 0) {
    throw Object.assign(new Error(`${label} must be a positive integer`), { statusCode: 400 });
  }
  return parsed;
};
const optionalInteger = (value, label, minimum, maximum) => {
  if (value === '' || value === null || value === undefined) return null;
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < minimum || parsed > maximum) {
    throw Object.assign(new Error(`${label} must be between ${minimum} and ${maximum}`), { statusCode: 400 });
  }
  return parsed;
};
const boundedNumber = (value, label, { positive = false } = {}) => {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed < 0 || parsed > 1 || (positive && parsed === 0)) {
    throw Object.assign(
      new Error(`${label} must be ${positive ? 'greater than 0 and ' : ''}between 0 and 1`),
      { statusCode: 400 }
    );
  }
  return parsed;
};
const countBy = (rows, key, predicate = () => true) => {
  const matching = rows.filter(predicate);
  const counts = new Map();
  for (const row of matching) {
    const value = text(row[key]) || 'Unlabeled';
    counts.set(value, (counts.get(value) || 0) + 1);
  }
  return [...counts.entries()]
    .map(([name, count]) => ({ name, count, percentage: pct(count, matching.length) }))
    .sort((a, b) => b.count - a.count || a.name.localeCompare(b.name));
};

const storage = multer.diskStorage({
  destination: (_req, _file, cb) => cb(null, CLIP_DIR),
  filename: (_req, file, cb) => cb(
    null,
    `${Date.now()}-${Math.random().toString(36).slice(2, 8)}-${safeName(file.originalname)}`
  )
});

const upload = multer({
  storage,
  limits: { fileSize: 2 * 1024 * 1024 * 1024, files: 100 },
  fileFilter: (_req, file, cb) => {
    const valid = file.mimetype === 'video/mp4' || lower(file.originalname).endsWith('.mp4');
    cb(valid ? null : new Error('Only MP4 files are allowed'), valid);
  }
});

const detectionUpload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 12 * 1024 * 1024, files: 1 },
  fileFilter: (_req, file, cb) => {
    const allowed = new Set(['image/jpeg', 'image/png', 'image/webp']);
    if (!allowed.has(lower(file.mimetype))) {
      const error = Object.assign(new Error('Use a JPEG, PNG, or WebP image'), { statusCode: 415 });
      return cb(error);
    }
    cb(null, true);
  }
});

const boundedUnitOption = (value, label) => {
  if (value === undefined || value === null || value === '') return null;
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed < 0.01 || parsed > 1) {
    throw Object.assign(new Error(`${label} must be between 0.01 and 1`), { statusCode: 400 });
  }
  return parsed;
};

async function fetchAiService(pathname, options = {}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), AI_DETECTION_TIMEOUT_MS);
  try {
    return await fetch(`${AI_SERVICE_URL}${pathname}`, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timeout);
  }
}

async function initDb() {
  await db.query(`
    CREATE TABLE IF NOT EXISTS clips (
      id SERIAL PRIMARY KEY,
      team TEXT NOT NULL,
      game_name TEXT NOT NULL DEFAULT '',
      opponent TEXT,
      game_date DATE,
      season TEXT,
      original_name TEXT NOT NULL,
      stored_name TEXT NOT NULL UNIQUE,
      file_size BIGINT NOT NULL DEFAULT 0,
      status TEXT NOT NULL DEFAULT 'needs_labeling',
      queue_status TEXT NOT NULL DEFAULT 'not_queued',
      film_side TEXT NOT NULL DEFAULT 'needs_review',
      possession_confidence INTEGER,
      possession_reason TEXT,
      review_notes TEXT,
      jersey_color TEXT,
      helmet_color TEXT,
      home_away TEXT,
      use_for_ai BOOLEAN NOT NULL DEFAULT TRUE,
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      labeled_at TIMESTAMPTZ
    );
  `);

  const clipMigrations = [
    `ALTER TABLE clips ADD COLUMN IF NOT EXISTS opponent TEXT;`,
    `ALTER TABLE clips ADD COLUMN IF NOT EXISTS game_date DATE;`,
    `ALTER TABLE clips ADD COLUMN IF NOT EXISTS season TEXT;`,
    `ALTER TABLE clips ADD COLUMN IF NOT EXISTS queue_status TEXT NOT NULL DEFAULT 'not_queued';`,
    `ALTER TABLE clips ADD COLUMN IF NOT EXISTS film_side TEXT NOT NULL DEFAULT 'needs_review';`,
    `ALTER TABLE clips ADD COLUMN IF NOT EXISTS possession_confidence INTEGER;`,
    `ALTER TABLE clips ADD COLUMN IF NOT EXISTS possession_reason TEXT;`,
    `ALTER TABLE clips ADD COLUMN IF NOT EXISTS review_notes TEXT;`,
    `ALTER TABLE clips ADD COLUMN IF NOT EXISTS jersey_color TEXT;`,
    `ALTER TABLE clips ADD COLUMN IF NOT EXISTS helmet_color TEXT;`,
    `ALTER TABLE clips ADD COLUMN IF NOT EXISTS home_away TEXT;`,
    `ALTER TABLE clips ADD COLUMN IF NOT EXISTS use_for_ai BOOLEAN NOT NULL DEFAULT TRUE;`
  ];
  for (const migration of clipMigrations) await db.query(migration);

  await db.query(`
    CREATE TABLE IF NOT EXISTS plays (
      id SERIAL PRIMARY KEY,
      clip_id INTEGER UNIQUE REFERENCES clips(id) ON DELETE CASCADE,
      team TEXT NOT NULL,
      game_name TEXT NOT NULL DEFAULT '',
      clip TEXT NOT NULL,
      down INTEGER,
      distance INTEGER,
      hash TEXT,
      play_type TEXT,
      play_call TEXT,
      offense_formation TEXT,
      defense_formation TEXT,
      blitz BOOLEAN,
      coverage TEXT,
      run_direction TEXT,
      pass_depth TEXT,
      completed BOOLEAN,
      yards_gained INTEGER,
      notes TEXT,
      label_source TEXT NOT NULL DEFAULT 'coach',
      confidence REAL,
      ai_hash TEXT,
      ai_defense_formation TEXT,
      ai_blitz BOOLEAN,
      ai_coverage TEXT,
      ai_run_direction TEXT,
      ai_confidence REAL,
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
  `);

  const playMigrations = [
    `ALTER TABLE plays ADD COLUMN IF NOT EXISTS yards_gained INTEGER;`,
    `ALTER TABLE plays ADD COLUMN IF NOT EXISTS ai_hash TEXT;`,
    `ALTER TABLE plays ADD COLUMN IF NOT EXISTS ai_defense_formation TEXT;`,
    `ALTER TABLE plays ADD COLUMN IF NOT EXISTS ai_blitz BOOLEAN;`,
    `ALTER TABLE plays ADD COLUMN IF NOT EXISTS ai_coverage TEXT;`,
    `ALTER TABLE plays ADD COLUMN IF NOT EXISTS ai_run_direction TEXT;`,
    `ALTER TABLE plays ADD COLUMN IF NOT EXISTS ai_confidence REAL;`
  ];
  for (const migration of playMigrations) await db.query(migration);

  await db.query(`
    CREATE TABLE IF NOT EXISTS ai_model_counts (
      team TEXT NOT NULL DEFAULT '',
      game_name TEXT NOT NULL DEFAULT '',
      feature_key TEXT NOT NULL,
      target TEXT NOT NULL,
      target_value TEXT NOT NULL,
      sample_count INTEGER NOT NULL DEFAULT 0,
      updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      PRIMARY KEY (team, game_name, feature_key, target, target_value)
    );
  `);

  await db.query(`
    CREATE TABLE IF NOT EXISTS ai_predictions (
      clip_id INTEGER PRIMARY KEY REFERENCES clips(id) ON DELETE CASCADE,
      model_version TEXT NOT NULL,
      prediction JSONB NOT NULL,
      overall_confidence REAL,
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
  `);

  await db.query(`
    CREATE TABLE IF NOT EXISTS ai_training_events (
      id BIGSERIAL PRIMARY KEY,
      clip_id INTEGER REFERENCES clips(id) ON DELETE SET NULL,
      team TEXT NOT NULL,
      game_name TEXT NOT NULL DEFAULT '',
      labels JSONB NOT NULL,
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
  `);

  await db.query(`
    CREATE TABLE IF NOT EXISTS ai_datasets (
      id BIGSERIAL PRIMARY KEY,
      name TEXT NOT NULL,
      description TEXT,
      team TEXT,
      target_type TEXT NOT NULL DEFAULT 'defense_detection',
      status TEXT NOT NULL DEFAULT 'draft'
        CHECK (status IN ('draft', 'active', 'archived', 'ready')),
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS ai_dataset_clips (
      dataset_id BIGINT REFERENCES ai_datasets(id) ON DELETE CASCADE,
      clip_id BIGINT REFERENCES clips(id) ON DELETE CASCADE,
      added_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      PRIMARY KEY (dataset_id, clip_id)
    );

    CREATE TABLE IF NOT EXISTS ai_annotation_frames (
      id BIGSERIAL PRIMARY KEY,
      dataset_id BIGINT NOT NULL REFERENCES ai_datasets(id) ON DELETE CASCADE,
      clip_id BIGINT NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
      frame_time_ms INTEGER NOT NULL CHECK (frame_time_ms >= 0),
      frame_width INTEGER NOT NULL CHECK (frame_width > 0),
      frame_height INTEGER NOT NULL CHECK (frame_height > 0),
      review_status TEXT NOT NULL DEFAULT 'draft'
        CHECK (review_status IN ('draft', 'reviewed', 'verified')),
      defensive_front TEXT,
      box_count INTEGER CHECK (box_count BETWEEN 0 AND 11),
      coverage_shell TEXT,
      blitz_look TEXT,
      corner_leverage TEXT,
      safety_rotation TEXT,
      notes TEXT,
      version INTEGER NOT NULL DEFAULT 1 CHECK (version > 0),
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      UNIQUE (dataset_id, clip_id, frame_time_ms)
    );

    CREATE TABLE IF NOT EXISTS ai_annotations (
      id BIGSERIAL PRIMARY KEY,
      frame_id BIGINT NOT NULL REFERENCES ai_annotation_frames(id) ON DELETE CASCADE,
      class_index INTEGER NOT NULL CHECK (class_index BETWEEN 0 AND 8),
      class_name TEXT NOT NULL,
      x NUMERIC NOT NULL CHECK (x BETWEEN 0 AND 1),
      y NUMERIC NOT NULL CHECK (y BETWEEN 0 AND 1),
      width NUMERIC NOT NULL CHECK (width > 0 AND width <= 1),
      height NUMERIC NOT NULL CHECK (height > 0 AND height <= 1),
      attributes JSONB NOT NULL DEFAULT '{}'::jsonb,
      version INTEGER NOT NULL DEFAULT 1 CHECK (version > 0),
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS ai_annotation_versions (
      id BIGSERIAL PRIMARY KEY,
      frame_id BIGINT,
      dataset_id BIGINT NOT NULL,
      clip_id BIGINT NOT NULL,
      frame_time_ms INTEGER NOT NULL,
      version INTEGER NOT NULL,
      snapshot JSONB NOT NULL,
      action TEXT NOT NULL CHECK (action IN ('updated', 'deleted')),
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS ai_dataset_clips_clip_idx
      ON ai_dataset_clips (clip_id);
    CREATE INDEX IF NOT EXISTS ai_annotation_frames_dataset_idx
      ON ai_annotation_frames (dataset_id, clip_id, frame_time_ms);
    CREATE INDEX IF NOT EXISTS ai_annotations_frame_idx
      ON ai_annotations (frame_id);
    CREATE INDEX IF NOT EXISTS ai_annotation_versions_frame_idx
      ON ai_annotation_versions (frame_id, created_at DESC);
  `);

  await db.query(`
    CREATE INDEX IF NOT EXISTS clips_team_game_idx
    ON clips (team, game_name, created_at);
  `);
  await db.query(`
    CREATE INDEX IF NOT EXISTS clips_queue_idx
    ON clips (queue_status, status, film_side);
  `);
  await db.query(`
    CREATE INDEX IF NOT EXISTS plays_team_game_idx
    ON plays (team, game_name, updated_at);
  `);

  console.log('Database ready');
}

function buildClipFilters(query, { requireLabeled = false } = {}) {
  const values = [];
  const where = [];
  const status = lower(query.status);
  const queueStatus = lower(query.queueStatus);
  const filmSide = lower(query.filmSide);
  const team = text(query.team);
  const gameName = text(query.gameName);
  const opponent = text(query.opponent);

  if (requireLabeled) where.push(`c.status = 'labeled'`);
  if (status) {
    if (!CLIP_STATUSES.has(status)) throw Object.assign(new Error('Invalid status filter'), { statusCode: 400 });
    values.push(status);
    where.push(`c.status = $${values.length}`);
  }
  if (queueStatus) {
    if (!QUEUE_STATUSES.has(queueStatus)) throw Object.assign(new Error('Invalid queueStatus filter'), { statusCode: 400 });
    values.push(queueStatus);
    where.push(`c.queue_status = $${values.length}`);
  }
  if (team) {
    values.push(team);
    where.push(`c.team = $${values.length}`);
  }
  if (gameName) {
    values.push(gameName);
    where.push(`c.game_name = $${values.length}`);
  }
  if (opponent) {
    values.push(opponent);
    where.push(`c.opponent = $${values.length}`);
  }
  if (filmSide) {
    if (!FILM_SIDES.has(filmSide)) throw Object.assign(new Error('Invalid filmSide filter'), { statusCode: 400 });
    values.push(filmSide);
    where.push(`c.film_side = $${values.length}`);
  }
  return { values, where };
}

async function getBreakdown(team, gameName = '', opponent = '') {
  const values = [team];
  const where = ['p.team = $1'];
  if (gameName) {
    values.push(gameName);
    where.push(`p.game_name = $${values.length}`);
  }
  if (opponent) {
    values.push(opponent);
    where.push(`c.opponent = $${values.length}`);
  }

  const result = await db.query(
    `SELECT p.*, c.film_side, c.labeled_at
     FROM plays p
     JOIN clips c ON c.id = p.clip_id
     WHERE c.status='labeled' AND ${where.join(' AND ')}
     ORDER BY p.created_at ASC`,
    values
  );
  const plays = result.rows;
  const runs = plays.filter((row) => lower(row.play_type) === 'run');
  const passes = plays.filter((row) => ['pass', 'screen'].includes(lower(row.play_type)));
  const rpos = plays.filter((row) => lower(row.play_type) === 'rpo');
  const blitzes = plays.filter((row) => row.blitz === true);
  const completedPasses = passes.filter((row) => row.completed === true);
  const deepPasses = passes.filter((row) => lower(row.pass_depth) === 'deep');
  const shortPasses = passes.filter((row) => lower(row.pass_depth) === 'short');
  const intermediatePasses = passes.filter((row) => lower(row.pass_depth) === 'intermediate');
  const withYards = plays.filter((row) => Number.isFinite(Number(row.yards_gained)));
  const explosive = withYards.filter((row) => {
    const yards = Number(row.yards_gained);
    return lower(row.play_type) === 'run' ? yards >= 10 : yards >= 20;
  });

  const downBreakdown = [1, 2, 3, 4].map((down) => {
    const onDown = plays.filter((row) => Number(row.down) === down);
    const downRuns = onDown.filter((row) => lower(row.play_type) === 'run');
    const downPasses = onDown.filter((row) => ['pass', 'screen'].includes(lower(row.play_type)));
    const downBlitzes = onDown.filter((row) => row.blitz === true);
    return {
      down,
      plays: onDown.length,
      runs: downRuns.length,
      passes: downPasses.length,
      runPercentage: pct(downRuns.length, onDown.length),
      passPercentage: pct(downPasses.length, onDown.length),
      blitzes: downBlitzes.length,
      blitzPercentage: pct(downBlitzes.length, onDown.length)
    };
  });

  const averageYards = (subset) => {
    const valuesWithYards = subset
      .map((row) => Number(row.yards_gained))
      .filter((value) => Number.isFinite(value));
    if (!valuesWithYards.length) return null;
    return Math.round((valuesWithYards.reduce((sum, value) => sum + value, 0) / valuesWithYards.length) * 10) / 10;
  };

  return {
    team,
    gameName: gameName || null,
    opponent: opponent || null,
    games: [...new Set(plays.map((row) => row.game_name).filter(Boolean))].sort(),
    totalPlays: plays.length,
    run: { count: runs.length, percentage: pct(runs.length, plays.length), averageYards: averageYards(runs) },
    pass: {
      count: passes.length,
      percentage: pct(passes.length, plays.length),
      completed: completedPasses.length,
      completionPercentage: pct(completedPasses.length, passes.length),
      averageYards: averageYards(passes)
    },
    rpo: { count: rpos.length, percentage: pct(rpos.length, plays.length), averageYards: averageYards(rpos) },
    explosive: { count: explosive.length, percentage: pct(explosive.length, withYards.length) },
    blitz: { count: blitzes.length, percentage: pct(blitzes.length, plays.length) },
    blitzVsRun: {
      count: runs.filter((row) => row.blitz === true).length,
      opportunities: runs.length,
      percentage: pct(runs.filter((row) => row.blitz === true).length, runs.length)
    },
    blitzVsPass: {
      count: passes.filter((row) => row.blitz === true).length,
      opportunities: passes.length,
      percentage: pct(passes.filter((row) => row.blitz === true).length, passes.length)
    },
    passDepth: {
      short: {
        attempts: shortPasses.length,
        percentage: pct(shortPasses.length, passes.length),
        completed: shortPasses.filter((row) => row.completed === true).length,
        completionPercentage: pct(shortPasses.filter((row) => row.completed === true).length, shortPasses.length)
      },
      intermediate: {
        attempts: intermediatePasses.length,
        percentage: pct(intermediatePasses.length, passes.length),
        completed: intermediatePasses.filter((row) => row.completed === true).length,
        completionPercentage: pct(intermediatePasses.filter((row) => row.completed === true).length, intermediatePasses.length)
      },
      deep: {
        attempts: deepPasses.length,
        percentage: pct(deepPasses.length, passes.length),
        completed: deepPasses.filter((row) => row.completed === true).length,
        completionPercentage: pct(deepPasses.filter((row) => row.completed === true).length, deepPasses.length)
      }
    },
    offenseFormations: countBy(plays, 'offense_formation', (row) => Boolean(row.offense_formation)),
    defenses: countBy(plays, 'defense_formation', (row) => Boolean(row.defense_formation)),
    playCalls: countBy(plays, 'play_call', (row) => Boolean(row.play_call)),
    playTypes: countBy(plays, 'play_type', (row) => Boolean(row.play_type)),
    coverages: countBy(plays, 'coverage', (row) => Boolean(row.coverage)),
    directions: countBy(plays, 'run_direction', (row) => Boolean(row.run_direction)),
    hashes: countBy(plays, 'hash', (row) => Boolean(row.hash)),
    passDepths: countBy(
      plays,
      'pass_depth',
      (row) => Boolean(row.pass_depth) && ['pass', 'screen'].includes(lower(row.play_type))
    ),
    downBreakdown
  };
}

function confidenceLabel(sampleCount) {
  if (sampleCount >= 25) return 'high';
  if (sampleCount >= 10) return 'medium';
  return 'developing';
}

function buildCoachNotes(breakdown) {
  const notes = [];
  const total = breakdown.totalPlays;
  const push = (category, title, detail, evidence, priority = 'medium') => {
    if (!evidence) return;
    notes.push({
      category,
      title,
      detail,
      evidence,
      priority,
      confidence: confidenceLabel(evidence)
    });
  };

  if (!total) {
    return [{
      category: 'readiness',
      title: 'More labeled film is needed',
      detail: 'Coach Notes will appear after clips are labeled for the selected team or game.',
      evidence: 0,
      priority: 'low',
      confidence: 'developing'
    }];
  }

  const topPlayType = breakdown.playTypes[0];
  if (topPlayType) push(
    'offense',
    `${topPlayType.name} is the most common play type`,
    `${topPlayType.count} of ${total} labeled plays (${topPlayType.percentage}%).`,
    topPlayType.count,
    topPlayType.percentage >= 60 ? 'high' : 'medium'
  );

  const topFormation = breakdown.offenseFormations[0];
  if (topFormation) push(
    'offense',
    `${topFormation.name} leads offensive formation usage`,
    `${topFormation.count} labeled snaps came from this formation (${topFormation.percentage}%).`,
    topFormation.count
  );

  const topPlayCall = breakdown.playCalls[0];
  if (topPlayCall) push(
    'offense',
    `${topPlayCall.name} is the most frequent play call`,
    `${topPlayCall.count} occurrences were recorded (${topPlayCall.percentage}%).`,
    topPlayCall.count
  );

  const topDirection = breakdown.directions[0];
  if (topDirection) push(
    'offense',
    `${topDirection.name} is the preferred direction`,
    `${topDirection.count} labeled directional plays went ${topDirection.name.toLowerCase()} (${topDirection.percentage}%).`,
    topDirection.count
  );

  const topDefense = breakdown.defenses[0];
  if (topDefense) push(
    'defense',
    `${topDefense.name} is the primary defensive front`,
    `${topDefense.count} snaps used this front (${topDefense.percentage}%).`,
    topDefense.count
  );

  const topCoverage = breakdown.coverages[0];
  if (topCoverage) push(
    'defense',
    `${topCoverage.name} is the primary coverage`,
    `${topCoverage.count} labeled snaps used this coverage (${topCoverage.percentage}%).`,
    topCoverage.count
  );

  if (breakdown.blitz.count) push(
    'defense',
    `Overall blitz rate is ${breakdown.blitz.percentage}%`,
    `${breakdown.blitz.count} of ${total} labeled plays were marked as blitzes.`,
    breakdown.blitz.count,
    breakdown.blitz.percentage >= 45 ? 'high' : 'medium'
  );

  const strongestBlitzDown = [...breakdown.downBreakdown]
    .filter((row) => row.plays >= 3)
    .sort((a, b) => b.blitzPercentage - a.blitzPercentage)[0];
  if (strongestBlitzDown && strongestBlitzDown.blitzes) push(
    'situation',
    `${strongestBlitzDown.down}${strongestBlitzDown.down === 1 ? 'st' : strongestBlitzDown.down === 2 ? 'nd' : strongestBlitzDown.down === 3 ? 'rd' : 'th'} down has the highest blitz rate`,
    `${strongestBlitzDown.blitzes} blitzes on ${strongestBlitzDown.plays} labeled plays (${strongestBlitzDown.blitzPercentage}%).`,
    strongestBlitzDown.plays,
    strongestBlitzDown.blitzPercentage >= 50 ? 'high' : 'medium'
  );

  if (breakdown.passDepth.deep.attempts) push(
    'offense',
    `Deep-pass completion rate is ${breakdown.passDepth.deep.completionPercentage}%`,
    `${breakdown.passDepth.deep.completed} completions on ${breakdown.passDepth.deep.attempts} labeled deep attempts.`,
    breakdown.passDepth.deep.attempts
  );

  if (breakdown.explosive.count) push(
    'offense',
    `${breakdown.explosive.count} explosive plays are labeled`,
    `Explosives represent ${breakdown.explosive.percentage}% of plays with yardage entered.`,
    breakdown.explosive.count
  );

  return notes.slice(0, 12);
}

function validateDatasetPayload(body, { partial = false } = {}) {
  const output = {};
  if (!partial || Object.hasOwn(body, 'name')) {
    output.name = text(body.name);
    if (!output.name) throw Object.assign(new Error('Dataset name is required'), { statusCode: 400 });
  }
  if (Object.hasOwn(body, 'description')) output.description = nullableText(body.description);
  if (Object.hasOwn(body, 'team')) output.team = nullableText(body.team);
  if (Object.hasOwn(body, 'status')) {
    output.status = lower(body.status);
    if (!DATASET_STATUSES.has(output.status)) {
      throw Object.assign(new Error('Dataset status must be draft, active, archived, or ready'), { statusCode: 400 });
    }
  }
  return output;
}

function validateFramePayload(body) {
  const reviewStatus = lower(body.reviewStatus || 'draft');
  if (!REVIEW_STATUSES.has(reviewStatus)) {
    throw Object.assign(new Error('reviewStatus must be draft, reviewed, or verified'), { statusCode: 400 });
  }
  const frameWidth = positiveId(body.frameWidth, 'frameWidth');
  const frameHeight = positiveId(body.frameHeight, 'frameHeight');
  const frameTimeMs = optionalInteger(body.frameTimeMs, 'frameTimeMs', 0, 2147483647);
  if (frameTimeMs === null) {
    throw Object.assign(new Error('frameTimeMs is required'), { statusCode: 400 });
  }
  const annotations = Array.isArray(body.annotations) ? body.annotations.map((annotation, index) => {
    const classIndex = optionalInteger(
      annotation.classIndex ?? annotation.class_index,
      `annotations[${index}].classIndex`,
      0,
      8
    );
    if (classIndex === null) {
      throw Object.assign(new Error(`annotations[${index}].classIndex is required`), { statusCode: 400 });
    }
    const suppliedClassName = annotation.className ?? annotation.class_name;
    if (suppliedClassName !== undefined && text(suppliedClassName) !== DEFENSIVE_OBJECT_CLASSES[classIndex]) {
      throw Object.assign(
        new Error(`annotations[${index}].className does not match classIndex`),
        { statusCode: 400 }
      );
    }
    const attributes = annotation.attributes ?? {};
    if (!attributes || Array.isArray(attributes) || typeof attributes !== 'object') {
      throw Object.assign(new Error(`annotations[${index}].attributes must be an object`), { statusCode: 400 });
    }
    return {
      classIndex,
      className: DEFENSIVE_OBJECT_CLASSES[classIndex],
      x: boundedNumber(annotation.x, `annotations[${index}].x`),
      y: boundedNumber(annotation.y, `annotations[${index}].y`),
      width: boundedNumber(annotation.width, `annotations[${index}].width`, { positive: true }),
      height: boundedNumber(annotation.height, `annotations[${index}].height`, { positive: true }),
      attributes
    };
  }) : [];
  return {
    datasetId: positiveId(body.datasetId, 'datasetId'),
    clipId: positiveId(body.clipId, 'clipId'),
    frameTimeMs,
    frameWidth,
    frameHeight,
    reviewStatus,
    defensiveFront: nullableText(body.defensiveFront),
    boxCount: optionalInteger(body.boxCount, 'boxCount', 0, 11),
    coverageShell: nullableText(body.coverageShell),
    blitzLook: nullableText(body.blitzLook),
    cornerLeverage: nullableText(body.cornerLeverage),
    safetyRotation: nullableText(body.safetyRotation),
    notes: nullableText(body.notes),
    annotations
  };
}

async function frameWithAnnotations(client, frameId, { lock = false } = {}) {
  const frameResult = await client.query(
    `SELECT * FROM ai_annotation_frames WHERE id=$1${lock ? ' FOR UPDATE' : ''}`,
    [frameId]
  );
  if (!frameResult.rowCount) return null;
  const annotations = await client.query(
    `SELECT id, frame_id, class_index, class_name,
            x::float8, y::float8, width::float8, height::float8,
            attributes, version, created_at, updated_at
     FROM ai_annotations WHERE frame_id=$1 ORDER BY id`,
    [frameId]
  );
  return { ...frameResult.rows[0], annotations: annotations.rows };
}

async function snapshotFrame(client, frame, action) {
  await client.query(
    `INSERT INTO ai_annotation_versions
       (frame_id, dataset_id, clip_id, frame_time_ms, version, snapshot, action)
     VALUES ($1,$2,$3,$4,$5,$6::jsonb,$7)`,
    [
      frame.id,
      frame.dataset_id,
      frame.clip_id,
      frame.frame_time_ms,
      frame.version,
      JSON.stringify(frame),
      action
    ]
  );
}

async function snapshotFramesForClipIds(client, clipIds) {
  if (!clipIds.length) return;
  const frames = await client.query(
    `SELECT id FROM ai_annotation_frames
     WHERE clip_id=ANY($1::bigint[]) FOR UPDATE`,
    [clipIds]
  );
  for (const row of frames.rows) {
    const frame = await frameWithAnnotations(client, row.id);
    await snapshotFrame(client, frame, 'deleted');
  }
}

function readinessFromCounts({ annotatedFrames, verifiedFrames, classCounts }) {
  const annotatedScore = Math.min(1, annotatedFrames / ANNOTATION_READINESS_TARGETS.annotatedFrames);
  const verifiedScore = Math.min(1, verifiedFrames / ANNOTATION_READINESS_TARGETS.verifiedFrames);
  const classScore = DEFENSIVE_OBJECT_CLASSES.reduce(
    (sum, name) => sum + Math.min(1, (classCounts[name] || 0) / ANNOTATION_READINESS_TARGETS.boxesPerClass),
    0
  ) / DEFENSIVE_OBJECT_CLASSES.length;
  return Math.round((annotatedScore * 0.4 + verifiedScore * 0.3 + classScore * 0.3) * 100);
}

app.get('/api/health', async (_req, res) => {
  try {
    await db.query('SELECT 1');
    res.json({ status: 'ok', database: 'connected', version: '1.0.0' });
  } catch (_error) {
    res.status(503).json({ status: 'error', database: 'disconnected', version: '1.0.0' });
  }
});

app.post('/api/upload', upload.array('files', 100), async (req, res, next) => {
  const files = req.files || [];
  const client = await db.connect();
  try {
    const team = text(req.body.team);
    const gameName = text(req.body.gameName);
    const opponent = nullableText(req.body.opponent);
    const gameDate = normalizeDate(req.body.gameDate);
    const season = nullableText(req.body.season);
    const autoLabel = lower(req.body.autoLabel) === 'true';
    const autoSortPossession = lower(req.body.autoSortPossession) === 'true';
    const jerseyColor = lower(req.body.jerseyColor);
    const helmetColor = lower(req.body.helmetColor);
    const homeAway = ['home', 'away'].includes(lower(req.body.homeAway)) ? lower(req.body.homeAway) : '';
    const offenseOnlyAi = lower(req.body.offenseOnlyAi) === 'true';

    if (!team || !gameName) {
      for (const file of files) fs.rmSync(file.path, { force: true });
      return res.status(400).json({ error: 'Team and game name are required' });
    }
    if (!files.length) return res.status(400).json({ error: 'Select at least one MP4 clip' });

    await client.query('BEGIN');
    const inserted = [];
    for (const file of files) {
      const possession = autoSortPossession
        ? await classifyPossession(file.path, { jerseyColor, helmetColor, homeAway })
        : {
            filmSide: 'needs_review',
            confidence: 0,
            reason: 'Automatic Team Identity sorting was not selected'
          };
      const filmSide = FILM_SIDES.has(lower(possession.filmSide))
        ? lower(possession.filmSide)
        : 'needs_review';
      const useForAi = !offenseOnlyAi || filmSide === 'offense';

      const result = await client.query(
        `INSERT INTO clips (
           team, game_name, opponent, game_date, season,
           original_name, stored_name, file_size,
           status, queue_status, film_side,
           possession_confidence, possession_reason,
           jersey_color, helmet_color, home_away, use_for_ai
         ) VALUES (
           $1,$2,$3,$4,$5,$6,$7,$8,
           'needs_labeling','not_queued',$9,$10,$11,$12,$13,$14,$15
         ) RETURNING *`,
        [
          team, gameName, opponent, gameDate, season,
          file.originalname, file.filename, file.size,
          filmSide, normalizeInteger(possession.confidence) ?? 0, nullableText(possession.reason),
          nullableText(jerseyColor), nullableText(helmetColor), nullableText(homeAway), useForAi
        ]
      );

      const clip = result.rows[0];
      const prediction = useForAi ? await predictClip(client, clip) : null;
      if (prediction) await savePrediction(client, clip, prediction);

      if (autoLabel && prediction) {
        await client.query(
          `UPDATE plays SET
             hash = COALESCE($2, hash),
             defense_formation = COALESCE($3, defense_formation),
             blitz = COALESCE($4, blitz),
             coverage = COALESCE($5, coverage),
             run_direction = COALESCE($6, run_direction),
             label_source = 'ai',
             confidence = $7,
             notes = CASE
               WHEN notes IS NULL OR notes = '' THEN 'AI pre-label — coach review required'
               ELSE notes
             END,
             updated_at = NOW()
           WHERE clip_id = $1`,
          [
            clip.id,
            prediction.hash,
            prediction.defense_formation,
            prediction.blitz,
            prediction.coverage,
            prediction.run_direction,
            prediction.overall_confidence
          ]
        );
      }

      inserted.push({
        ...clip,
        autoLabel: Boolean(autoLabel && prediction),
        aiPrediction: prediction,
        possession
      });
    }

    await client.query('COMMIT');
    res.status(201).json({ uploaded: inserted });
  } catch (error) {
    await client.query('ROLLBACK').catch(() => {});
    for (const file of files) {
      if (file?.path) fs.rmSync(file.path, { force: true });
    }
    next(error);
  } finally {
    client.release();
  }
});

app.get('/api/clips', async (req, res, next) => {
  try {
    const { values, where } = buildClipFilters(req.query);
    const result = await db.query(
      `SELECT c.*, p.id AS play_id, p.down, p.distance, p.hash, p.play_type,
              p.play_call, p.offense_formation, p.defense_formation,
              p.blitz, p.coverage, p.run_direction, p.pass_depth,
              p.completed, p.yards_gained, p.notes, p.ai_confidence
       FROM clips c
       LEFT JOIN plays p ON p.clip_id = c.id
       ${where.length ? `WHERE ${where.join(' AND ')}` : ''}
       ORDER BY c.created_at ASC`,
      values
    );
    res.json(result.rows);
  } catch (error) {
    next(error);
  }
});

app.get('/api/teams', async (_req, res, next) => {
  try {
    const result = await db.query(`
      SELECT team,
             COUNT(*)::int AS clip_count,
             COUNT(DISTINCT NULLIF(game_name,''))::int AS game_count,
             MAX(created_at) AS last_activity
      FROM clips
      GROUP BY team
      ORDER BY team;
    `);
    res.json(result.rows);
  } catch (error) {
    next(error);
  }
});

app.get('/api/games', async (req, res, next) => {
  try {
    const team = text(req.query.team);
    const values = [];
    const where = [`game_name <> ''`];
    if (team) {
      values.push(team);
      where.push(`team = $${values.length}`);
    }
    const result = await db.query(
      `SELECT team, game_name,
              MAX(opponent) AS opponent,
              MAX(game_date) AS game_date,
              MAX(season) AS season,
              COUNT(*)::int AS clip_count,
              COUNT(*) FILTER (WHERE film_side='offense')::int AS offense,
              COUNT(*) FILTER (WHERE film_side='defense')::int AS defense,
              COUNT(*) FILTER (WHERE film_side='needs_review')::int AS needs_review,
              COUNT(*) FILTER (WHERE queue_status='queued' AND status='needs_labeling')::int AS queued,
              COUNT(*) FILTER (WHERE status='labeled')::int AS labeled,
              MAX(created_at) AS updated_at
       FROM clips
       WHERE ${where.join(' AND ')}
       GROUP BY team, game_name
       ORDER BY MAX(COALESCE(game_date, created_at::date)) DESC, game_name`,
      values
    );
    res.json(result.rows);
  } catch (error) {
    next(error);
  }
});

app.get('/api/team-review-summary', async (req, res, next) => {
  try {
    const team = text(req.query.team);
    const gameName = text(req.query.gameName);
    if (!team || !gameName) return res.status(400).json({ error: 'Team and gameName are required' });

    const result = await db.query(
      `SELECT
         COUNT(*)::int AS total,
         COUNT(*) FILTER (WHERE film_side='offense')::int AS offense,
         COUNT(*) FILTER (WHERE film_side='defense')::int AS defense,
         COUNT(*) FILTER (WHERE film_side='offense' AND status='needs_labeling')::int AS eligible_offense,
         COUNT(*) FILTER (WHERE film_side='defense' AND status='needs_labeling')::int AS eligible_defense,
         COUNT(*) FILTER (WHERE film_side='needs_review')::int AS needs_review,
         COUNT(*) FILTER (WHERE queue_status='queued')::int AS queued,
         COUNT(*) FILTER (WHERE queue_status='queued' AND film_side='offense')::int AS queued_offense,
         COUNT(*) FILTER (WHERE queue_status='queued' AND film_side='defense')::int AS queued_defense
       FROM clips
       WHERE team=$1 AND game_name=$2`,
      [team, gameName]
    );
    res.json({ team, gameName, ...result.rows[0] });
  } catch (error) {
    next(error);
  }
});

app.post('/api/queue-clips', async (req, res, next) => {
  const team = text(req.body?.team);
  const gameName = text(req.body?.gameName);
  const includeOffense = req.body?.includeOffense === true;
  const includeDefense = req.body?.includeDefense === true;

  if (!team || !gameName) return res.status(400).json({ error: 'Team and gameName are required' });
  if (!includeOffense && !includeDefense) {
    return res.status(400).json({ error: 'Select Offense, Defense, or both' });
  }

  const selectedSides = [];
  if (includeOffense) selectedSides.push('offense');
  if (includeDefense) selectedSides.push('defense');

  const client = await db.connect();
  try {
    await client.query('BEGIN');
    const before = await client.query(
      `SELECT film_side, COUNT(*)::int AS count
       FROM clips
       WHERE team=$1 AND game_name=$2
         AND status='needs_labeling'
         AND queue_status='not_queued'
         AND film_side = ANY($3::text[])
       GROUP BY film_side`,
      [team, gameName, selectedSides]
    );

    const update = await client.query(
      `UPDATE clips
       SET queue_status='queued'
       WHERE team=$1 AND game_name=$2
         AND status='needs_labeling'
         AND film_side = ANY($3::text[])
       RETURNING id, film_side`,
      [team, gameName, selectedSides]
    );

    await client.query('COMMIT');
    const newlyQueued = Object.fromEntries(before.rows.map((row) => [row.film_side, row.count]));
    const totalBySide = update.rows.reduce((accumulator, row) => {
      accumulator[row.film_side] = (accumulator[row.film_side] || 0) + 1;
      return accumulator;
    }, {});

    res.json({
      queued: true,
      team,
      gameName,
      selectedSides,
      newlyQueued: {
        offense: newlyQueued.offense || 0,
        defense: newlyQueued.defense || 0,
        total: (newlyQueued.offense || 0) + (newlyQueued.defense || 0)
      },
      queuedTotals: {
        offense: totalBySide.offense || 0,
        defense: totalBySide.defense || 0,
        total: update.rowCount
      }
    });
  } catch (error) {
    await client.query('ROLLBACK');
    next(error);
  } finally {
    client.release();
  }
});

app.post('/api/games/reset', async (req, res, next) => {
  const team = text(req.body?.team);
  const gameName = text(req.body?.gameName);
  if (!team || !gameName) return res.status(400).json({ error: 'Team and gameName are required' });

  const client = await db.connect();
  try {
    await client.query('BEGIN');
    const clips = await client.query(
      `SELECT id FROM clips WHERE team=$1 AND game_name=$2 FOR UPDATE`,
      [team, gameName]
    );
    if (!clips.rowCount) {
      await client.query('ROLLBACK');
      return res.status(404).json({ error: 'Game not found' });
    }
    const ids = clips.rows.map((row) => row.id);
    await client.query(`DELETE FROM ai_training_events WHERE team=$1 AND game_name=$2`, [team, gameName]);
    await client.query(`DELETE FROM plays WHERE clip_id = ANY($1::int[])`, [ids]);
    await client.query(`DELETE FROM ai_predictions WHERE clip_id = ANY($1::int[])`, [ids]);
    await client.query(
      `UPDATE clips SET
         status='needs_labeling',
         queue_status='not_queued',
         labeled_at=NULL
       WHERE id = ANY($1::int[])`,
      [ids]
    );
    await rebuildModelCounts(client);
    await client.query('COMMIT');
    res.json({ reset: true, team, gameName, resetClips: clips.rowCount });
  } catch (error) {
    await client.query('ROLLBACK');
    next(error);
  } finally {
    client.release();
  }
});

app.delete('/api/games', async (req, res, next) => {
  const team = text(req.query.team || req.body?.team);
  const gameName = text(req.query.gameName || req.body?.gameName);
  if (!team || !gameName) return res.status(400).json({ error: 'Team and gameName are required' });

  const client = await db.connect();
  try {
    await client.query('BEGIN');
    const clips = await client.query(
      `SELECT id, stored_name FROM clips WHERE team=$1 AND game_name=$2 FOR UPDATE`,
      [team, gameName]
    );
    if (!clips.rowCount) {
      await client.query('ROLLBACK');
      return res.status(404).json({ error: 'Game not found' });
    }

    await client.query(`DELETE FROM ai_training_events WHERE team=$1 AND game_name=$2`, [team, gameName]);
    await client.query(`DELETE FROM ai_model_counts WHERE team=$1 AND game_name=$2`, [team, gameName]);
    await snapshotFramesForClipIds(client, clips.rows.map((clip) => clip.id));
    await client.query(`DELETE FROM clips WHERE team=$1 AND game_name=$2`, [team, gameName]);
    await rebuildModelCounts(client);
    await client.query('COMMIT');

    let deletedFiles = 0;
    for (const clip of clips.rows) {
      const filePath = path.join(CLIP_DIR, clip.stored_name);
      if (fs.existsSync(filePath)) deletedFiles += 1;
      fs.rmSync(filePath, { force: true });
    }
    res.json({ deleted: true, team, gameName, deletedClips: clips.rowCount, deletedFiles });
  } catch (error) {
    await client.query('ROLLBACK');
    next(error);
  } finally {
    client.release();
  }
});

app.delete('/api/clips/:id', async (req, res, next) => {
  const id = normalizeInteger(req.params.id);
  if (!id) return res.status(400).json({ error: 'Invalid clip id' });

  const client = await db.connect();
  try {
    await client.query('BEGIN');
    const result = await client.query(`SELECT stored_name FROM clips WHERE id=$1 FOR UPDATE`, [id]);
    if (!result.rowCount) {
      await client.query('ROLLBACK');
      return res.status(404).json({ error: 'Clip not found' });
    }
    await snapshotFramesForClipIds(client, [id]);
    await client.query(`DELETE FROM clips WHERE id=$1`, [id]);
    await client.query('COMMIT');
    fs.rmSync(path.join(CLIP_DIR, result.rows[0].stored_name), { force: true });
    res.json({ deleted: true });
  } catch (error) {
    await client.query('ROLLBACK');
    next(error);
  } finally {
    client.release();
  }
});

app.get('/api/clips/:id/video', async (req, res, next) => {
  try {
    const id = normalizeInteger(req.params.id);
    if (!id) return res.status(400).json({ error: 'Invalid clip id' });
    const result = await db.query(`SELECT stored_name FROM clips WHERE id=$1`, [id]);
    if (!result.rowCount) return res.status(404).json({ error: 'Clip not found' });

    const filePath = path.join(CLIP_DIR, result.rows[0].stored_name);
    if (!fs.existsSync(filePath)) return res.status(404).json({ error: 'Video file not found' });

    const stat = fs.statSync(filePath);
    const range = req.headers.range;
    if (!range) {
      res.writeHead(200, {
        'Content-Length': stat.size,
        'Content-Type': 'video/mp4',
        'Accept-Ranges': 'bytes'
      });
      return fs.createReadStream(filePath).pipe(res);
    }

    const [startText, endText] = range.replace(/bytes=/, '').split('-');
    const start = Number(startText);
    const end = endText ? Number(endText) : stat.size - 1;
    if (!Number.isFinite(start) || start > end || end >= stat.size) return res.status(416).end();

    res.writeHead(206, {
      'Content-Range': `bytes ${start}-${end}/${stat.size}`,
      'Accept-Ranges': 'bytes',
      'Content-Length': end - start + 1,
      'Content-Type': 'video/mp4'
    });
    return fs.createReadStream(filePath, { start, end }).pipe(res);
  } catch (error) {
    next(error);
  }
});

app.get('/api/clips/:id/label', async (req, res, next) => {
  try {
    const result = await db.query(`SELECT * FROM plays WHERE clip_id=$1`, [req.params.id]);
    res.json(result.rows[0] || null);
  } catch (error) {
    next(error);
  }
});

app.get('/api/clips/:id/prediction', async (req, res, next) => {
  try {
    const result = await db.query(`SELECT * FROM ai_predictions WHERE clip_id=$1`, [req.params.id]);
    res.json(result.rows[0] || null);
  } catch (error) {
    next(error);
  }
});

app.put('/api/clips/:id/label', async (req, res, next) => {
  const id = normalizeInteger(req.params.id);
  if (!id) return res.status(400).json({ error: 'Invalid clip id' });

  const client = await db.connect();
  try {
    await client.query('BEGIN');
    const clipResult = await client.query(`SELECT * FROM clips WHERE id=$1 FOR UPDATE`, [id]);
    if (!clipResult.rowCount) {
      await client.query('ROLLBACK');
      return res.status(404).json({ error: 'Clip not found' });
    }

    const clip = clipResult.rows[0];
    const body = req.body || {};
    const correctedSideCandidate = lower(body.filmSide);
    const correctedSide = FILM_SIDES.has(correctedSideCandidate)
      ? correctedSideCandidate
      : clip.film_side;

    const values = [
      clip.id,
      clip.team,
      clip.game_name,
      clip.original_name,
      normalizeInteger(body.down),
      normalizeInteger(body.distance),
      nullableText(body.hash),
      nullableText(body.playType),
      nullableText(body.playCall),
      nullableText(body.offenseFormation),
      nullableText(body.defenseFormation),
      normalizeBoolean(body.blitz) ?? false,
      nullableText(body.coverage),
      nullableText(body.runDirection),
      nullableText(body.passDepth),
      normalizeBoolean(body.completed),
      normalizeInteger(body.yardsGained),
      nullableText(body.notes)
    ];

    const result = await client.query(
      `INSERT INTO plays (
         clip_id, team, game_name, clip,
         down, distance, hash, play_type, play_call,
         offense_formation, defense_formation, blitz, coverage,
         run_direction, pass_depth, completed, yards_gained, notes,
         label_source, updated_at
       ) VALUES (
         $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,
         'coach',NOW()
       )
       ON CONFLICT (clip_id) DO UPDATE SET
         down=EXCLUDED.down,
         distance=EXCLUDED.distance,
         hash=EXCLUDED.hash,
         play_type=EXCLUDED.play_type,
         play_call=EXCLUDED.play_call,
         offense_formation=EXCLUDED.offense_formation,
         defense_formation=EXCLUDED.defense_formation,
         blitz=EXCLUDED.blitz,
         coverage=EXCLUDED.coverage,
         run_direction=EXCLUDED.run_direction,
         pass_depth=EXCLUDED.pass_depth,
         completed=EXCLUDED.completed,
         yards_gained=EXCLUDED.yards_gained,
         notes=EXCLUDED.notes,
         label_source='coach',
         updated_at=NOW()
       RETURNING *`,
      values
    );

    const correctedUseForAi = correctedSide !== 'needs_review';
    if (correctedUseForAi) {
      await learnFromCorrection(client, { ...clip, use_for_ai: true }, result.rows[0]);
    }

    await client.query(
      `UPDATE clips SET
         status='labeled',
         labeled_at=NOW(),
         film_side=$2,
         use_for_ai=$3,
         possession_confidence=100,
         possession_reason='Coach verified'
       WHERE id=$1`,
      [clip.id, correctedSide, correctedUseForAi]
    );

    await client.query('COMMIT');
    res.json(result.rows[0]);
  } catch (error) {
    await client.query('ROLLBACK');
    next(error);
  } finally {
    client.release();
  }
});

app.patch('/api/clips/:id/film-side', async (req, res, next) => {
  try {
    const id = normalizeInteger(req.params.id);
    const filmSide = lower(req.body?.filmSide);
    const reviewNotes = nullableText(req.body?.reviewNotes);
    if (!id) return res.status(400).json({ error: 'Invalid clip id' });
    if (!FILM_SIDES.has(filmSide)) return res.status(400).json({ error: 'Invalid filmSide' });

    const useForAi = filmSide !== 'needs_review';
    const result = await db.query(
      `UPDATE clips SET
         film_side=$2,
         use_for_ai=$3,
         possession_confidence=CASE WHEN $2='needs_review' THEN possession_confidence ELSE 100 END,
         possession_reason=CASE
           WHEN $2='needs_review' THEN 'Coach left clip in Needs Review'
           ELSE 'Coach corrected in Team Identity Review'
         END,
         review_notes=COALESCE($4, review_notes)
       WHERE id=$1
       RETURNING *`,
      [id, filmSide, useForAi, reviewNotes]
    );
    if (!result.rowCount) return res.status(404).json({ error: 'Clip not found' });
    res.json(result.rows[0]);
  } catch (error) {
    next(error);
  }
});

app.post('/api/clips/:id/retry-processing', async (req, res, next) => {
  try {
    const id = normalizeInteger(req.params.id);
    if (!id) return res.status(400).json({ error: 'Invalid clip id' });
    const result = await db.query(`SELECT * FROM clips WHERE id=$1`, [id]);
    if (!result.rowCount) return res.status(404).json({ error: 'Clip not found' });

    const clip = result.rows[0];
    const filePath = path.join(CLIP_DIR, clip.stored_name);
    if (!fs.existsSync(filePath)) return res.status(404).json({ error: 'Video file not found' });

    const possession = await classifyPossession(filePath, {
      jerseyColor: clip.jersey_color,
      helmetColor: clip.helmet_color,
      homeAway: clip.home_away
    });
    const filmSide = FILM_SIDES.has(lower(possession.filmSide))
      ? lower(possession.filmSide)
      : 'needs_review';
    const useForAi = filmSide !== 'needs_review';
    const updated = await db.query(
      `UPDATE clips SET
         film_side=$2,
         possession_confidence=$3,
         possession_reason=$4,
         use_for_ai=$5
       WHERE id=$1
       RETURNING *`,
      [id, filmSide, normalizeInteger(possession.confidence) ?? 0, nullableText(possession.reason), useForAi]
    );

    if (useForAi) {
      const prediction = await predictClip(db, updated.rows[0]);
      if (prediction) await savePrediction(db, updated.rows[0], prediction);
    }
    res.json({ clip: updated.rows[0], retried: true });
  } catch (error) {
    next(error);
  }
});

app.patch('/api/clips/:id/status', async (req, res, next) => {
  try {
    const status = lower(req.body?.status);
    if (!CLIP_STATUSES.has(status)) return res.status(400).json({ error: 'Invalid status' });
    const result = await db.query(
      `UPDATE clips SET status=$1 WHERE id=$2 RETURNING *`,
      [status, req.params.id]
    );
    if (!result.rowCount) return res.status(404).json({ error: 'Clip not found' });
    res.json(result.rows[0]);
  } catch (error) {
    next(error);
  }
});

app.get('/api/accuracy', async (req, res, next) => {
  try {
    const team = text(req.query.team);
    const gameName = text(req.query.gameName);
    const values = [];
    const where = [`c.status='labeled'`];
    if (team) {
      values.push(team);
      where.push(`c.team=$${values.length}`);
    }
    if (gameName) {
      values.push(gameName);
      where.push(`c.game_name=$${values.length}`);
    }

    const result = await db.query(
      `SELECT p.*, c.labeled_at, c.team, c.game_name
       FROM plays p
       JOIN clips c ON c.id=p.clip_id
       WHERE ${where.join(' AND ')}
       ORDER BY c.labeled_at DESC`,
      values
    );
    const rows = result.rows;
    const definitions = [
      ['hash', 'ai_hash'],
      ['defenseFormation', 'ai_defense_formation'],
      ['blitz', 'ai_blitz'],
      ['coverage', 'ai_coverage'],
      ['playDirection', 'ai_run_direction']
    ];
    const coachColumns = {
      hash: 'hash',
      defenseFormation: 'defense_formation',
      blitz: 'blitz',
      coverage: 'coverage',
      playDirection: 'run_direction'
    };

    const categories = {};
    let totalCorrect = 0;
    let totalCompared = 0;
    for (const [name, aiColumn] of definitions) {
      let correct = 0;
      let compared = 0;
      for (const row of rows) {
        const aiValue = row[aiColumn];
        const coachValue = row[coachColumns[name]];
        if (aiValue === null || aiValue === undefined || aiValue === '' ||
            coachValue === null || coachValue === undefined || coachValue === '') continue;
        compared += 1;
        if (lower(aiValue) === lower(coachValue)) correct += 1;
      }
      categories[name] = {
        correct,
        compared,
        accuracy: compared ? pct(correct, compared) : null
      };
      totalCorrect += correct;
      totalCompared += compared;
    }

    const recentRows = rows.slice(0, 25);
    let recentCorrect = 0;
    let recentCompared = 0;
    for (const row of recentRows) {
      for (const [name, aiColumn] of definitions) {
        const aiValue = row[aiColumn];
        const coachValue = row[coachColumns[name]];
        if (aiValue === null || aiValue === undefined || aiValue === '' ||
            coachValue === null || coachValue === undefined || coachValue === '') continue;
        recentCompared += 1;
        if (lower(aiValue) === lower(coachValue)) recentCorrect += 1;
      }
    }

    const overallAccuracy = totalCompared ? pct(totalCorrect, totalCompared) : null;
    const readiness = totalCompared < 100
      ? 'Not enough data'
      : overallAccuracy >= 90
        ? 'Nearly ready'
        : overallAccuracy >= 80
          ? 'Needs supervision'
          : 'Training needed';

    const byGameMap = new Map();
    for (const row of rows) {
      const key = `${row.team}||${row.game_name}`;
      if (!byGameMap.has(key)) byGameMap.set(key, { team: row.team, gameName: row.game_name, correct: 0, compared: 0 });
      const bucket = byGameMap.get(key);
      for (const [name, aiColumn] of definitions) {
        const aiValue = row[aiColumn];
        const coachValue = row[coachColumns[name]];
        if (aiValue === null || aiValue === undefined || aiValue === '' ||
            coachValue === null || coachValue === undefined || coachValue === '') continue;
        bucket.compared += 1;
        if (lower(aiValue) === lower(coachValue)) bucket.correct += 1;
      }
    }
    const byGame = [...byGameMap.values()].map((row) => ({
      ...row,
      accuracy: row.compared ? pct(row.correct, row.compared) : null
    }));

    res.json({
      overallAccuracy,
      totalComparisons: totalCompared,
      labeledClips: rows.length,
      recentAccuracy: recentCompared ? pct(recentCorrect, recentCompared) : null,
      readiness,
      categories,
      byGame
    });
  } catch (error) {
    next(error);
  }
});

app.get('/api/breakdown', async (req, res, next) => {
  try {
    const team = text(req.query.team);
    const gameName = text(req.query.gameName);
    const opponent = text(req.query.opponent);
    if (!team) return res.status(400).json({ error: 'Team is required' });
    res.json(await getBreakdown(team, gameName, opponent));
  } catch (error) {
    next(error);
  }
});

app.get('/api/coach-notes', async (req, res, next) => {
  try {
    const team = text(req.query.team);
    const gameName = text(req.query.gameName);
    const opponent = text(req.query.opponent);
    if (!team) return res.status(400).json({ error: 'Team is required' });
    const breakdown = await getBreakdown(team, gameName, opponent);
    res.json({
      team,
      gameName: gameName || null,
      opponent: opponent || null,
      generatedAt: new Date().toISOString(),
      sampleCount: breakdown.totalPlays,
      notes: buildCoachNotes(breakdown)
    });
  } catch (error) {
    next(error);
  }
});

app.get('/api/labeled-clips', async (req, res, next) => {
  try {
    const { values, where } = buildClipFilters(req.query, { requireLabeled: true });
    const result = await db.query(
      `SELECT c.id, c.team, c.game_name, c.opponent, c.game_date, c.season,
              c.original_name, c.created_at, c.labeled_at, c.film_side,
              c.use_for_ai, c.queue_status,
              p.down, p.distance, p.hash, p.play_type, p.play_call,
              p.offense_formation, p.defense_formation, p.blitz, p.coverage,
              p.run_direction, p.pass_depth, p.completed, p.yards_gained, p.notes
       FROM clips c
       JOIN plays p ON p.clip_id = c.id
       ${where.length ? `WHERE ${where.join(' AND ')}` : ''}
       ORDER BY c.team, c.game_name, c.created_at`,
      values
    );
    res.json(result.rows);
  } catch (error) {
    next(error);
  }
});

app.get('/api/plays', async (req, res, next) => {
  try {
    const team = text(req.query.team);
    const gameName = text(req.query.gameName);
    const values = [];
    const where = [];
    if (team) {
      values.push(team);
      where.push(`team=$${values.length}`);
    }
    if (gameName) {
      values.push(gameName);
      where.push(`game_name=$${values.length}`);
    }
    const result = await db.query(
      `SELECT * FROM plays
       ${where.length ? `WHERE ${where.join(' AND ')}` : ''}
       ORDER BY updated_at DESC`,
      values
    );
    res.json(result.rows);
  } catch (error) {
    next(error);
  }
});

app.get('/api/formations', async (req, res, next) => {
  try {
    const team = text(req.query.team);
    const side = lower(req.query.side);
    if (!team) return res.status(400).json({ error: 'Team is required' });
    if (side && !FORMATION_SIDES.has(side)) return res.status(400).json({ error: 'Invalid formation side' });

    const column = side === 'defense' ? 'defense_formation' : 'offense_formation';
    const sides = side ? [side] : ['offense', 'defense'];
    const output = {};
    for (const currentSide of sides) {
      const currentColumn = currentSide === 'defense' ? 'defense_formation' : 'offense_formation';
      const result = await db.query(
        `SELECT ${currentColumn} AS name,
                COUNT(*)::int AS snap_count,
                COUNT(DISTINCT NULLIF(game_name,''))::int AS game_count,
                MAX(updated_at) AS last_used
         FROM plays
         WHERE team=$1 AND NULLIF(TRIM(${currentColumn}), '') IS NOT NULL
         GROUP BY ${currentColumn}
         ORDER BY COUNT(*) DESC, ${currentColumn}`,
        [team]
      );
      const total = result.rows.reduce((sum, row) => sum + Number(row.snap_count), 0);
      output[currentSide] = result.rows.map((row) => ({
        ...row,
        percentage: pct(row.snap_count, total)
      }));
    }
    res.json(side ? output[side] : output);
  } catch (error) {
    next(error);
  }
});

app.post('/api/formations', async (req, res, next) => {
  try {
    const team = text(req.body?.team);
    const side = lower(req.body?.side);
    const name = text(req.body?.name);
    if (!team || !name) return res.status(400).json({ error: 'Team and formation name are required' });
    if (!FORMATION_SIDES.has(side)) return res.status(400).json({ error: 'Invalid formation side' });

    res.status(201).json({
      created: true,
      team,
      side,
      name,
      message: 'Formation names become active when used on a labeled clip.'
    });
  } catch (error) {
    next(error);
  }
});

app.patch('/api/formations/rename', async (req, res, next) => {
  try {
    const team = text(req.body?.team);
    const side = lower(req.body?.side);
    const oldName = text(req.body?.oldName);
    const newName = text(req.body?.newName);
    if (!team || !oldName || !newName) {
      return res.status(400).json({ error: 'Team, oldName, and newName are required' });
    }
    if (!FORMATION_SIDES.has(side)) return res.status(400).json({ error: 'Invalid formation side' });

    const column = side === 'defense' ? 'defense_formation' : 'offense_formation';
    const result = await db.query(
      `UPDATE plays SET ${column}=$3, updated_at=NOW()
       WHERE team=$1 AND LOWER(TRIM(${column}))=LOWER(TRIM($2))`,
      [team, oldName, newName]
    );
    res.json({ renamed: true, team, side, oldName, newName, updatedPlays: result.rowCount });
  } catch (error) {
    next(error);
  }
});

app.post('/api/formations/merge', async (req, res, next) => {
  try {
    const team = text(req.body?.team);
    const side = lower(req.body?.side);
    const sourceNames = Array.isArray(req.body?.sourceNames)
      ? req.body.sourceNames.map(text).filter(Boolean)
      : [];
    const targetName = text(req.body?.targetName);
    if (!team || !sourceNames.length || !targetName) {
      return res.status(400).json({ error: 'Team, sourceNames, and targetName are required' });
    }
    if (!FORMATION_SIDES.has(side)) return res.status(400).json({ error: 'Invalid formation side' });

    const column = side === 'defense' ? 'defense_formation' : 'offense_formation';
    const result = await db.query(
      `UPDATE plays SET ${column}=$3, updated_at=NOW()
       WHERE team=$1 AND LOWER(TRIM(${column})) = ANY($2::text[])`,
      [team, sourceNames.map((name) => lower(name)), targetName]
    );
    res.json({ merged: true, team, side, sourceNames, targetName, updatedPlays: result.rowCount });
  } catch (error) {
    next(error);
  }
});

app.delete('/api/formations', async (req, res, next) => {
  try {
    const team = text(req.query.team || req.body?.team);
    const side = lower(req.query.side || req.body?.side);
    const name = text(req.query.name || req.body?.name);
    if (!team || !name) return res.status(400).json({ error: 'Team and formation name are required' });
    if (!FORMATION_SIDES.has(side)) return res.status(400).json({ error: 'Invalid formation side' });

    const column = side === 'defense' ? 'defense_formation' : 'offense_formation';
    const result = await db.query(
      `UPDATE plays SET ${column}=NULL, updated_at=NOW()
       WHERE team=$1 AND LOWER(TRIM(${column}))=LOWER(TRIM($2))`,
      [team, name]
    );
    res.json({ deleted: true, team, side, name, updatedPlays: result.rowCount });
  } catch (error) {
    next(error);
  }
});

app.get('/api/formation-matchups', async (req, res, next) => {
  try {
    const team = text(req.query.team);
    const gameName = text(req.query.gameName);
    const opponent = text(req.query.opponent);
    if (!team) return res.status(400).json({ error: 'Team is required' });

    const values = [team];
    const where = [`p.team=$1`, `c.status='labeled'`];
    if (gameName) {
      values.push(gameName);
      where.push(`p.game_name=$${values.length}`);
    }
    if (opponent) {
      values.push(opponent);
      where.push(`c.opponent=$${values.length}`);
    }
    const result = await db.query(
      `SELECT p.offense_formation, p.defense_formation, p.coverage,
              p.play_type, p.down, p.blitz, p.yards_gained
       FROM plays p
       JOIN clips c ON c.id=p.clip_id
       WHERE ${where.join(' AND ')}
         AND NULLIF(TRIM(p.offense_formation),'') IS NOT NULL
       ORDER BY p.updated_at DESC`,
      values
    );

    const buckets = new Map();
    for (const row of result.rows) {
      const offenseFormation = text(row.offense_formation) || 'Unlabeled';
      const defenseFormation = text(row.defense_formation) || 'Unlabeled';
      const key = `${offenseFormation}||${defenseFormation}`;
      if (!buckets.has(key)) {
        buckets.set(key, {
          offenseFormation,
          defenseFormation,
          snaps: 0,
          blitzes: 0,
          yardsTotal: 0,
          yardsSamples: 0,
          coverages: new Map(),
          playTypes: new Map(),
          downs: new Map()
        });
      }
      const bucket = buckets.get(key);
      bucket.snaps += 1;
      if (row.blitz === true) bucket.blitzes += 1;
      if (Number.isFinite(Number(row.yards_gained))) {
        bucket.yardsTotal += Number(row.yards_gained);
        bucket.yardsSamples += 1;
      }
      const coverage = text(row.coverage) || 'Unlabeled';
      bucket.coverages.set(coverage, (bucket.coverages.get(coverage) || 0) + 1);
      const playType = text(row.play_type) || 'Unlabeled';
      bucket.playTypes.set(playType, (bucket.playTypes.get(playType) || 0) + 1);
      const down = Number(row.down) || 0;
      bucket.downs.set(down, (bucket.downs.get(down) || 0) + 1);
    }

    const matchups = [...buckets.values()].map((bucket) => ({
      offenseFormation: bucket.offenseFormation,
      defenseFormation: bucket.defenseFormation,
      snaps: bucket.snaps,
      blitzPercentage: pct(bucket.blitzes, bucket.snaps),
      averageYards: bucket.yardsSamples
        ? Math.round((bucket.yardsTotal / bucket.yardsSamples) * 10) / 10
        : null,
      topCoverage: [...bucket.coverages.entries()].sort((a, b) => b[1] - a[1])[0]?.[0] || null,
      topPlayType: [...bucket.playTypes.entries()].sort((a, b) => b[1] - a[1])[0]?.[0] || null,
      downBreakdown: [...bucket.downs.entries()]
        .filter(([down]) => down)
        .map(([down, count]) => ({ down, count }))
        .sort((a, b) => a.down - b.down)
    })).sort((a, b) => b.snaps - a.snaps);

    res.json({ team, gameName: gameName || null, opponent: opponent || null, totalSnaps: result.rowCount, matchups });
  } catch (error) {
    next(error);
  }
});

app.get('/api/training/datasets', async (_req, res, next) => {
  try {
    const result = await db.query(`
      SELECT d.*,
             (SELECT COUNT(*)::int FROM ai_dataset_clips dc WHERE dc.dataset_id=d.id) AS clip_count,
             (SELECT COUNT(*)::int FROM ai_annotation_frames f WHERE f.dataset_id=d.id) AS annotated_frames,
             (SELECT COUNT(*)::int FROM ai_annotation_frames f WHERE f.dataset_id=d.id AND f.review_status='verified') AS verified_frames,
             (SELECT COUNT(*)::int FROM ai_annotations a JOIN ai_annotation_frames f ON f.id=a.frame_id WHERE f.dataset_id=d.id) AS box_count
      FROM ai_datasets d
      ORDER BY d.updated_at DESC, d.id DESC
    `);
    res.json(result.rows);
  } catch (error) {
    next(error);
  }
});

app.post('/api/training/datasets', async (req, res, next) => {
  try {
    const input = validateDatasetPayload(req.body || {});
    const result = await db.query(
      `INSERT INTO ai_datasets (name, description, team, status)
       VALUES ($1,$2,$3,$4) RETURNING *`,
      [input.name, input.description ?? null, input.team ?? null, input.status || 'draft']
    );
    res.status(201).json(result.rows[0]);
  } catch (error) {
    next(error);
  }
});

app.get('/api/training/datasets/:id', async (req, res, next) => {
  try {
    const id = positiveId(req.params.id, 'Dataset ID');
    const dataset = await db.query(
      `SELECT d.*,
              (SELECT COUNT(*)::int FROM ai_dataset_clips dc WHERE dc.dataset_id=d.id) AS clip_count,
              (SELECT COUNT(*)::int FROM ai_annotation_frames f WHERE f.dataset_id=d.id) AS annotated_frames,
              (SELECT COUNT(*)::int FROM ai_annotation_frames f WHERE f.dataset_id=d.id AND f.review_status='verified') AS verified_frames,
              (SELECT COUNT(*)::int FROM ai_annotations a JOIN ai_annotation_frames f ON f.id=a.frame_id WHERE f.dataset_id=d.id) AS box_count
       FROM ai_datasets d
       WHERE d.id=$1`,
      [id]
    );
    if (!dataset.rowCount) return res.status(404).json({ error: 'Dataset not found' });
    const classes = await db.query(
      `SELECT class_index, class_name, COUNT(*)::int AS count
       FROM ai_annotations a
       JOIN ai_annotation_frames f ON f.id=a.frame_id
       WHERE f.dataset_id=$1
       GROUP BY class_index, class_name ORDER BY class_index`,
      [id]
    );
    const classCounts = Object.fromEntries(DEFENSIVE_OBJECT_CLASSES.map((name) => [name, 0]));
    for (const row of classes.rows) classCounts[row.class_name] = row.count;
    res.json({
      ...dataset.rows[0],
      class_counts: DEFENSIVE_OBJECT_CLASSES.map((name, index) => ({
        class_index: index,
        class_name: name,
        count: classCounts[name],
        warning: classCounts[name] < ANNOTATION_READINESS_TARGETS.boxesPerClass
      }))
    });
  } catch (error) {
    next(error);
  }
});

app.patch('/api/training/datasets/:id', async (req, res, next) => {
  try {
    const id = positiveId(req.params.id, 'Dataset ID');
    const input = validateDatasetPayload(req.body || {}, { partial: true });
    const entries = Object.entries(input);
    if (!entries.length) return res.status(400).json({ error: 'Provide at least one dataset field' });
    const columns = {
      name: 'name',
      description: 'description',
      team: 'team',
      status: 'status'
    };
    const values = entries.map(([, value]) => value);
    const sets = entries.map(([key], index) => `${columns[key]}=$${index + 1}`);
    values.push(id);
    const result = await db.query(
      `UPDATE ai_datasets SET ${sets.join(', ')}, updated_at=NOW()
       WHERE id=$${values.length} RETURNING *`,
      values
    );
    if (!result.rowCount) return res.status(404).json({ error: 'Dataset not found' });
    res.json(result.rows[0]);
  } catch (error) {
    next(error);
  }
});

app.delete('/api/training/datasets/:id', async (req, res, next) => {
  const client = await db.connect();
  try {
    const id = positiveId(req.params.id, 'Dataset ID');
    await client.query('BEGIN');
    const counts = await client.query(
      `SELECT d.name,
              (SELECT COUNT(*)::int FROM ai_dataset_clips dc WHERE dc.dataset_id=d.id) AS clips_removed,
              (SELECT COUNT(*)::int FROM ai_annotation_frames f WHERE f.dataset_id=d.id) AS frames_removed,
              (SELECT COUNT(*)::int FROM ai_annotations a JOIN ai_annotation_frames f ON f.id=a.frame_id WHERE f.dataset_id=d.id) AS annotations_removed
       FROM ai_datasets d
       WHERE d.id=$1`,
      [id]
    );
    if (!counts.rowCount) {
      await client.query('ROLLBACK');
      return res.status(404).json({ error: 'Dataset not found' });
    }
    const frames = await client.query('SELECT id FROM ai_annotation_frames WHERE dataset_id=$1 FOR UPDATE', [id]);
    for (const row of frames.rows) {
      const frame = await frameWithAnnotations(client, row.id);
      await snapshotFrame(client, frame, 'deleted');
    }
    await client.query('DELETE FROM ai_datasets WHERE id=$1', [id]);
    await client.query('COMMIT');
    res.json({
      removed: counts.rows[0],
      videos_deleted: 0,
      message: 'Dataset assignments and annotation data removed; clip videos were preserved'
    });
  } catch (error) {
    await client.query('ROLLBACK').catch(() => {});
    next(error);
  } finally {
    client.release();
  }
});

app.get('/api/training/datasets/:id/clips', async (req, res, next) => {
  try {
    const id = positiveId(req.params.id, 'Dataset ID');
    const exists = await db.query('SELECT 1 FROM ai_datasets WHERE id=$1', [id]);
    if (!exists.rowCount) return res.status(404).json({ error: 'Dataset not found' });
    const result = await db.query(
      `SELECT c.id, c.team, c.game_name, c.opponent, c.original_name, c.film_side,
              c.created_at, dc.added_at, '/api/clips/' || c.id || '/video' AS media_url,
              COUNT(DISTINCT f.id)::int AS annotated_frames,
              COUNT(DISTINCT f.id) FILTER (WHERE f.review_status='verified')::int AS verified_frames,
              COUNT(a.id)::int AS box_count
       FROM ai_dataset_clips dc
       JOIN clips c ON c.id=dc.clip_id
       LEFT JOIN ai_annotation_frames f ON f.dataset_id=dc.dataset_id AND f.clip_id=c.id
       LEFT JOIN ai_annotations a ON a.frame_id=f.id
       WHERE dc.dataset_id=$1
       GROUP BY c.id, dc.added_at
       ORDER BY dc.added_at DESC`,
      [id]
    );
    res.json(result.rows);
  } catch (error) {
    next(error);
  }
});

app.post('/api/training/datasets/:id/clips', async (req, res, next) => {
  const client = await db.connect();
  try {
    const datasetId = positiveId(req.params.id, 'Dataset ID');
    if (!Array.isArray(req.body?.clipIds) || !req.body.clipIds.length) {
      return res.status(400).json({ error: 'clipIds must be a non-empty array' });
    }
    const clipIds = [...new Set(req.body.clipIds.map((id) => positiveId(id, 'Clip ID')))];
    await client.query('BEGIN');
    const dataset = await client.query('SELECT 1 FROM ai_datasets WHERE id=$1 FOR UPDATE', [datasetId]);
    if (!dataset.rowCount) {
      await client.query('ROLLBACK');
      return res.status(404).json({ error: 'Dataset not found' });
    }
    const clips = await client.query('SELECT id FROM clips WHERE id=ANY($1::bigint[])', [clipIds]);
    if (clips.rowCount !== clipIds.length) {
      await client.query('ROLLBACK');
      return res.status(404).json({ error: 'One or more clips were not found' });
    }
    const inserted = await client.query(
      `INSERT INTO ai_dataset_clips (dataset_id, clip_id)
       SELECT $1, UNNEST($2::bigint[])
       ON CONFLICT DO NOTHING RETURNING clip_id`,
      [datasetId, clipIds]
    );
    await client.query('UPDATE ai_datasets SET updated_at=NOW() WHERE id=$1', [datasetId]);
    await client.query('COMMIT');
    res.status(201).json({ added: inserted.rows.map((row) => row.clip_id), requested: clipIds.length });
  } catch (error) {
    await client.query('ROLLBACK').catch(() => {});
    next(error);
  } finally {
    client.release();
  }
});

app.delete('/api/training/datasets/:id/clips/:clipId', async (req, res, next) => {
  try {
    const datasetId = positiveId(req.params.id, 'Dataset ID');
    const clipId = positiveId(req.params.clipId, 'Clip ID');
    const result = await db.query(
      `DELETE FROM ai_dataset_clips WHERE dataset_id=$1 AND clip_id=$2 RETURNING clip_id`,
      [datasetId, clipId]
    );
    if (!result.rowCount) return res.status(404).json({ error: 'Clip assignment not found' });
    await db.query('UPDATE ai_datasets SET updated_at=NOW() WHERE id=$1', [datasetId]);
    res.json({ removed_clip_id: clipId, video_deleted: false });
  } catch (error) {
    next(error);
  }
});

app.get('/api/training/clips', async (req, res, next) => {
  try {
    const values = [];
    const where = [];
    const team = text(req.query.team);
    const gameName = text(req.query.gameName);
    const search = text(req.query.search);
    const datasetId = req.query.datasetId ? positiveId(req.query.datasetId, 'datasetId') : null;
    if (team) { values.push(team); where.push(`c.team=$${values.length}`); }
    if (gameName) { values.push(gameName); where.push(`c.game_name=$${values.length}`); }
    if (search) {
      values.push(`%${search}%`);
      where.push(`(c.original_name ILIKE $${values.length} OR c.team ILIKE $${values.length} OR c.game_name ILIKE $${values.length})`);
    }
    if (datasetId) {
      values.push(datasetId);
      where.push(`EXISTS (SELECT 1 FROM ai_dataset_clips dc WHERE dc.clip_id=c.id AND dc.dataset_id=$${values.length})`);
    }
    const result = await db.query(
      `SELECT c.id, c.team, c.game_name, c.opponent, c.game_date, c.season,
              c.original_name, c.film_side, c.created_at,
              '/api/clips/' || c.id || '/video' AS media_url
       FROM clips c ${where.length ? `WHERE ${where.join(' AND ')}` : ''}
       ORDER BY c.created_at DESC LIMIT 500`,
      values
    );
    res.json(result.rows);
  } catch (error) {
    next(error);
  }
});

app.get('/api/training/frames', async (req, res, next) => {
  try {
    const datasetId = positiveId(req.query.datasetId, 'datasetId');
    const clipId = positiveId(req.query.clipId, 'clipId');
    const values = [datasetId, clipId];
    let timeClause = '';
    if (req.query.timeMs !== undefined) {
      const timeMs = optionalInteger(req.query.timeMs, 'timeMs', 0, 2147483647);
      values.push(timeMs);
      timeClause = `ORDER BY ABS(frame_time_ms-$3), frame_time_ms LIMIT 1`;
    } else {
      timeClause = 'ORDER BY frame_time_ms';
    }
    const frames = await db.query(
      `SELECT * FROM ai_annotation_frames
       WHERE dataset_id=$1 AND clip_id=$2 ${timeClause}`,
      values
    );
    const output = [];
    for (const frame of frames.rows) output.push(await frameWithAnnotations(db, frame.id));
    res.json(output);
  } catch (error) {
    next(error);
  }
});

app.post('/api/training/frames', async (req, res, next) => {
  const client = await db.connect();
  try {
    const input = validateFramePayload(req.body || {});
    await client.query('BEGIN');
    const assignment = await client.query(
      `SELECT 1 FROM ai_dataset_clips WHERE dataset_id=$1 AND clip_id=$2`,
      [input.datasetId, input.clipId]
    );
    if (!assignment.rowCount) {
      await client.query('ROLLBACK');
      return res.status(409).json({ error: 'Add this clip to the dataset before annotating it' });
    }
    const existingResult = await client.query(
      `SELECT id FROM ai_annotation_frames
       WHERE dataset_id=$1 AND clip_id=$2 AND frame_time_ms=$3 FOR UPDATE`,
      [input.datasetId, input.clipId, input.frameTimeMs]
    );
    let frameId;
    let version = 1;
    let statusCode = 201;
    if (existingResult.rowCount) {
      frameId = existingResult.rows[0].id;
      const previous = await frameWithAnnotations(client, frameId, { lock: true });
      await snapshotFrame(client, previous, 'updated');
      version = previous.version + 1;
      statusCode = 200;
      await client.query(
        `UPDATE ai_annotation_frames SET
           frame_width=$2, frame_height=$3, review_status=$4,
           defensive_front=$5, box_count=$6, coverage_shell=$7,
           blitz_look=$8, corner_leverage=$9, safety_rotation=$10,
           notes=$11, version=$12, updated_at=NOW()
         WHERE id=$1`,
        [frameId, input.frameWidth, input.frameHeight, input.reviewStatus,
          input.defensiveFront, input.boxCount, input.coverageShell,
          input.blitzLook, input.cornerLeverage, input.safetyRotation,
          input.notes, version]
      );
      await client.query('DELETE FROM ai_annotations WHERE frame_id=$1', [frameId]);
    } else {
      const inserted = await client.query(
        `INSERT INTO ai_annotation_frames (
           dataset_id, clip_id, frame_time_ms, frame_width, frame_height,
           review_status, defensive_front, box_count, coverage_shell,
           blitz_look, corner_leverage, safety_rotation, notes
         ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
         RETURNING id`,
        [input.datasetId, input.clipId, input.frameTimeMs, input.frameWidth,
          input.frameHeight, input.reviewStatus, input.defensiveFront,
          input.boxCount, input.coverageShell, input.blitzLook,
          input.cornerLeverage, input.safetyRotation, input.notes]
      );
      frameId = inserted.rows[0].id;
    }
    for (const annotation of input.annotations) {
      await client.query(
        `INSERT INTO ai_annotations
           (frame_id, class_index, class_name, x, y, width, height, attributes, version)
         VALUES ($1,$2,$3,$4,$5,$6,$7,$8::jsonb,$9)`,
        [frameId, annotation.classIndex, annotation.className, annotation.x,
          annotation.y, annotation.width, annotation.height,
          JSON.stringify(annotation.attributes), version]
      );
    }
    await client.query('UPDATE ai_datasets SET updated_at=NOW() WHERE id=$1', [input.datasetId]);
    const output = await frameWithAnnotations(client, frameId);
    await client.query('COMMIT');
    res.status(statusCode).json(output);
  } catch (error) {
    await client.query('ROLLBACK').catch(() => {});
    next(error);
  } finally {
    client.release();
  }
});

app.delete('/api/training/frames/:id', async (req, res, next) => {
  const client = await db.connect();
  try {
    const id = positiveId(req.params.id, 'Frame ID');
    await client.query('BEGIN');
    const frame = await frameWithAnnotations(client, id, { lock: true });
    if (!frame) {
      await client.query('ROLLBACK');
      return res.status(404).json({ error: 'Annotation frame not found' });
    }
    await snapshotFrame(client, frame, 'deleted');
    await client.query('DELETE FROM ai_annotation_frames WHERE id=$1', [id]);
    await client.query('UPDATE ai_datasets SET updated_at=NOW() WHERE id=$1', [frame.dataset_id]);
    await client.query('COMMIT');
    res.json({ removed_frame_id: id, annotations_removed: frame.annotations.length, history_preserved: true });
  } catch (error) {
    await client.query('ROLLBACK').catch(() => {});
    next(error);
  } finally {
    client.release();
  }
});

app.get('/api/training/frames/:id/history', async (req, res, next) => {
  try {
    const id = positiveId(req.params.id, 'Frame ID');
    const result = await db.query(
      `SELECT id, frame_id, dataset_id, clip_id, frame_time_ms,
              version, snapshot, action, created_at
       FROM ai_annotation_versions WHERE frame_id=$1
       ORDER BY version DESC, created_at DESC`,
      [id]
    );
    res.json(result.rows);
  } catch (error) {
    next(error);
  }
});

async function trainingDashboard(datasetId = null) {
  const values = datasetId ? [datasetId] : [];
  const datasetFilter = datasetId ? 'WHERE id=$1' : '';
  const frameFilter = datasetId ? 'WHERE dataset_id=$1' : '';
  const annotationFilter = datasetId ? 'WHERE f.dataset_id=$1' : '';
  const [datasets, clips, frames, boxes, classes, statuses] = await Promise.all([
    db.query(`SELECT COUNT(*)::int AS count FROM ai_datasets ${datasetFilter}`, values),
    db.query(`SELECT COUNT(DISTINCT clip_id)::int AS count FROM ai_dataset_clips ${frameFilter}`, values),
    db.query(
      `SELECT COUNT(*)::int AS annotated,
              COUNT(*) FILTER (WHERE review_status='verified')::int AS verified
       FROM ai_annotation_frames ${frameFilter}`,
      values
    ),
    db.query(
      `SELECT COUNT(*)::int AS count FROM ai_annotations a
       JOIN ai_annotation_frames f ON f.id=a.frame_id ${annotationFilter}`,
      values
    ),
    db.query(
      `SELECT a.class_index, a.class_name, COUNT(*)::int AS count
       FROM ai_annotations a JOIN ai_annotation_frames f ON f.id=a.frame_id
       ${annotationFilter}
       GROUP BY a.class_index, a.class_name ORDER BY a.class_index`,
      values
    ),
    db.query(
      `SELECT review_status, COUNT(*)::int AS count
       FROM ai_annotation_frames ${frameFilter}
       GROUP BY review_status`,
      values
    )
  ]);
  const classCountMap = Object.fromEntries(DEFENSIVE_OBJECT_CLASSES.map((name) => [name, 0]));
  for (const row of classes.rows) classCountMap[row.class_name] = row.count;
  const annotatedFrames = frames.rows[0].annotated;
  const verifiedFrames = frames.rows[0].verified;
  return {
    thresholds: ANNOTATION_READINESS_TARGETS,
    dataset_count: datasets.rows[0].count,
    clips_assigned: clips.rows[0].count,
    annotated_frames: annotatedFrames,
    verified_frames: verifiedFrames,
    total_box_count: boxes.rows[0].count,
    class_counts: DEFENSIVE_OBJECT_CLASSES.map((name, index) => ({
      class_index: index,
      class_name: name,
      count: classCountMap[name],
      low_sample: classCountMap[name] < ANNOTATION_READINESS_TARGETS.boxesPerClass
    })),
    class_balance_warnings: DEFENSIVE_OBJECT_CLASSES
      .filter((name) => classCountMap[name] < ANNOTATION_READINESS_TARGETS.boxesPerClass)
      .map((name) => `${name} needs ${ANNOTATION_READINESS_TARGETS.boxesPerClass - classCountMap[name]} more boxes`),
    review_status_counts: Object.fromEntries(
      ['draft', 'reviewed', 'verified'].map((status) => [
        status,
        statuses.rows.find((row) => row.review_status === status)?.count || 0
      ])
    ),
    readiness_percentage: readinessFromCounts({
      annotatedFrames,
      verifiedFrames,
      classCounts: classCountMap
    })
  };
}

app.get('/api/training/dashboard', async (req, res, next) => {
  try {
    const datasetId = req.query.datasetId ? positiveId(req.query.datasetId, 'datasetId') : null;
    res.json(await trainingDashboard(datasetId));
  } catch (error) {
    next(error);
  }
});

app.get('/api/training/models', async (req, res, next) => {
  try {
    const datasetId = req.query.datasetId ? positiveId(req.query.datasetId, 'datasetId') : null;
    const dashboard = await trainingDashboard(datasetId);
    const names = [
      'Defensive Player & Football Detection',
      'Defensive Front Classification',
      'Blitz Detection',
      'Coverage Classification',
      'Safety Rotation',
      'Corner Leverage'
    ];
    res.json(names.map((name) => ({
      name,
      status: 'not_connected',
      version: null,
      accuracy: null,
      last_trained_at: null,
      readiness_percentage: dashboard.readiness_percentage,
      actions_enabled: false,
      disabled_reason: 'Model training and deployment are not connected in this annotation phase'
    })));
  } catch (error) {
    next(error);
  }
});

app.get('/api/training/ai-status', async (_req, res) => {
  try {
    const response = await fetchAiService('/model-status');
    const body = await response.json().catch(() => ({}));
    if (!response.ok) {
      return res.json({
        connected: false,
        message: body.detail || 'AI service is unavailable',
        model_loaded: false
      });
    }
    res.json({ connected: true, ...body });
  } catch (error) {
    res.json({
      connected: false,
      message: error.name === 'AbortError'
        ? 'AI service status request timed out'
        : 'AI service is offline',
      model_loaded: false
    });
  }
});

app.post('/api/training/detect-frame', (req, res, next) => {
  detectionUpload.single('image')(req, res, async (uploadError) => {
    if (uploadError) {
      const status = uploadError instanceof multer.MulterError && uploadError.code === 'LIMIT_FILE_SIZE'
        ? 413
        : uploadError.statusCode || 400;
      return res.status(status).json({ error: uploadError.message });
    }
    try {
      if (!req.file?.buffer?.length) return res.status(400).json({ error: 'Image is required' });
      const confidence = boundedUnitOption(req.body?.confidence, 'confidence');
      const iou = boundedUnitOption(req.body?.iou, 'iou');
      const form = new FormData();
      form.append('image', new Blob([req.file.buffer], { type: req.file.mimetype }), req.file.originalname || 'frame.jpg');
      if (confidence !== null) form.append('confidence', String(confidence));
      if (iou !== null) form.append('iou', String(iou));
      const response = await fetchAiService('/detect/frame', { method: 'POST', body: form });
      const body = await response.json().catch(() => ({}));
      if (!response.ok) {
        const message = typeof body.detail === 'string'
          ? body.detail
          : body.error || 'Frame detection failed';
        return res.status(response.status).json({ error: message });
      }
      res.json(body);
    } catch (error) {
      if (error.statusCode) return next(error);
      res.status(503).json({
        error: error.name === 'AbortError'
          ? 'Frame detection timed out'
          : 'AI detection service is unavailable'
      });
    }
  });
});

app.get('/api/summary', async (req, res, next) => {
  try {
    const team = text(req.query.team);
    const gameName = text(req.query.gameName);
    const values = [];
    const where = [];
    if (team) {
      values.push(team);
      where.push(`team=$${values.length}`);
    }
    if (gameName) {
      values.push(gameName);
      where.push(`game_name=$${values.length}`);
    }
    const result = await db.query(
      `SELECT
         COUNT(*) FILTER (WHERE status='needs_labeling')::int AS needs_labeling,
         COUNT(*) FILTER (WHERE status='labeled')::int AS labeled,
         COUNT(*) FILTER (WHERE status='skipped')::int AS skipped,
         COUNT(*) FILTER (WHERE queue_status='queued')::int AS queued,
         COUNT(*) FILTER (WHERE queue_status='not_queued')::int AS not_queued,
         COUNT(*) FILTER (WHERE queue_status='queued' AND status='needs_labeling')::int AS queue_ready,
         COUNT(*) FILTER (WHERE film_side='needs_review')::int AS needs_review,
         COUNT(*) FILTER (WHERE film_side='offense')::int AS offense,
         COUNT(*) FILTER (WHERE film_side='defense')::int AS defense,
         COUNT(*)::int AS total
       FROM clips
       ${where.length ? `WHERE ${where.join(' AND ')}` : ''}`,
      values
    );
    res.json(result.rows[0]);
  } catch (error) {
    next(error);
  }
});

app.get('/api/model/status', async (req, res, next) => {
  try {
    const team = text(req.query.team);
    const values = [];
    const where = [];
    if (team) {
      values.push(team);
      where.push(`team=$${values.length}`);
    }
    const clause = where.length ? `WHERE ${where.join(' AND ')}` : '';
    const counts = await db.query(
      `SELECT COUNT(*)::int AS events FROM ai_training_events ${clause}`,
      values
    );
    const models = await db.query(
      `SELECT COUNT(*)::int AS learned_rules,
              COALESCE(SUM(sample_count),0)::int AS weighted_samples
       FROM ai_model_counts ${clause}`,
      values
    );
    const trainingEvents = counts.rows[0].events;
    res.json({
      modelVersion: 'persistent-categorical-v1',
      persistedIn: 'PostgreSQL',
      trainingEvents,
      learnedRules: models.rows[0].learned_rules,
      weightedSamples: models.rows[0].weighted_samples,
      survivesDeploys: true,
      readiness: trainingEvents >= 250
        ? 'Independent review candidate'
        : trainingEvents >= 100
          ? 'Advanced supervised training'
          : trainingEvents >= 25
            ? 'Early supervised training'
            : 'Building foundation',
      progressPercentage: Math.min(100, Math.round((trainingEvents / 250) * 100))
    });
  } catch (error) {
    next(error);
  }
});

app.get('/api/dashboard', async (req, res, next) => {
  try {
    const team = text(req.query.team);
    const gameName = text(req.query.gameName);
    const filterValues = [];
    const filterWhere = [];
    if (team) {
      filterValues.push(team);
      filterWhere.push(`team=$${filterValues.length}`);
    }
    if (gameName) {
      filterValues.push(gameName);
      filterWhere.push(`game_name=$${filterValues.length}`);
    }

    const summaryResult = await db.query(
      `SELECT
         COUNT(*)::int AS total,
         COUNT(*) FILTER (WHERE status='labeled')::int AS labeled,
         COUNT(*) FILTER (WHERE status='needs_labeling' AND queue_status='queued')::int AS queue_ready,
         COUNT(*) FILTER (WHERE film_side='needs_review')::int AS needs_review,
         COUNT(*) FILTER (WHERE created_at >= date_trunc('year', NOW()))::int AS uploads_this_season
       FROM clips
       ${filterWhere.length ? `WHERE ${filterWhere.join(' AND ')}` : ''}`,
      filterValues
    );

    const recentGamesValues = [];
    const recentGamesWhere = [`game_name<>''`];
    if (team) {
      recentGamesValues.push(team);
      recentGamesWhere.push(`team=$${recentGamesValues.length}`);
    }
    const recentGames = await db.query(
      `SELECT team, game_name, MAX(opponent) AS opponent, MAX(game_date) AS game_date,
              COUNT(*)::int AS clips,
              COUNT(*) FILTER (WHERE status='labeled')::int AS labeled,
              COUNT(*) FILTER (WHERE film_side='needs_review')::int AS needs_review,
              MAX(created_at) AS updated_at
       FROM clips
       WHERE ${recentGamesWhere.join(' AND ')}
       GROUP BY team, game_name
       ORDER BY MAX(COALESCE(game_date, created_at::date)) DESC
       LIMIT 6`,
      recentGamesValues
    );

    const activityValues = [];
    const activityWhere = [];
    if (team) {
      activityValues.push(team);
      activityWhere.push(`team=$${activityValues.length}`);
    }
    const activity = await db.query(
      `SELECT id, team, game_name, original_name, status, queue_status,
              film_side, created_at, labeled_at
       FROM clips
       ${activityWhere.length ? `WHERE ${activityWhere.join(' AND ')}` : ''}
       ORDER BY GREATEST(created_at, COALESCE(labeled_at, created_at)) DESC
       LIMIT 10`,
      activityValues
    );

    let accuracy = null;
    let model = null;
    let breakdown = null;
    let coachNotes = [];
    if (team) {
      const accuracyValues = [team];
      const accuracyWhere = [`c.status='labeled'`, `c.team=$1`];
      if (gameName) {
        accuracyValues.push(gameName);
        accuracyWhere.push(`c.game_name=$2`);
      }
      const accuracyResult = await db.query(
        `SELECT p.hash,p.ai_hash,p.defense_formation,p.ai_defense_formation,
                p.blitz,p.ai_blitz,p.coverage,p.ai_coverage,
                p.run_direction,p.ai_run_direction
         FROM plays p JOIN clips c ON c.id=p.clip_id
         WHERE ${accuracyWhere.join(' AND ')}`,
        accuracyValues
      );
      let correct = 0;
      let compared = 0;
      for (const row of accuracyResult.rows) {
        const pairs = [
          [row.hash, row.ai_hash],
          [row.defense_formation, row.ai_defense_formation],
          [row.blitz, row.ai_blitz],
          [row.coverage, row.ai_coverage],
          [row.run_direction, row.ai_run_direction]
        ];
        for (const [coachValue, aiValue] of pairs) {
          if (coachValue === null || coachValue === undefined || coachValue === '' ||
              aiValue === null || aiValue === undefined || aiValue === '') continue;
          compared += 1;
          if (lower(coachValue) === lower(aiValue)) correct += 1;
        }
      }
      accuracy = { overallAccuracy: compared ? pct(correct, compared) : null, comparisons: compared };
      breakdown = await getBreakdown(team, gameName);
      coachNotes = buildCoachNotes(breakdown).slice(0, 4);

      const modelCount = await db.query(
        `SELECT COUNT(*)::int AS events FROM ai_training_events WHERE team=$1`,
        [team]
      );
      const events = modelCount.rows[0].events;
      model = {
        trainingEvents: events,
        progressPercentage: Math.min(100, Math.round((events / 250) * 100)),
        readiness: events >= 250 ? 'Independent review candidate' : events >= 100 ? 'Advanced supervised' : events >= 25 ? 'Early supervised' : 'Building foundation'
      };
    }

    let storage = { clipsBytes: 0, clipsGigabytes: 0, fileCount: 0 };
    try {
      const names = fs.readdirSync(CLIP_DIR);
      let bytes = 0;
      let fileCount = 0;
      for (const name of names) {
        const fullPath = path.join(CLIP_DIR, name);
        const stat = fs.statSync(fullPath);
        if (stat.isFile()) {
          bytes += stat.size;
          fileCount += 1;
        }
      }
      storage = {
        clipsBytes: bytes,
        clipsGigabytes: Math.round((bytes / (1024 ** 3)) * 100) / 100,
        fileCount
      };
    } catch (_error) {
      // Keep zeroed storage values if the disk cannot be inspected.
    }

    res.json({
      summary: summaryResult.rows[0],
      recentGames: recentGames.rows,
      activity: activity.rows,
      accuracy,
      model,
      breakdown,
      coachNotes,
      system: {
        api: 'online',
        database: 'connected',
        videoProcessing: 'ready',
        aiModel: 'ready'
      },
      storage
    });
  } catch (error) {
    next(error);
  }
});

app.use(express.static(publicDir));
app.get('*', (_req, res) => res.sendFile(path.join(publicDir, 'index.html')));

app.use((error, _req, res, _next) => {
  console.error(error);
  const statusCode = error.statusCode || (error instanceof multer.MulterError ? 400 : 500);
  res.status(statusCode).json({ error: error.message || 'Server error' });
});

const PORT = Number(process.env.PORT) || 8080;
initDb()
  .then(() => app.listen(PORT, () => console.log(`TCHS Film Tool V1.0 listening on ${PORT}`)))
  .catch((error) => {
    console.error('Database initialization failed', error);
    process.exit(1);
  });
