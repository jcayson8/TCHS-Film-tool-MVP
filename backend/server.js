import express from 'express';
import multer from 'multer';
import path from 'path';
import fs from 'fs';
import cors from 'cors';
import { fileURLToPath } from 'url';
import { Pool } from 'pg';
import { learnFromCorrection, predictClip, savePrediction, rebuildModelCounts } from './analysis.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const app = express();

app.use(cors());
app.use(express.json({ limit: '2mb' }));

const DATA_DIR = process.env.DATA_DIR || '/data';
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

const safeName = (name) => name.replace(/[^a-zA-Z0-9._-]/g, '_');
const storage = multer.diskStorage({
  destination: (_req, _file, cb) => cb(null, CLIP_DIR),
  filename: (_req, file, cb) => cb(null, `${Date.now()}-${Math.random().toString(36).slice(2, 8)}-${safeName(file.originalname)}`)
});
const upload = multer({
  storage,
  limits: { fileSize: 2 * 1024 * 1024 * 1024 },
  fileFilter: (_req, file, cb) => {
    const ok = file.mimetype === 'video/mp4' || file.originalname.toLowerCase().endsWith('.mp4');
    cb(ok ? null : new Error('Only MP4 files are allowed'), ok);
  }
});

async function initDb() {
  await db.query(`
    CREATE TABLE IF NOT EXISTS clips (
      id SERIAL PRIMARY KEY,
      team TEXT NOT NULL,
      game_name TEXT NOT NULL DEFAULT '',
      original_name TEXT NOT NULL,
      stored_name TEXT NOT NULL UNIQUE,
      file_size BIGINT NOT NULL DEFAULT 0,
      status TEXT NOT NULL DEFAULT 'needs_labeling',
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      labeled_at TIMESTAMPTZ
    );
  `);
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
  await db.query(`ALTER TABLE plays ADD COLUMN IF NOT EXISTS ai_hash TEXT;`);
  await db.query(`ALTER TABLE plays ADD COLUMN IF NOT EXISTS ai_defense_formation TEXT;`);
  await db.query(`ALTER TABLE plays ADD COLUMN IF NOT EXISTS ai_blitz BOOLEAN;`);
  await db.query(`ALTER TABLE plays ADD COLUMN IF NOT EXISTS ai_coverage TEXT;`);
  await db.query(`ALTER TABLE plays ADD COLUMN IF NOT EXISTS ai_run_direction TEXT;`);
  await db.query(`ALTER TABLE plays ADD COLUMN IF NOT EXISTS ai_confidence REAL;`);
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
  console.log('Database ready');
}

app.get('/api/health', async (_req, res) => {
  try {
    await db.query('SELECT 1');
    res.json({ status: 'ok', database: 'connected' });
  } catch (error) {
    res.status(503).json({ status: 'error', database: 'disconnected' });
  }
});

app.post('/api/upload', upload.array('files', 100), async (req, res, next) => {
  try {
    const team = String(req.body.team || '').trim();
    const gameName = String(req.body.gameName || '').trim();
    if (!team) {
      for (const file of req.files || []) fs.rmSync(file.path, { force: true });
      return res.status(400).json({ error: 'Team is required' });
    }
    const inserted = [];
    for (const file of req.files || []) {
      const result = await db.query(
        `INSERT INTO clips (team, game_name, original_name, stored_name, file_size)
         VALUES ($1,$2,$3,$4,$5) RETURNING *`,
        [team, gameName, file.originalname, file.filename, file.size]
      );
      const clip = result.rows[0];
      const prediction = await predictClip(db, clip);
      await savePrediction(db, clip, prediction);
      inserted.push({ ...clip, aiPrediction: prediction });
    }
    res.status(201).json({ uploaded: inserted });
  } catch (error) {
    next(error);
  }
});

app.get('/api/clips', async (req, res, next) => {
  try {
    const status = String(req.query.status || '');
    const team = String(req.query.team || '');
    const values = [];
    const where = [];
    if (status) { values.push(status); where.push(`c.status = $${values.length}`); }
    if (team) { values.push(team); where.push(`c.team = $${values.length}`); }
    const gameName = String(req.query.gameName || '');
    if (gameName) { values.push(gameName); where.push(`c.game_name = $${values.length}`); }
    const result = await db.query(
      `SELECT c.*, p.id AS play_id
       FROM clips c LEFT JOIN plays p ON p.clip_id = c.id
       ${where.length ? `WHERE ${where.join(' AND ')}` : ''}
       ORDER BY c.created_at ASC`, values
    );
    res.json(result.rows);
  } catch (error) { next(error); }
});


app.get('/api/games', async (req, res, next) => {
  try {
    const team = String(req.query.team || '');
    const result = team
      ? await db.query(`SELECT game_name, COUNT(*)::int AS clip_count FROM clips WHERE team=$1 AND game_name<>'' GROUP BY game_name ORDER BY game_name`, [team])
      : await db.query(`SELECT team, game_name, COUNT(*)::int AS clip_count FROM clips WHERE game_name<>'' GROUP BY team, game_name ORDER BY team, game_name`);
    res.json(result.rows);
  } catch (error) { next(error); }
});


app.post('/api/games/reset', async (req, res, next) => {
  const team = String(req.body?.team || '').trim();
  const gameName = String(req.body?.gameName || '').trim();
  if (!team || !gameName) return res.status(400).json({ error: 'Team and gameName are required' });

  const client = await db.connect();
  try {
    await client.query('BEGIN');
    const clips = await client.query(
      'SELECT id FROM clips WHERE team=$1 AND game_name=$2 FOR UPDATE',
      [team, gameName]
    );
    if (!clips.rowCount) {
      await client.query('ROLLBACK');
      return res.status(404).json({ error: 'Game not found' });
    }
    const ids = clips.rows.map(row => row.id);
    await client.query('DELETE FROM ai_training_events WHERE team=$1 AND game_name=$2', [team, gameName]);
    await client.query('DELETE FROM plays WHERE clip_id = ANY($1::int[])', [ids]);
    await client.query('DELETE FROM ai_predictions WHERE clip_id = ANY($1::int[])', [ids]);
    await client.query(`UPDATE clips SET status='needs_labeling', labeled_at=NULL WHERE id = ANY($1::int[])`, [ids]);
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
  const team = String(req.query.team || req.body?.team || '').trim();
  const gameName = String(req.query.gameName || req.body?.gameName || '').trim();
  if (!team || !gameName) return res.status(400).json({ error: 'Team and gameName are required' });

  const client = await db.connect();
  try {
    await client.query('BEGIN');
    const clips = await client.query(
      'SELECT id, stored_name FROM clips WHERE team=$1 AND game_name=$2 FOR UPDATE',
      [team, gameName]
    );
    if (!clips.rowCount) {
      await client.query('ROLLBACK');
      return res.status(404).json({ error: 'Game not found' });
    }

    await client.query('DELETE FROM ai_training_events WHERE team=$1 AND game_name=$2', [team, gameName]);
    await client.query('DELETE FROM ai_model_counts WHERE team=$1 AND game_name=$2', [team, gameName]);
    await client.query('DELETE FROM clips WHERE team=$1 AND game_name=$2', [team, gameName]);
    await rebuildModelCounts(client);
    await client.query('COMMIT');

    let deletedFiles = 0;
    for (const clip of clips.rows) {
      const filePath = path.join(CLIP_DIR, clip.stored_name);
      if (fs.existsSync(filePath)) deletedFiles++;
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
  const client = await db.connect();
  try {
    await client.query('BEGIN');
    const result = await client.query('SELECT stored_name FROM clips WHERE id=$1 FOR UPDATE', [req.params.id]);
    if (!result.rowCount) { await client.query('ROLLBACK'); return res.status(404).json({ error: 'Clip not found' }); }
    await client.query('DELETE FROM clips WHERE id=$1', [req.params.id]);
    await client.query('COMMIT');
    fs.rmSync(path.join(CLIP_DIR, result.rows[0].stored_name), { force: true });
    res.json({ deleted: true });
  } catch (error) { await client.query('ROLLBACK'); next(error); } finally { client.release(); }
});

app.get('/api/accuracy', async (req, res, next) => {
  try {
    const team = String(req.query.team || '');
    const gameName = String(req.query.gameName || '');
    const values=[]; const where=[`c.status='labeled'`];
    if(team){values.push(team);where.push(`c.team=$${values.length}`)}
    if(gameName){values.push(gameName);where.push(`c.game_name=$${values.length}`)}
    const result=await db.query(`SELECT p.*, c.labeled_at FROM plays p JOIN clips c ON c.id=p.clip_id WHERE ${where.join(' AND ')} ORDER BY c.labeled_at DESC`,values);
    const rows=result.rows;
    const defs=[
      ['hash','ai_hash'],['defenseFormation','ai_defense_formation'],['blitz','ai_blitz'],['coverage','ai_coverage'],['playDirection','ai_run_direction']
    ];
    const coach={hash:'hash',defenseFormation:'defense_formation',blitz:'blitz',coverage:'coverage',playDirection:'run_direction'};
    const categories={}; let totalCorrect=0,totalCompared=0;
    for(const [name,ai] of defs){let correct=0,compared=0;for(const r of rows){const a=r[ai],c=r[coach[name]];if(a===null||a===undefined||a===''||c===null||c===undefined||c==='')continue;compared++;if(String(a).toLowerCase()===String(c).toLowerCase())correct++;}categories[name]={correct,compared,accuracy:compared?Math.round(correct/compared*1000)/10:null};totalCorrect+=correct;totalCompared+=compared;}
    const recent=rows.slice(0,25);let rc=0,rn=0;for(const r of recent){for(const [name,ai] of defs){const a=r[ai],c=r[coach[name]];if(a===null||a===undefined||a===''||c===null||c===undefined||c==='')continue;rn++;if(String(a).toLowerCase()===String(c).toLowerCase())rc++;}}
    const overall=totalCompared?Math.round(totalCorrect/totalCompared*1000)/10:null;
    const readiness=totalCompared<100?'Not enough data':overall>=90?'Nearly ready':overall>=80?'Needs supervision':'Training needed';
    res.json({overallAccuracy:overall,totalComparisons:totalCompared,labeledClips:rows.length,recentAccuracy:rn?Math.round(rc/rn*1000)/10:null,readiness,categories});
  } catch(error){next(error)}
});

app.get('/api/clips/:id/video', async (req, res, next) => {
  try {
    const result = await db.query('SELECT stored_name FROM clips WHERE id=$1', [req.params.id]);
    if (!result.rowCount) return res.status(404).json({ error: 'Clip not found' });
    const filePath = path.join(CLIP_DIR, result.rows[0].stored_name);
    if (!fs.existsSync(filePath)) return res.status(404).json({ error: 'Video file not found' });

    const stat = fs.statSync(filePath);
    const range = req.headers.range;
    if (!range) {
      res.writeHead(200, { 'Content-Length': stat.size, 'Content-Type': 'video/mp4', 'Accept-Ranges': 'bytes' });
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
    fs.createReadStream(filePath, { start, end }).pipe(res);
  } catch (error) { next(error); }
});

app.get('/api/clips/:id/label', async (req, res, next) => {
  try {
    const result = await db.query('SELECT * FROM plays WHERE clip_id=$1', [req.params.id]);
    res.json(result.rows[0] || null);
  } catch (error) { next(error); }
});

app.get('/api/clips/:id/prediction', async (req, res, next) => {
  try {
    const result = await db.query('SELECT * FROM ai_predictions WHERE clip_id=$1', [req.params.id]);
    res.json(result.rows[0] || null);
  } catch (error) { next(error); }
});

app.put('/api/clips/:id/label', async (req, res, next) => {
  const client = await db.connect();
  try {
    await client.query('BEGIN');
    const clipResult = await client.query('SELECT * FROM clips WHERE id=$1 FOR UPDATE', [req.params.id]);
    if (!clipResult.rowCount) {
      await client.query('ROLLBACK');
      return res.status(404).json({ error: 'Clip not found' });
    }
    const c = clipResult.rows[0];
    const b = req.body || {};
    const values = [
      c.id, c.team, c.game_name, c.original_name,
      b.down || null, b.distance || null, b.hash || null,
      b.playType || null, b.playCall || null, b.offenseFormation || null,
      b.defenseFormation || null, b.blitz === true, b.coverage || null,
      b.runDirection || null, b.passDepth || null,
      typeof b.completed === 'boolean' ? b.completed : null,
      b.notes || null
    ];
    const result = await client.query(`
      INSERT INTO plays (
        clip_id, team, game_name, clip, down, distance, hash, play_type, play_call,
        offense_formation, defense_formation, blitz, coverage, run_direction,
        pass_depth, completed, notes, label_source, updated_at
      ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,'coach',NOW())
      ON CONFLICT (clip_id) DO UPDATE SET
        down=EXCLUDED.down, distance=EXCLUDED.distance, hash=EXCLUDED.hash,
        play_type=EXCLUDED.play_type, play_call=EXCLUDED.play_call,
        offense_formation=EXCLUDED.offense_formation,
        defense_formation=EXCLUDED.defense_formation, blitz=EXCLUDED.blitz,
        coverage=EXCLUDED.coverage, run_direction=EXCLUDED.run_direction,
        pass_depth=EXCLUDED.pass_depth, completed=EXCLUDED.completed,
        notes=EXCLUDED.notes, label_source='coach', updated_at=NOW()
      RETURNING *`, values);
    await learnFromCorrection(client, c, result.rows[0]);
    await client.query("UPDATE clips SET status='labeled', labeled_at=NOW() WHERE id=$1", [c.id]);
    await client.query('COMMIT');
    res.json(result.rows[0]);
  } catch (error) {
    await client.query('ROLLBACK');
    next(error);
  } finally { client.release(); }
});

app.patch('/api/clips/:id/status', async (req, res, next) => {
  try {
    const allowed = new Set(['needs_labeling', 'labeled', 'skipped']);
    if (!allowed.has(req.body.status)) return res.status(400).json({ error: 'Invalid status' });
    const result = await db.query('UPDATE clips SET status=$1 WHERE id=$2 RETURNING *', [req.body.status, req.params.id]);
    if (!result.rowCount) return res.status(404).json({ error: 'Clip not found' });
    res.json(result.rows[0]);
  } catch (error) { next(error); }
});

app.get('/api/plays', async (req, res, next) => {
  try {
    const team = String(req.query.team || '');
    const result = team
      ? await db.query('SELECT * FROM plays WHERE team=$1 ORDER BY updated_at DESC', [team])
      : await db.query('SELECT * FROM plays ORDER BY updated_at DESC');
    res.json(result.rows);
  } catch (error) { next(error); }
});

app.get('/api/summary', async (_req, res, next) => {
  try {
    const result = await db.query(`
      SELECT
        COUNT(*) FILTER (WHERE status='needs_labeling')::int AS needs_labeling,
        COUNT(*) FILTER (WHERE status='labeled')::int AS labeled,
        COUNT(*) FILTER (WHERE status='skipped')::int AS skipped,
        COUNT(*)::int AS total
      FROM clips`);
    res.json(result.rows[0]);
  } catch (error) { next(error); }
});

app.get('/api/model/status', async (req, res, next) => {
  try {
    const team = String(req.query.team || '');
    const values=[]; const where=[];
    if (team) { values.push(team); where.push(`team=$${values.length}`); }
    const counts = await db.query(`SELECT COUNT(*)::int AS events FROM ai_training_events ${where.length?'WHERE '+where.join(' AND '):''}`, values);
    const models = await db.query(`SELECT COUNT(*)::int AS learned_rules, COALESCE(SUM(sample_count),0)::int AS weighted_samples FROM ai_model_counts ${where.length?'WHERE '+where.join(' AND '):''}`, values);
    res.json({
      modelVersion:'persistent-categorical-v1',
      persistedIn:'PostgreSQL',
      trainingEvents:counts.rows[0].events,
      learnedRules:models.rows[0].learned_rules,
      weightedSamples:models.rows[0].weighted_samples,
      survivesDeploys:true
    });
  } catch (error) { next(error); }
});

const publicDir = path.join(__dirname, 'public');
app.use(express.static(publicDir));
app.get('*', (_req, res) => res.sendFile(path.join(publicDir, 'index.html')));

app.use((error, _req, res, _next) => {
  console.error(error);
  res.status(error instanceof multer.MulterError ? 400 : 500).json({ error: error.message || 'Server error' });
});

const PORT = Number(process.env.PORT) || 8080;
initDb()
  .then(() => app.listen(PORT, () => console.log(`TCHS Film Tool listening on ${PORT}`)))
  .catch((error) => { console.error('Database initialization failed', error); process.exit(1); });
