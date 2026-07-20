import express from 'express';
import multer from 'multer';
import path from 'path';
import fs from 'fs';
import cors from 'cors';
import { fileURLToPath } from 'url';
import { Pool } from 'pg';
import { learnFromCorrection, predictClip, savePrediction, rebuildModelCounts } from './analysis.js';
import { classifyPossession } from './teamIdentity.js';

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
      film_side TEXT NOT NULL DEFAULT 'needs_review',
      possession_confidence INTEGER,
      possession_reason TEXT,
      jersey_color TEXT,
      helmet_color TEXT,
      home_away TEXT,
      use_for_ai BOOLEAN NOT NULL DEFAULT TRUE,
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      labeled_at TIMESTAMPTZ
    );
  `);
  await db.query(`ALTER TABLE clips ADD COLUMN IF NOT EXISTS film_side TEXT NOT NULL DEFAULT 'needs_review';`);
  await db.query(`ALTER TABLE clips ADD COLUMN IF NOT EXISTS possession_confidence INTEGER;`);
  await db.query(`ALTER TABLE clips ADD COLUMN IF NOT EXISTS possession_reason TEXT;`);
  await db.query(`ALTER TABLE clips ADD COLUMN IF NOT EXISTS jersey_color TEXT;`);
  await db.query(`ALTER TABLE clips ADD COLUMN IF NOT EXISTS helmet_color TEXT;`);
  await db.query(`ALTER TABLE clips ADD COLUMN IF NOT EXISTS home_away TEXT;`);
  await db.query(`ALTER TABLE clips ADD COLUMN IF NOT EXISTS use_for_ai BOOLEAN NOT NULL DEFAULT TRUE;`);
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
    const autoLabel = String(req.body.autoLabel || '').toLowerCase() === 'true';
    const autoSortPossession = String(req.body.autoSortPossession || '').toLowerCase() === 'true';
    const jerseyColor = String(req.body.jerseyColor || '').toLowerCase();
    const helmetColor = String(req.body.helmetColor || '').toLowerCase();
    const homeAway = ['home','away'].includes(String(req.body.homeAway || '').toLowerCase()) ? String(req.body.homeAway).toLowerCase() : '';
    const offenseOnlyAi = String(req.body.offenseOnlyAi || '').toLowerCase() === 'true';
    if (!team) {
      for (const file of req.files || []) fs.rmSync(file.path, { force: true });
      return res.status(400).json({ error: 'Team is required' });
    }
    const inserted = [];
    for (const file of req.files || []) {
      const possession = autoSortPossession
        ? await classifyPossession(file.path, { jerseyColor, helmetColor, homeAway })
        : { filmSide: 'needs_review', confidence: 0, reason: 'Automatic possession sorting was not selected' };
      const filmSide = possession.filmSide;
      const useForAi = !offenseOnlyAi || filmSide === 'offense';
      const result = await db.query(
        `INSERT INTO clips (team, game_name, original_name, stored_name, file_size, film_side, possession_confidence, possession_reason, jersey_color, helmet_color, home_away, use_for_ai)
         VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12) RETURNING *`,
        [team, gameName, file.originalname, file.filename, file.size, filmSide, possession.confidence, possession.reason, jerseyColor, helmetColor, homeAway, useForAi]
      );
      const clip = result.rows[0];
      const prediction = useForAi ? await predictClip(db, clip) : null;
      if (prediction) await savePrediction(db, clip, prediction);
      if (autoLabel && prediction) {
        await db.query(`
          UPDATE plays SET
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
          [clip.id, prediction.hash, prediction.defense_formation,
           prediction.blitz, prediction.coverage, prediction.run_direction,
           prediction.overall_confidence]
        );
      }
      inserted.push({ ...clip, autoLabel: autoLabel && useForAi, aiPrediction: prediction, offenseOnlyAi, possession });
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
    const filmSide = String(req.query.filmSide || '').trim().toLowerCase();
    if (filmSide) { values.push(filmSide); where.push(`c.film_side = $${values.length}`); }
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
    const correctedSide = ['offense','defense','needs_review'].includes(String(b.filmSide||'')) ? String(b.filmSide) : c.film_side;
    const correctedUseForAi = correctedSide === 'offense';
    if (correctedUseForAi) await learnFromCorrection(client, {...c, use_for_ai:true}, result.rows[0]);
    await client.query("UPDATE clips SET status='labeled', labeled_at=NOW(), film_side=$2, use_for_ai=$3, possession_reason='Coach verified' WHERE id=$1", [c.id, correctedSide, correctedUseForAi]);
    await client.query('COMMIT');
    res.json(result.rows[0]);
  } catch (error) {
    await client.query('ROLLBACK');
    next(error);
  } finally { client.release(); }
});

app.patch('/api/clips/:id/film-side', async (req, res, next) => {
  try {
    const id = Number(req.params.id);
    const filmSide = String(req.body?.filmSide || '').toLowerCase();
    if (!['offense','defense','needs_review'].includes(filmSide)) return res.status(400).json({ error: 'Invalid filmSide' });
    const useForAi = filmSide === 'offense';
    const result = await db.query(
      `UPDATE clips SET film_side=$2, use_for_ai=$3, possession_confidence=100, possession_reason='Coach corrected in Team Identity Review' WHERE id=$1 RETURNING *`,
      [id, filmSide, useForAi]
    );
    if (!result.rowCount) return res.status(404).json({ error: 'Clip not found' });
    res.json(result.rows[0]);
  } catch (error) { next(error); }
});

app.post('/api/clips/:id/retry-processing', async (req, res, next) => {
  try {
    const id = Number(req.params.id);
    const result = await db.query('SELECT * FROM clips WHERE id=$1', [id]);
    if (!result.rowCount) return res.status(404).json({ error: 'Clip not found' });
    const clip = result.rows[0];
    const filePath = path.join(CLIP_DIR, clip.stored_name);
    if (!fs.existsSync(filePath)) return res.status(404).json({ error: 'Video file not found' });
    const possession = await classifyPossession(filePath, { jerseyColor: clip.jersey_color, helmetColor: clip.helmet_color, homeAway: clip.home_away });
    const useForAi = possession.filmSide === 'offense';
    const updated = await db.query(`UPDATE clips SET film_side=$2, possession_confidence=$3, possession_reason=$4, use_for_ai=$5 WHERE id=$1 RETURNING *`, [id, possession.filmSide, possession.confidence, possession.reason, useForAi]);
    if (useForAi) {
      const prediction = await predictClip(db, updated.rows[0]);
      if (prediction) await savePrediction(db, updated.rows[0], prediction);
    }
    res.json({ clip: updated.rows[0], retried: true });
  } catch (error) { next(error); }
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



app.get('/api/formations', async (req, res, next) => {
  try {
    const team = String(req.query.team || '').trim();
    if (!team) return res.json({ offense: [], defense: [] });
    const result = await db.query(`
      SELECT offense_formation, defense_formation
      FROM plays
      WHERE team=$1 AND label_source='coach'
      ORDER BY updated_at DESC`, [team]);
    const unique = key => [...new Set(result.rows.map(r => String(r[key] || '').trim()).filter(Boolean))]
      .sort((a,b) => a.localeCompare(b));
    res.json({ offense: unique('offense_formation'), defense: unique('defense_formation') });
  } catch (error) { next(error); }
});

app.get('/api/formation-library', async (req, res, next) => {
  try {
    const team = String(req.query.team || '').trim();
    if (!team) return res.status(400).json({ error: 'Team is required' });
    const result = await db.query(`
      SELECT offense_formation, defense_formation
      FROM plays WHERE team=$1 AND label_source='coach'`, [team]);
    const build = key => {
      const counts = new Map();
      for (const row of result.rows) {
        const value = String(row[key] || '').trim();
        if (value) counts.set(value, (counts.get(value) || 0) + 1);
      }
      const total = [...counts.values()].reduce((a,b)=>a+b,0);
      return [...counts.entries()].map(([name,count]) => ({
        name, count, percentage: total ? Math.round(count / total * 1000) / 10 : 0
      })).sort((a,b)=>b.count-a.count || a.name.localeCompare(b.name));
    };
    res.json({ offense: build('offense_formation'), defense: build('defense_formation') });
  } catch (error) { next(error); }
});

app.post('/api/formations/rename', async (req, res, next) => {
  try {
    const team=String(req.body?.team||'').trim();
    const side=String(req.body?.side||'').toLowerCase();
    const from=String(req.body?.from||'').trim();
    const to=String(req.body?.to||'').trim();
    if(!team || !['offense','defense'].includes(side) || !from || !to) return res.status(400).json({error:'Team, side, current name, and new name are required'});
    const column=side==='offense'?'offense_formation':'defense_formation';
    const result=await db.query(`UPDATE plays SET ${column}=$1, updated_at=NOW() WHERE team=$2 AND LOWER(TRIM(${column}))=LOWER($3) RETURNING clip_id`,[to,team,from]);
    res.json({updated:result.rowCount});
  } catch(error){next(error)}
});

app.post('/api/formations/merge', async (req, res, next) => {
  try {
    const team=String(req.body?.team||'').trim();
    const side=String(req.body?.side||'').toLowerCase();
    const sources=Array.isArray(req.body?.sources)?req.body.sources.map(x=>String(x).trim()).filter(Boolean):[];
    const target=String(req.body?.target||'').trim();
    if(!team || !['offense','defense'].includes(side) || !sources.length || !target) return res.status(400).json({error:'Team, side, source formations, and target are required'});
    const column=side==='offense'?'offense_formation':'defense_formation';
    const result=await db.query(`UPDATE plays SET ${column}=$1, updated_at=NOW() WHERE team=$2 AND ${column}=ANY($3::text[]) RETURNING clip_id`,[target,team,sources]);
    res.json({updated:result.rowCount});
  } catch(error){next(error)}
});

app.delete('/api/formations', async (req, res, next) => {
  try {
    const team=String(req.body?.team||'').trim();
    const side=String(req.body?.side||'').toLowerCase();
    const name=String(req.body?.name||'').trim();
    if(!team || !['offense','defense'].includes(side) || !name) return res.status(400).json({error:'Team, side, and formation are required'});
    const column=side==='offense'?'offense_formation':'defense_formation';
    const result=await db.query(`UPDATE plays SET ${column}=NULL, updated_at=NOW() WHERE team=$1 AND LOWER(TRIM(${column}))=LOWER($2) RETURNING clip_id`,[team,name]);
    res.json({updated:result.rowCount});
  } catch(error){next(error)}
});

app.get('/api/formation-matchups', async (req, res, next) => {
  try {
    const team=String(req.query.team||'').trim();
    const gameName=String(req.query.gameName||'').trim();
    const selected=String(req.query.offenseFormation||'').trim();
    if(!team) return res.status(400).json({error:'Team is required'});
    const values=[team]; const where=[`team=$1`,`label_source='coach'`,`COALESCE(TRIM(offense_formation),'')<>''`];
    if(gameName){values.push(gameName);where.push(`game_name=$${values.length}`)}
    const result=await db.query(`SELECT clip_id,game_name,down,distance,hash,play_type,play_call,offense_formation,defense_formation,coverage,blitz,run_direction FROM plays WHERE ${where.join(' AND ')} ORDER BY updated_at DESC`,values);
    const plays=result.rows;
    const pct=(n,d)=>d?Math.round(n/d*1000)/10:0;
    const group=(rows,key)=>{
      const m=new Map();
      rows.forEach(r=>{const v=String(r[key]||'').trim();if(v)m.set(v,(m.get(v)||0)+1)});
      return [...m.entries()].map(([name,count])=>({name,count,percentage:pct(count,rows.length)})).sort((a,b)=>b.count-a.count||a.name.localeCompare(b.name));
    };
    const formations=group(plays,'offense_formation');
    const active=selected || formations[0]?.name || '';
    const focus=plays.filter(p=>p.offense_formation===active);
    const manZoneOf=c=>{const v=String(c||'').toLowerCase();if(!v)return'';if(v.includes('man')||v.includes('cover 0')||v.includes('cover 1'))return'Man';if(v.includes('zone')||/cover\s*[2346]/.test(v))return'Zone';return'Other'};
    const manZoneMap=new Map();focus.forEach(p=>{const v=manZoneOf(p.coverage);if(v)manZoneMap.set(v,(manZoneMap.get(v)||0)+1)});
    const manZone=[...manZoneMap.entries()].map(([name,count])=>({name,count,percentage:pct(count,focus.length)})).sort((a,b)=>b.count-a.count);
    const summary=formations.map(f=>{
      const rows=plays.filter(p=>p.offense_formation===f.name);
      const fronts=group(rows,'defense_formation');const cov=group(rows,'coverage');
      const mz=new Map();rows.forEach(p=>{const v=manZoneOf(p.coverage);if(v)mz.set(v,(mz.get(v)||0)+1)});
      return {offenseFormation:f.name,snaps:rows.length,percentage:f.percentage,topDefense:fronts[0]||null,topCoverage:cov[0]||null,manPercentage:pct(mz.get('Man')||0,rows.length),zonePercentage:pct(mz.get('Zone')||0,rows.length)};
    });
    const downBreakdown=[1,2,3,4].map(d=>({down:d,snaps:focus.filter(p=>Number(p.down)===d).length}));
    const uniqueDefenses=[...new Set(plays.map(p=>p.defense_formation).filter(Boolean))];
    const uniqueCoverage=[...new Set(plays.map(p=>p.coverage).filter(Boolean))];
    res.json({team,gameName:gameName||null,totalSnaps:plays.length,runSnaps:plays.filter(p=>String(p.play_type).toLowerCase()==='run').length,passSnaps:plays.filter(p=>['pass','screen'].includes(String(p.play_type).toLowerCase())).length,uniqueOffenseFormations:formations.length,uniqueDefensiveFronts:uniqueDefenses.length,uniqueCoverages:uniqueCoverage.length,formations,selectedFormation:active,selectedSnaps:focus.length,defensiveResponses:group(focus,'defense_formation'),coverageResponses:group(focus,'coverage'),manZone,summary,downBreakdown});
  } catch(error){next(error)}
});

app.get('/api/breakdown', async (req, res, next) => {
  try {
    const team = String(req.query.team || '').trim();
    const gameName = String(req.query.gameName || '').trim();
    if (!team) return res.status(400).json({ error: 'Team is required' });

    const values = [team];
    const where = ['team = $1'];
    if (gameName) { values.push(gameName); where.push(`game_name = $${values.length}`); }

    const result = await db.query(
      `SELECT game_name, down, hash, play_type, defense_formation, blitz, coverage, run_direction, pass_depth, completed
       FROM plays
       WHERE ${where.join(' AND ')}
       ORDER BY created_at ASC`, values
    );
    const plays = result.rows;
    const pct = (n, d) => d ? Math.round((n / d) * 1000) / 10 : 0;
    const countBy = (key, predicate = () => true) => {
      const out = {};
      for (const play of plays) {
        if (!predicate(play)) continue;
        const value = play[key] || 'Unlabeled';
        out[value] = (out[value] || 0) + 1;
      }
      return Object.entries(out)
        .map(([name, count]) => ({ name, count, percentage: pct(count, plays.filter(predicate).length) }))
        .sort((a, b) => b.count - a.count || a.name.localeCompare(b.name));
    };

    const runs = plays.filter(p => String(p.play_type || '').toLowerCase() === 'run');
    const passes = plays.filter(p => ['pass', 'screen'].includes(String(p.play_type || '').toLowerCase()));
    const rpos = plays.filter(p => String(p.play_type || '').toLowerCase() === 'rpo');
    const blitzes = plays.filter(p => p.blitz === true);
    const blitzRuns = runs.filter(p => p.blitz === true);
    const blitzPasses = passes.filter(p => p.blitz === true);

    const downBreakdown = [1,2,3,4].map(down => {
      const onDown = plays.filter(p => Number(p.down) === down);
      const downBlitzes = onDown.filter(p => p.blitz === true).length;
      return { down, plays: onDown.length, blitzes: downBlitzes, blitzPercentage: pct(downBlitzes, onDown.length) };
    });

    res.json({
      team,
      gameName: gameName || null,
      games: [...new Set(plays.map(p => p.game_name).filter(Boolean))].sort(),
      totalPlays: plays.length,
      run: { count: runs.length, percentage: pct(runs.length, plays.length) },
      pass: { count: passes.length, percentage: pct(passes.length, plays.length) },
      rpo: { count: rpos.length, percentage: pct(rpos.length, plays.length) },
      blitz: { count: blitzes.length, percentage: pct(blitzes.length, plays.length) },
      blitzVsRun: { count: blitzRuns.length, opportunities: runs.length, percentage: pct(blitzRuns.length, runs.length) },
      blitzVsPass: { count: blitzPasses.length, opportunities: passes.length, percentage: pct(blitzPasses.length, passes.length) },
      defenses: countBy('defense_formation', p => Boolean(p.defense_formation)),
      coverages: countBy('coverage', p => Boolean(p.coverage)),
      directions: countBy('run_direction', p => Boolean(p.run_direction)),
      hashes: countBy('hash', p => Boolean(p.hash)),
      passDepths: countBy('pass_depth', p => Boolean(p.pass_depth) && ['pass','screen'].includes(String(p.play_type || '').toLowerCase())),
      downBreakdown
    });
  } catch (error) { next(error); }
});


app.get('/api/labeled-clips', async (req, res, next) => {
  try {
    const values = [];
    const where = ["c.status = 'labeled'"];
    const team = String(req.query.team || '').trim();
    const gameName = String(req.query.gameName || '').trim();
    const filmSide = String(req.query.filmSide || '').trim().toLowerCase();
    if (team) { values.push(team); where.push(`c.team = $${values.length}`); }
    if (gameName) { values.push(gameName); where.push(`c.game_name = $${values.length}`); }
    if (filmSide) { values.push(filmSide); where.push(`c.film_side = $${values.length}`); }
    const result = await db.query(
      `SELECT c.id, c.team, c.game_name, c.original_name, c.created_at, c.labeled_at, c.film_side, c.use_for_ai,
              p.down, p.distance, p.hash, p.play_type, p.play_call,
              p.offense_formation, p.defense_formation, p.blitz, p.coverage,
              p.run_direction, p.pass_depth, p.completed, p.notes
       FROM clips c
       JOIN plays p ON p.clip_id = c.id
       WHERE ${where.join(' AND ')}
       ORDER BY c.team, c.game_name, c.created_at`, values
    );
    res.json(result.rows);
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
        COUNT(*) FILTER (WHERE status='needs_labeling' AND film_side='offense')::int AS needs_labeling,
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
