const TARGETS = {
  hash: { coachColumn: 'hash', aiColumn: 'ai_hash' },
  defense_formation: { coachColumn: 'defense_formation', aiColumn: 'ai_defense_formation' },
  blitz: { coachColumn: 'blitz', aiColumn: 'ai_blitz' },
  coverage: { coachColumn: 'coverage', aiColumn: 'ai_coverage' },
  run_direction: { coachColumn: 'run_direction', aiColumn: 'ai_run_direction' }
};

function tokensFromName(name = '') {
  return [...new Set(String(name).toLowerCase().replace(/\.mp4$/i, '').split(/[^a-z0-9]+/).filter(t => t.length >= 2).slice(0, 20))];
}

function normalizeValue(value) {
  if (value === null || value === undefined || value === '') return null;
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  return String(value).trim();
}

async function upsertCount(client, team, gameName, featureKey, target, targetValue) {
  await client.query(`
    INSERT INTO ai_model_counts (team, game_name, feature_key, target, target_value, sample_count, updated_at)
    VALUES ($1,$2,$3,$4,$5,1,NOW())
    ON CONFLICT (team, game_name, feature_key, target, target_value)
    DO UPDATE SET sample_count=ai_model_counts.sample_count+1, updated_at=NOW()`,
    [team, gameName, featureKey, target, targetValue]);
}

export async function learnFromCorrection(client, clip, labels) {
  const scopes = [
    [clip.team, clip.game_name || ''],
    [clip.team, ''],
    ['', '']
  ];
  const featureKeys = ['__prior__', ...tokensFromName(clip.original_name).map(t => `name:${t}`)];
  for (const [target, config] of Object.entries(TARGETS)) {
    const value = normalizeValue(labels[config.coachColumn]);
    if (value === null) continue;
    for (const [team, gameName] of scopes) {
      for (const featureKey of featureKeys) {
        await upsertCount(client, team, gameName, featureKey, target, value);
      }
    }
  }
  await client.query(`
    INSERT INTO ai_training_events (clip_id, team, game_name, labels, created_at)
    VALUES ($1,$2,$3,$4::jsonb,NOW())`,
    [clip.id, clip.team, clip.game_name || '', JSON.stringify(labels)]);
}

async function scoreTarget(client, clip, target) {
  const tokens = tokensFromName(clip.original_name).map(t => `name:${t}`);
  const scopes = [
    [clip.team, clip.game_name || '', 5],
    [clip.team, '', 3],
    ['', '', 1]
  ];
  const scores = new Map();
  let evidence = 0;
  for (const [team, gameName, weight] of scopes) {
    const features = ['__prior__', ...tokens];
    const result = await client.query(`
      SELECT feature_key, target_value, sample_count
      FROM ai_model_counts
      WHERE team=$1 AND game_name=$2 AND target=$3 AND feature_key = ANY($4::text[])`,
      [team, gameName, target, features]);
    for (const row of result.rows) {
      const featureWeight = row.feature_key === '__prior__' ? 1 : 2;
      const points = Number(row.sample_count) * weight * featureWeight;
      scores.set(row.target_value, (scores.get(row.target_value) || 0) + points);
      evidence += points;
    }
  }
  if (!scores.size) return { value: null, confidence: null, evidence: 0 };
  const ranked = [...scores.entries()].sort((a,b) => b[1]-a[1]);
  const [value, top] = ranked[0];
  const total = ranked.reduce((sum, [,score]) => sum + score, 0);
  return { value, confidence: Math.round((top / total) * 1000) / 10, evidence };
}

export async function predictClip(client, clip) {
  const prediction = {};
  const confidences = [];
  for (const target of Object.keys(TARGETS)) {
    const scored = await scoreTarget(client, clip, target);
    prediction[target] = target === 'blitz' && scored.value !== null ? scored.value === 'true' : scored.value;
    prediction[`${target}_confidence`] = scored.confidence;
    if (scored.confidence !== null) confidences.push(scored.confidence);
  }
  prediction.overall_confidence = confidences.length
    ? Math.round(confidences.reduce((a,b)=>a+b,0) / confidences.length * 10) / 10
    : null;
  prediction.model_version = 'persistent-categorical-v1';
  prediction.predicted_at = new Date().toISOString();
  return prediction;
}

export async function savePrediction(client, clip, prediction) {
  await client.query(`
    INSERT INTO ai_predictions (
      clip_id, model_version, prediction, overall_confidence, created_at
    ) VALUES ($1,$2,$3::jsonb,$4,NOW())
    ON CONFLICT (clip_id) DO UPDATE SET
      model_version=EXCLUDED.model_version,
      prediction=EXCLUDED.prediction,
      overall_confidence=EXCLUDED.overall_confidence,
      created_at=NOW()`,
    [clip.id, prediction.model_version, JSON.stringify(prediction), prediction.overall_confidence]);

  await client.query(`
    INSERT INTO plays (
      clip_id, team, game_name, clip, label_source,
      ai_hash, ai_defense_formation, ai_blitz, ai_coverage, ai_run_direction, ai_confidence, updated_at
    ) VALUES ($1,$2,$3,$4,'ai',$5,$6,$7,$8,$9,$10,NOW())
    ON CONFLICT (clip_id) DO UPDATE SET
      ai_hash=EXCLUDED.ai_hash,
      ai_defense_formation=EXCLUDED.ai_defense_formation,
      ai_blitz=EXCLUDED.ai_blitz,
      ai_coverage=EXCLUDED.ai_coverage,
      ai_run_direction=EXCLUDED.ai_run_direction,
      ai_confidence=EXCLUDED.ai_confidence,
      updated_at=NOW()`,
    [clip.id, clip.team, clip.game_name || '', clip.original_name,
      prediction.hash, prediction.defense_formation, prediction.blitz,
      prediction.coverage, prediction.run_direction, prediction.overall_confidence]);
}

export { TARGETS };
