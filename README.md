# TCHS Football Film Breakdown Tool

## Persistent AI learning

The app now creates a saved AI prediction when each clip is uploaded. When the coach saves the corrected label, the correction is added to persistent model statistics stored in PostgreSQL.

Stored permanently in PostgreSQL:
- every AI prediction
- every coach correction/training event
- learned per-team, per-game, and global category counts
- AI accuracy comparisons

Because learning is stored in the Render PostgreSQL database, replacing code or redeploying the web service does not reset training. Do not delete or replace the database unless you intend to erase the training history.

### Current model

`persistent-categorical-v1` is a real online categorical learner. It learns team/game tendencies and useful filename-token patterns for defensive formation, hash, direction, blitz, and coverage. It provides predictions and confidence scores immediately after enough labeled samples exist.

This version does not yet contain a computer-vision neural network that understands players directly from video pixels. That later model can use the same persistent tables and accuracy pipeline without losing the coach's saved labels.
