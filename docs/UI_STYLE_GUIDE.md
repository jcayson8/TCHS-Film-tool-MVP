# TCHS Football Intelligence UI Style Guide

This document is the permanent source of truth for the application's user interface and user experience. Use it to guide every new workspace, component, and visual refinement.

## Project Vision

**TCHS Football Intelligence** is a football coaching platform focused on:

- Film study
- Analytics
- Scouting
- Game planning
- Reports
- Team management

The application should always feel like professional football coaching software. It should never feel like an AI annotation tool.

AI exists to assist coaches—not become the focus.

## Primary Navigation

The application has eight primary destinations:

1. Dashboard
2. Film Room
3. Library
4. Analytics
5. Scout / Game Plan
6. Reports
7. Teams
8. Settings

Do not add secondary workflows to the primary navigation.

## Purpose of Each Workspace

### Dashboard

**Purpose:** Provide an overview and show what needs attention.

The Dashboard should remain concise. Detailed film, tendencies, scouting, and management workflows belong in their dedicated workspaces.

### Film Room

**Purpose:**

- Watch film
- Annotate
- Track
- Build clips
- Train AI naturally through coaching work

The video is always the hero.

### Library

**Purpose:** Organize all games, practices, opponents, and clips.

### Analytics

**Purpose:** Help coaches understand tendencies and performance.

Every statistic should link directly to supporting film.

### Scout / Game Plan

**Purpose:** Prepare for upcoming opponents.

This workspace should:

- Show how defenses react to our formations, personnel, motion, and concepts
- Provide evidence-backed recommendations
- Connect recommendations directly to film

### Reports

**Purpose:** Generate professional scouting reports and exports.

### Teams

**Purpose:** Manage rosters, opponents, schedules, and season organization.

### Settings

**Purpose:** Configure the application and its workflows.

## Design Principles

- Coaches first.
- AI second.
- Video is the hero.
- Clean before clever.
- Reduce clutter.
- Use one primary action per screen.
- Use honest empty states instead of fake data.
- Every statistic should lead to film.
- Use consistent spacing everywhere.
- Use consistent typography.
- Use consistent card design.
- Use consistent navigation.
- Build responsive layouts.
- Maintain a professional coaching-software aesthetic.

## Design System

### Theme

Use a dark navy theme as the primary visual foundation.

### Accent

Use orange selectively for:

- The active navigation destination
- The primary action
- Selected tabs
- Important highlights

Orange should not dominate cards, headings, or metadata.

### Cards

Cards should use:

- A consistent border radius
- Consistent internal padding
- Consistent spacing between cards
- Subtle borders and restrained contrast
- Clear content hierarchy without decorative clutter

### Typography

Use:

- Large, clear page titles
- Medium-weight section titles
- Strong metric hierarchy
- Muted metadata and supporting text
- Compact, readable labels

### Buttons

#### Primary

Orange. Reserve for the main action on a screen.

#### Secondary

Dark and outlined. Use for important supporting actions.

#### Danger

Red. Use only for destructive actions.

### Sidebar

The sidebar should be compact and contain only the eight primary destinations.

### Icons

Icons throughout the application should follow one consistent visual style, size, and alignment.

## Current Implementation Order

- ✅ Shared Application Shell
- ✅ Dashboard

Next:

1. Film Room
2. Library
3. Analytics
4. Scout / Game Plan
5. Reports
6. Teams
7. Settings

## Long-Term Product Direction

```text
Detection
    ↓
Tracking
    ↓
Football Knowledge
    ↓
Recognition AI
    ↓
Football Event Engine
    ↓
Analytics
    ↓
Scout / Game Plan
    ↓
Reports
```

