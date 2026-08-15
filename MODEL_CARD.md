# Football Copilot Model Card

## Production Model

**Model:** Model 2  
**Method:** Enhanced Poisson regression  
**Status:** Frozen production model


## Purpose

The model estimates Premier League match outcomes including:

- home win probability
- draw probability
- away win probability
- expected home goals
- expected away goals
- most likely scorelines


## Model Inputs

Model 2 uses pre-match historical football features.

These include:

### Five-match form

- home recent goals scored
- home recent goals conceded
- home recent points per game
- away recent goals scored
- away recent goals conceded
- away recent points per game

### Ten-match form

- home ten-match goals scored
- home ten-match goals conceded
- home ten-match points per game
- away ten-match goals scored
- away ten-match goals conceded
- away ten-match points per game

### Venue performance

- home-team home points per game
- home-team home scoring
- away-team away points per game
- away-team away scoring

### Season-to-date strength

- home season points per game
- home season goal difference per game
- away season points per game
- away season goal difference per game

### Relative strength

- recent PPG difference
- ten-match PPG difference
- season PPG difference
- home attack versus away defence
- away attack versus home defence


## Modelling Approach

Two Poisson regression models are trained:

```text
Football features
      │
      ├── Home-goals Poisson model
      │
      └── Away-goals Poisson model