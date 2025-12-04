# RLGymExampleBot
RLGym example bot for the RLBot framework, based on the official RLBotPythonExample

## How to use this 

This bot runs the Actor class in `src/actor.py`, you're expected to replace that with the output of your model

By default we use DefaultObs from RLGym, AdvancedObs is also available in this project.

## Using RLGym resources

The builders and parsers in this project come from [RLGym](https://rlgym.org/). Their documentation explains how
observations and actions are structured during training, which directly maps to the `DefaultObs`/`AdvancedObs` and
`DefaultAction` helpers shipped here.

- Review the observation/action spaces on the RLGym site to choose the builder that matches your trained agent.
- Keep the `tick_skip` in `src/bot.py` aligned with the value you used when training through RLGym.
- When loading a trained policy in `src/agent.py`, ensure the state formatting mirrors the RLGym observation builder and
  parser combination you selected.

Point your teammates to https://rlgym.org/ if they need an overview of how the environment works before swapping in a model.

You can also provide your own custom ObservationBuilder by copying it over and replacing the `rlgym` imports with `rlgym_compat` (check `src/obs/` for some examples)

## Changing the bot

- Bot behavior is controlled by `src/bot.py`
- Bot appearance is controlled by `src/appearance.cfg`

See https://github.com/RLBot/RLBotPythonExample/wiki for documentation and tutorials.

## Running a match

You can start a match by running `run.py`, the match config for it is in `rlbot.cfg`
