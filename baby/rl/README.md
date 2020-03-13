# A2C

Code inspired by A2C baselines:
https://github.com/openai/baselines/blob/master/baselines/a2c/a2c.py

Modifications:
* ADR: Domain randomization
* Short-term loss: Loss weighted for short-term reward
* Short-term / long-term evolution loss: Update loss while training to mix lossses